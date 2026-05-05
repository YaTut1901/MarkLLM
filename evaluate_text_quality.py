"""Two-stage watermarked text quality evaluation.

``prepare`` — load LM + watermarks, generate truncated continuations against C4
(watermarked or pseudo-algorithm ``Unwatermarked`` via ``generate_unwatermarked_text``),
save JSON artifact.

``analyze`` — load artifact only into metric calculators (fresh LM load on chosen
devices): PPL, log diversity, BLEU, ROUGE-1/2/L, BERTScore F1.

Example::

    python evaluate_text_quality.py prepare --runs 20 --algorithms KGW Unwatermarked --output wm.json
    python evaluate_text_quality.py analyze --input wm.json --report tq_report.md \\
        --ppl-device cpu --bertscore-device cpu
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluation.dataset import C4Dataset
from evaluation.tools.text_editor import TruncatePromptTextEditor
from evaluation.tools.text_quality_analyzer import PPLCalculator
from utils.transformers_config import TransformersConfig
from watermark.auto_watermark import AutoWatermark

ARTIFACT_VERSION = 1

# Matches detection-side truncation for continuation-vs-natural comparisons.
DEFAULT_MIN_WORDS_GATE = 4

DEFAULT_MODEL_PATH = "models/Qwen2.5-Coder-1.5B"
DEFAULT_DATASET_PATH = "dataset/c4/processed_c4.json"
DEFAULT_OUTPUT_PREPARE = "text_quality_artifacts.json"
DEFAULT_REPORT_ANALYZE = "text_quality_report.md"
DEFAULT_BERTSCORE_MODEL = "roberta-large"

UNWATERMARKED_ALGO = "Unwatermarked"

ALGORITHM_SPECS: List[Dict[str, object]] = [
    {"name": "KGW",       "reverse": False},
    {"name": "EXP",       "reverse": True},
    {"name": "Unbiased",  "reverse": True},
    {"name": "SEMSTAMP",  "reverse": False},
    {"name": "SynthID",   "reverse": False},
    {"name": "SegmentWM", "reverse": False},
    {
        "name": "FairOzeWM",
        "load_name": "ETHWatermark",
        "config_path": "config/ETHWatermark.json",
        "reverse": False,
    },
    # Baseline: same LM + gen kwargs, no watermark (implemented via BaseWatermark.generate_unwatermarked_text).
    {"name": UNWATERMARKED_ALGO, "reverse": False, "unwatermarked_baseline": True},
]


def _wm_spec_load_params(spec: Dict[str, object]) -> tuple[str, str]:
    load_name = str(spec.get("load_name", spec["name"]))
    config_path = str(spec.get("config_path", f"config/{load_name}.json"))
    return load_name, config_path


def _normalize_algorithm_name(name: str) -> str:
    if name.strip().lower() == "unwatermarked":
        return UNWATERMARKED_ALGO
    return name


def _watermarked_algorithm_names() -> List[str]:
    return [str(s["name"]) for s in ALGORITHM_SPECS if not s.get("unwatermarked_baseline")]


def free_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _tq_log(msg: str) -> None:
    print(f"[text-quality] {msg}", flush=True)


def _tq_preview(text: object, max_len: int = 160) -> str:
    s = "" if text is None else str(text)
    one_line = " ".join(s.split())
    if len(one_line) > max_len:
        return one_line[: max_len - 3] + "..."
    return one_line if one_line else "(empty)"


def _ppl_token_length(tokenizer: Any, text: str) -> int:
    try:
        enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
        return int(enc.shape[1])
    except Exception:
        return -1


def _format_float_cell(value: object, ndigits: int = 4) -> str:
    if value is None:
        return "—"
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(x):
        return "—"
    return f"{x:.{ndigits}f}"


def _mean_finite(values: List[float]) -> Optional[float]:
    xs = [float(x) for x in values if math.isfinite(float(x))]
    if not xs:
        return None
    return sum(xs) / len(xs)


def aggregate_metrics_for_algorithm(
    hypotheses: List[str],
    references: List[str],
    *,
    algorithm_name: str,
    ppl_calc: Optional[PPLCalculator],
    bert_scorer: Optional[object],
    min_words_gate: int,
    log_samples: bool,
) -> Dict[str, object]:
    """Mean metrics; hypothesis = truncated continuation, reference = C4 natural."""
    from evaluation.tools.text_quality_analyzer import (
        BLEUCalculator,
        LogDiversityAnalyzer,
        ROUGE1Calculator,
        ROUGE2Calculator,
        ROUGELCalculator,
    )

    nh, nr = len(hypotheses), len(references)
    _tq_log(
        f"{algorithm_name}: aggregate — hypotheses={nh}, references={nr}, "
        f"min_words={min_words_gate}, per_sample_logs={'on' if log_samples else 'off'}, "
        f"ppl={'on' if ppl_calc else 'off'}",
    )
    if nh != nr:
        _tq_log(f"{algorithm_name}: WARNING length mismatch; zip uses min={min(nh, nr)}")

    log_div_an = LogDiversityAnalyzer()
    bleu_calc = BLEUCalculator()
    r1_calc = ROUGE1Calculator()
    r2_calc = ROUGE2Calculator()
    rl_calc = ROUGELCalculator()

    ppls: List[float] = []
    log_divs: List[float] = []
    bleus: List[float] = []
    r1s: List[float] = []
    r2s: List[float] = []
    rls: List[float] = []
    usable_hyp: List[str] = []
    usable_ref: List[str] = []

    n_skip_short = 0
    n_skip_ppl_err = 0
    n_logdiv_failed = 0
    n_ok = 0

    tok = getattr(ppl_calc, "tokenizer", None) if ppl_calc is not None else None

    for idx, (hyp, ref) in enumerate(zip(hypotheses, references)):
        hyp_s = (hyp or "").strip()
        ref_s = (ref or "").strip()
        wc = len(hyp_s.split())
        ref_wc = len(ref_s.split())
        n_tok = _ppl_token_length(tok, hyp_s) if tok is not None else -1

        if wc < min_words_gate:
            n_skip_short += 1
            if log_samples:
                _tq_log(
                    f"{algorithm_name}[{idx}] SKIP short_words wc={wc} tokens≈{n_tok} "
                    f"ref_wc={ref_wc} hyp={_tq_preview(hyp_s)!r}",
                )
            continue

        if ppl_calc is not None:
            try:
                ppl_val = float(ppl_calc.analyze(hyp_s))
                ppls.append(ppl_val)
            except Exception as exc:  # noqa: BLE001
                n_skip_ppl_err += 1
                if log_samples:
                    _tq_log(
                        f"{algorithm_name}[{idx}] SKIP PPL {type(exc).__name__}: {exc} "
                        f"wc={wc} tokens≈{n_tok} hyp={_tq_preview(hyp_s)!r}",
                    )
                    _tq_log(traceback.format_exc())
                continue
        else:
            ppl_val = float("nan")

        try:
            log_divs.append(float(log_div_an.analyze(hyp_s)))
        except Exception as exc:  # noqa: BLE001
            n_logdiv_failed += 1
            log_divs.append(float("nan"))
            if log_samples:
                _tq_log(f"{algorithm_name}[{idx}] log_div nan: {type(exc).__name__}: {exc}")

        bleus.append(float(bleu_calc.analyze(hyp_s, ref_s)))
        r1s.append(float(r1_calc.analyze(hyp_s, ref_s)))
        r2s.append(float(r2_calc.analyze(hyp_s, ref_s)))
        rls.append(float(rl_calc.analyze(hyp_s, ref_s)))
        usable_hyp.append(hyp_s)
        usable_ref.append(ref_s)
        n_ok += 1
        if log_samples and ppl_calc is not None:
            _tq_log(
                f"{algorithm_name}[{idx}] OK PPL={ppl_val:.4f} wc={wc} tokens≈{n_tok}",
            )

    bert_f1_mean: Optional[float] = None
    bert_err: Optional[str] = None
    if usable_hyp and usable_ref and bert_scorer is not None:
        try:
            _tq_log(f"{algorithm_name}: BERTScore n={len(usable_hyp)}")
            _, _, f1 = bert_scorer.score(usable_hyp, usable_ref)
            bert_f1_mean = _mean_finite([float(x) for x in f1.tolist()])
            _tq_log(f"{algorithm_name}: BERTScore mean_f1={bert_f1_mean}")
        except Exception as exc:  # noqa: BLE001
            bert_f1_mean = None
            bert_err = f"{type(exc).__name__}: {exc}"
            _tq_log(f"{algorithm_name}: BERTScore FAILED {bert_err}\n{traceback.format_exc()}")
    elif bert_scorer is None:
        bert_err = "skipped (--skip-bertscore)"
        _tq_log(f"{algorithm_name}: BERTScore skipped")
    else:
        _tq_log(f"{algorithm_name}: BERTScore skipped (no pairs usable_hyp={len(usable_hyp)})")

    counted_for_bleu_rouge = n_ok
    _tq_log(
        f"{algorithm_name}: done — bleu_rouge_rows={counted_for_bleu_rouge}, "
        f"ppl_values={len(ppls)}, skip_short={n_skip_short}, skip_ppl_err={n_skip_ppl_err}, "
        f"logdiv_exc={n_logdiv_failed}, ppl_disabled={ppl_calc is None}",
    )

    return {
        "ppl_mean": _mean_finite(ppls),
        "log_diversity_mean": _mean_finite([x for x in log_divs if math.isfinite(x)]),
        "bleu_mean": _mean_finite(bleus),
        "rouge1_mean": _mean_finite(r1s),
        "rouge2_mean": _mean_finite(r2s),
        "rougeL_mean": _mean_finite(rls),
        "bertscore_f1_mean": bert_f1_mean,
        "text_quality_samples_used": len(ppls),
        "text_quality_debug": {
            "rows_bleu_rouge": counted_for_bleu_rouge,
            "skip_short_word_count": n_skip_short,
            "skip_ppl_error": n_skip_ppl_err,
            "ppl_disabled": ppl_calc is None,
            "log_diversity_nan_from_exception": n_logdiv_failed,
            "bertscore_error": bert_err,
        },
    }


def _build_transformers_config(
    model: torch.nn.Module,
    tokenizer: Any,
    device: str,
    max_new_tokens: int,
    min_length: int,
) -> TransformersConfig:
    return TransformersConfig(
        model=model,
        tokenizer=tokenizer,
        vocab_size=model.config.vocab_size,
        device=device,
        max_new_tokens=max_new_tokens,
        min_length=min_length,
        do_sample=True,
        no_repeat_ngram_size=4,
    )


def cmd_prepare(ns: argparse.Namespace) -> None:
    spec_by_name = {str(s["name"]): s for s in ALGORITHM_SPECS}
    default_algolist = _watermarked_algorithm_names()
    selected_raw = ns.algorithms or default_algolist
    selected = [_normalize_algorithm_name(a) for a in selected_raw]
    wm_names = _watermarked_algorithm_names()

    if UNWATERMARKED_ALGO in selected and ns.unwatermarked_via not in wm_names:
        print(
            f"prepare: error — --unwatermarked-via must be a watermark algorithm "
            f"(one of {wm_names}), got {ns.unwatermarked_via!r}",
            flush=True,
        )
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[ns.dtype]

    print(f"prepare: loading LM {ns.model_path} ({ns.dtype}, {device})", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(ns.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        ns.model_path,
        trust_remote_code=True,
        torch_dtype=dtype if device == "cuda" else torch.float32,
    ).to(device).eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    min_len = ns.min_length if ns.min_length is not None else max(ns.max_new_tokens - 30, 30)
    transformers_config = _build_transformers_config(
        model, tokenizer, device, ns.max_new_tokens, min_len,
    )

    dataset = C4Dataset(ns.dataset_path, max_samples=ns.runs)
    print(
        f"prepare: C4 prompts={dataset.prompt_nums} natural_texts={dataset.natural_text_nums}",
        flush=True,
    )

    artifact: Dict[str, Any] = {
        "artifact_version": ARTIFACT_VERSION,
        "stage": "prepare",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "model_path": ns.model_path,
            "dataset_path": ns.dataset_path,
            "runs": ns.runs,
            "dtype": ns.dtype,
            "max_new_tokens": ns.max_new_tokens,
            "min_length": ns.min_length,
            "computed_min_length": min_len,
            "device_used": device,
            "algorithms_order": list(selected),
            **(
                {"unwatermarked_via": ns.unwatermarked_via}
                if UNWATERMARKED_ALGO in selected
                else {}
            ),
        },
        "algorithms": {},
    }

    editor = TruncatePromptTextEditor()

    for algo in selected:
        if algo not in spec_by_name:
            print(f"prepare: skip unknown algorithm {algo!r}", flush=True)
            continue

        print(f"\nprepare: === {algo} ===", flush=True)
        t0 = time.time()

        spec = spec_by_name[algo]
        load_name, config_path = _wm_spec_load_params(spec)

        if algo == UNWATERMARKED_ALGO:
            provider = ns.unwatermarked_via
            prov_spec = spec_by_name[provider]
            pl_name, pc_path = _wm_spec_load_params(prov_spec)
            print(
                f"prepare: Unwatermarked baseline via AutoWatermark.load({pl_name!r}, {pc_path!r}) "
                f"→ generate_unwatermarked_text",
                flush=True,
            )
            watermark = AutoWatermark.load(
                pl_name,
                algorithm_config=pc_path,
                transformers_config=transformers_config,
            )
        else:
            print(
                f"prepare: loading {algo!r} as {load_name!r} ({config_path})",
                flush=True,
            )
            watermark = AutoWatermark.load(
                load_name,
                algorithm_config=config_path,
                transformers_config=transformers_config,
            )

        rows: List[Dict[str, Any]] = []
        for i in tqdm(range(dataset.prompt_nums), desc=f"{algo} generate", leave=True):
            prompt = dataset.get_prompt(i)
            if algo == UNWATERMARKED_ALGO:
                raw = watermark.generate_unwatermarked_text(prompt)
            else:
                raw = watermark.generate_watermarked_text(prompt)
            hypothesis = editor.edit(raw, prompt)
            row: Dict[str, Any] = {
                "index": i,
                "prompt": prompt,
                "hypothesis": hypothesis,
                "reference": dataset.get_natural_text(i),
            }
            if ns.save_raw_generation:
                row["raw_generation"] = raw
            rows.append(row)

        block: Dict[str, Any] = {
            "reverse": bool(spec_by_name[algo]["reverse"]),
            "elapsed_seconds_prepare": round(time.time() - t0, 2),
            "samples": rows,
        }
        if algo == UNWATERMARKED_ALGO:
            block["unwatermarked_via"] = ns.unwatermarked_via

        del watermark
        free_memory()

        artifact["algorithms"][algo] = block
        print(f"prepare: {algo} saved {len(rows)} samples ({time.time()-t0:.1f}s)", flush=True)

    out_path = Path(ns.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2))
    print(f"\nprepare: artifact written to {out_path}", flush=True)

    del model
    free_memory()


def cmd_analyze(ns: argparse.Namespace) -> None:
    path = Path(ns.input)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("artifact_version") != ARTIFACT_VERSION:
        print(f"analyze: warning artifact_version={raw.get('artifact_version')!r} expected {ARTIFACT_VERSION}")

    cfg_art = raw.get("config") or {}
    model_path = ns.model_path or cfg_art.get("model_path") or DEFAULT_MODEL_PATH
    min_words = ns.min_words_gate

    results: Dict[str, Dict[str, object]] = {}
    failures: Dict[str, str] = {}

    # --- BERTScore (optional) ---
    bert_scorer = None
    if not ns.skip_bertscore:
        from bert_score import BERTScorer

        bert_dev = ns.bertscore_device or ("cuda" if torch.cuda.is_available() else "cpu")
        if bert_dev == "cuda" and not torch.cuda.is_available():
            bert_dev = "cpu"
        print(f"analyze: BERTScore model={ns.bertscore_model} device={bert_dev}", flush=True)
        bert_scorer = BERTScorer(
            model_type=ns.bertscore_model,
            num_layers=8,
            batch_size=32,
            nthreads=4,
            all_layers=False,
            idf=False,
            device=bert_dev,
            rescale_with_baseline=False,
            lang="en",
        )

    # --- LM for PPL (optional) ---
    ppl_calc: Optional[PPLCalculator] = None
    if not ns.skip_ppl:
        ppl_dev = ns.ppl_device
        if ppl_dev == "cuda" and not torch.cuda.is_available():
            ppl_dev = "cpu"
            print("analyze: ppl_device cuda unavailable → cpu", flush=True)

        dtype_key = ns.dtype
        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype_key]
        print(f"analyze: loading LM for PPL {model_path} ({dtype_key}, {ppl_dev})", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        lm = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=dtype if ppl_dev == "cuda" else torch.float32,
        ).to(ppl_dev).eval()
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        ppl_calc = PPLCalculator(lm, tokenizer, device=ppl_dev)
    else:
        lm = None
        tokenizer = None
        print("analyze: PPL skipped (--skip-ppl)", flush=True)

    algorithms_block = raw.get("algorithms") or {}
    log_samples = not ns.quiet

    for algo_name, block in algorithms_block.items():
        if not isinstance(block, dict):
            failures[algo_name] = "invalid algorithm block"
            continue
        samples = block.get("samples") or []
        hyps = [str(s.get("hypothesis", "") or "") for s in samples]
        refs = [str(s.get("reference", "") or "") for s in samples]
        try:
            results[algo_name] = aggregate_metrics_for_algorithm(
                hyps,
                refs,
                algorithm_name=algo_name,
                ppl_calc=ppl_calc,
                bert_scorer=bert_scorer,
                min_words_gate=min_words,
                log_samples=log_samples,
            )
            print(
                f"{algo_name}: PPL_mean={_format_float_cell(results[algo_name].get('ppl_mean'), 3)} "
                f"BLEU={_format_float_cell(results[algo_name].get('bleu_mean'))} "
                f"BERT-F1={_format_float_cell(results[algo_name].get('bertscore_f1_mean'))}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            failures[algo_name] = f"{type(exc).__name__}: {exc}"
            print(f"!! {algo_name} analyze failed: {exc}\n{traceback.format_exc()}", flush=True)

    if bert_scorer is not None:
        del bert_scorer
    if lm is not None:
        del lm
    free_memory()

    _write_analyze_report(ns.report, ns, raw, results, failures)


def _write_analyze_report(
    report_path: str,
    ns: argparse.Namespace,
    artifact: Dict[str, Any],
    results: Dict[str, Dict[str, object]],
    failures: Dict[str, str],
) -> None:
    out = Path(report_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    cfg_art = artifact.get("config") or {}
    model_path = ns.model_path or cfg_art.get("model_path")

    payload_common = {
        "artifact": str(ns.input),
        "artifact_created_at": artifact.get("created_at"),
        "prepare_config": cfg_art,
        "analyze_model_path": model_path,
        "dtype": ns.dtype,
        "ppl_device": ns.ppl_device if not ns.skip_ppl else None,
        "skip_ppl": ns.skip_ppl,
        "bertscore_model": None if ns.skip_bertscore else ns.bertscore_model,
        "bertscore_device": ns.bertscore_device,
        "skip_bertscore": ns.skip_bertscore,
        "min_words_gate": ns.min_words_gate,
    }

    if out.suffix.lower() == ".json":
        out.write_text(
            json.dumps(
                {"config": payload_common, "results": results, "failures": failures},
                indent=2,
                ensure_ascii=False,
            )
        )
        print(f"analyze: JSON report → {out}", flush=True)
        return

    lines = [
        "# Text quality report (analyze stage)\n",
        f"- Artifact: `{ns.input}`\n",
        f"- LM (PPL): `{model_path}` dtype=`{ns.dtype}` ppl_device=`{payload_common['ppl_device']}` "
        f"skip_ppl=`{ns.skip_ppl}`\n",
    ]
    if not ns.skip_bertscore:
        bd = ns.bertscore_device or ("cuda" if torch.cuda.is_available() else "cpu")
        lines.append(f"- BERTScore: `{ns.bertscore_model}` device=`{bd}`\n")
    else:
        lines.append("- BERTScore: skipped\n")
    lines.append(f"- Min words gate (PPL path): `{ns.min_words_gate}`\n\n")

    for name, r in results.items():
        lines.append(f"### {name}\n\n")
        tq = r
        n_used = int(tq.get("text_quality_samples_used") or 0)
        dbg = tq.get("text_quality_debug") if isinstance(tq.get("text_quality_debug"), dict) else {}
        n_bleu = dbg.get("rows_bleu_rouge", "—")
        lines.append(
            f"PPL aggregations over **{n_used}** samples (≥{ns.min_words_gate} words); "
            f"BLEU/ROUGE rows **{n_bleu}**.\n\n",
        )
        lines.append(
            "| PPL | Log diversity | BLEU | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 |\n"
            "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n",
        )
        cells = [
            _format_float_cell(tq.get("ppl_mean"), 3),
            _format_float_cell(tq.get("log_diversity_mean")),
            _format_float_cell(tq.get("bleu_mean")),
            _format_float_cell(tq.get("rouge1_mean")),
            _format_float_cell(tq.get("rouge2_mean")),
            _format_float_cell(tq.get("rougeL_mean")),
            _format_float_cell(tq.get("bertscore_f1_mean")),
        ]
        lines.append("| " + " | ".join(cells) + " |\n\n")
        if dbg:
            lines.append(
                f"*Debug:* rows_bleu_rouge=`{dbg.get('rows_bleu_rouge')}`, "
                f"skip_short=`{dbg.get('skip_short_word_count')}`, skip_ppl=`{dbg.get('skip_ppl_error')}`, "
                f"ppl_disabled=`{dbg.get('ppl_disabled')}`, logdiv_exc=`{dbg.get('log_diversity_nan_from_exception')}`",
            )
            if dbg.get("bertscore_error"):
                lines.append(f", bert=`{dbg.get('bertscore_error')}`")
            lines.append("\n\n")

    if failures:
        lines.append("## Failures\n\n")
        for k, v in failures.items():
            lines.append(f"- **{k}**: `{v}`\n")

    out.write_text("".join(lines))
    print(f"analyze: Markdown report → {out}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Watermarked text quality: prepare → analyze")
    sub = parser.add_subparsers(dest="command", required=True)

    p_prep = sub.add_parser(
        "prepare",
        help="Generate watermarked or Unwatermarked baselines and save JSON artifact",
    )
    p_prep.add_argument("--runs", type=int, default=50)
    p_prep.add_argument(
        "--algorithms",
        nargs="+",
        default=None,
        help=f"Subset / order. Include '{UNWATERMARKED_ALGO}' or 'unwatermarked' for no-watermark "
             "generation (needs --unwatermarked-via). Default: all watermark algos except baseline.",
    )
    p_prep.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    p_prep.add_argument("--dataset-path", type=str, default=DEFAULT_DATASET_PATH)
    p_prep.add_argument("--output", type=str, default=DEFAULT_OUTPUT_PREPARE)
    p_prep.add_argument("--max-new-tokens", type=int, default=200)
    p_prep.add_argument("--min-length", type=int, default=None)
    p_prep.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    p_prep.add_argument(
        "--save-raw-generation",
        action="store_true",
        help="Store full model output before prompt truncation in each sample.",
    )
    p_prep.add_argument(
        "--unwatermarked-via",
        type=str,
        default="KGW",
        metavar="NAME",
        help=f"When '{UNWATERMARKED_ALGO}' is in --algorithms, load this watermark (e.g. KGW) only "
             "to call BaseWatermark.generate_unwatermarked_text; gen kwargs stay identical.",
    )
    p_prep.set_defaults(func=cmd_prepare)

    p_an = sub.add_parser("analyze", help="Load artifact and compute metrics (fresh process)")
    p_an.add_argument("--input", type=str, required=True, help="JSON from prepare")
    p_an.add_argument("--report", type=str, default=DEFAULT_REPORT_ANALYZE)
    p_an.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="LM for PPL (default: path stored in artifact)",
    )
    p_an.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    p_an.add_argument(
        "--ppl-device",
        type=str,
        choices=["cuda", "cpu"],
        default="cpu",
        help="Device for LM during PPL only (default cpu to reduce VRAM spikes).",
    )
    p_an.add_argument("--skip-ppl", action="store_true", help="Skip PPL; BLEU/ROUGE/BERT still run.")
    p_an.add_argument("--bertscore-model", type=str, default=DEFAULT_BERTSCORE_MODEL)
    p_an.add_argument("--bertscore-device", type=str, choices=["cuda", "cpu"], default=None)
    p_an.add_argument("--skip-bertscore", action="store_true")
    p_an.add_argument("--min-words-gate", type=int, default=DEFAULT_MIN_WORDS_GATE)
    p_an.add_argument("--quiet", action="store_true", help="Fewer per-sample log lines")
    p_an.set_defaults(func=cmd_analyze)

    ns = parser.parse_args()
    ns.func(ns)


if __name__ == "__main__":
    main()
