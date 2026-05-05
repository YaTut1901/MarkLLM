"""Watermark detection metrics under **text attacks** (robustness evaluation).

Mirrors ``evaluate_detection_metrics.py`` but applies a post-generation attack
to watermarked continuations only (after ``TruncatePromptTextEditor``).
Unwatermarked side uses the same setup as the baseline script (no attack on
natural / generated unwatermarked text).

Attacks are wired incrementally. Currently supported:
  - ``context_synonym`` — ``ContextAwareSynonymSubstitution`` (BERT MLM + WordNet).
  - ``word_deletion`` — ``WordDeletion`` (random word drops; no extra models).
  - ``paraphrase`` — ``ParrotT5Paraphraser`` (T5 seq2seq, default ``prithivida/parrot_paraphraser_on_T5``).
  - ``back_translation`` — default **local NLLB** (:class:`LocalNLLBBackTranslationEditor`, e.g.
    ``facebook/nllb-200-distilled-600M``); optional ``--bt-backend online`` uses
    ``BackTranslationTextEditor`` / ``translate`` (MyMemory **500-char** query cap).

For Parrot checkpoints that ship only PyTorch ``.bin`` weights, transformers may otherwise spawn a
background safetensors-conversion thread that calls the HF *discussions* API (often **403** if
discussions are disabled on the repo). This script sets ``DISABLE_SAFETENSORS_CONVERSION=1``
when ``--attack paraphrase`` unless you already exported that variable.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from translate import Translator
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BertForMaskedLM,
    BertTokenizer,
)
from transformers.models.nllb import NllbTokenizer

from evaluation.dataset import C4Dataset
from evaluation.pipelines.detection import (
    DetectionPipelineReturnType,
    UnWatermarkedTextDetectionPipeline,
    WatermarkedTextDetectionPipeline,
)
from evaluation.tools.success_rate_calculator import DynamicThresholdSuccessRateCalculator
from evaluation.tools.text_editor import (
    BackTranslationTextEditor,
    ContextAwareSynonymSubstitution,
    LocalNLLBBackTranslationEditor,
    ParrotT5Paraphraser,
    TruncatePromptTextEditor,
    WordDeletion,
)
from utils.transformers_config import TransformersConfig
from watermark.auto_watermark import AutoWatermark


METRIC_LABELS: List[str] = ["TPR", "TNR", "FPR", "FNR", "F1", "ACC"]

ALGORITHM_SPECS: List[Dict[str, object]] = [
    {"name": "KGW", "reverse": False},
    {"name": "EXP", "reverse": True},
    {"name": "Unbiased", "reverse": True},
    {"name": "SEMSTAMP", "reverse": False},
    {"name": "SynthID", "reverse": False},
    {"name": "SegmentWM", "reverse": False},
    {
        "name": "FairOzeWM",
        "load_name": "ETHWatermark",
        "config_path": "config/ETHWatermark.json",
        "reverse": False,
    },
]


def _wm_spec_load_params(spec: Dict[str, object]) -> tuple[str, str]:
    load_name = str(spec.get("load_name", spec["name"]))
    config_path = str(spec.get("config_path", f"config/{load_name}.json"))
    return load_name, config_path


DEFAULT_MODEL_PATH = "models/Qwen2.5-Coder-1.5B"
DEFAULT_DATASET_PATH = "dataset/c4/processed_c4.json"
DEFAULT_REPORT_PATH = "attack_metrics_report.md"

ATTACK_CHOICES = ["context_synonym", "word_deletion", "paraphrase", "back_translation"]

DEFAULT_BERT_MLM = "bert-base-uncased"
DEFAULT_PARAPHRASE_MODEL = "prithivida/parrot_paraphraser_on_T5"
# Reasonable VRAM footprint; bilingual quality trade-off vs 1.3B distilled.
DEFAULT_BT_NLLB_MODEL = "facebook/nllb-200-distilled-600M"


def _format_metric_cell(value: object) -> str:
    if value is None:
        return "—"
    return f"{float(value):.3f}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate watermark detection metrics under post-generation text attacks.",
    )
    p.add_argument(
        "--attack",
        choices=ATTACK_CHOICES,
        required=True,
        help="Post-generation attack applied to watermarked generations only.",
    )
    p.add_argument("--runs", type=int, default=50, help="Prompts per watermark algorithm.")
    p.add_argument(
        "--algorithms",
        nargs="+",
        default=None,
        metavar="NAME",
        help=f"Subset/order. Default: all of {[s['name'] for s in ALGORITHM_SPECS]}.",
    )
    p.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    p.add_argument("--dataset-path", type=str, default=DEFAULT_DATASET_PATH)
    p.add_argument("--report", type=str, default=DEFAULT_REPORT_PATH)
    p.add_argument(
        "--unwm-source",
        choices=["natural", "generated"],
        default="natural",
        help="Unwatermarked text source (same as baseline detection script).",
    )
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--min-length", type=int, default=None)
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    p.add_argument("--save-scores", action="store_true")
    p.add_argument("--seed", type=int, default=42, help="RNG seed (Python, NumPy, Torch).")

    g = p.add_argument_group("context_synonym (ContextAwareSynonymSubstitution)")
    g.add_argument(
        "--synonym-ratio",
        type=float,
        default=0.5,
        help="Upper bound on fraction of words targeted for replacement (see TextEditor).",
    )
    g.add_argument(
        "--bert-masked-lm",
        type=str,
        default=DEFAULT_BERT_MLM,
        help="HF repo id or path for BERT masked LM.",
    )
    g.add_argument(
        "--bert-device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Device for BERT MLM. Default: same as LM (usually cuda if available).",
    )

    g2 = p.add_argument_group("word_deletion (WordDeletion)")
    g2.add_argument(
        "--deletion-ratio",
        type=float,
        default=0.3,
        help="Per-word deletion probability (each word kept iff random >= ratio; see TextEditor).",
    )

    g3 = p.add_argument_group("paraphrase (ParrotT5Paraphraser — Parrot T5 on HF)")
    g3.add_argument(
        "--paraphrase-model",
        type=str,
        default=DEFAULT_PARAPHRASE_MODEL,
        help="HF repo/path for Parrot-style T5 paraphraser (tokenizer loaded from the same repo).",
    )
    g3.add_argument(
        "--paraphrase-device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Device for paraphrase model (default: cuda if available, else cpu).",
    )
    g3.add_argument(
        "--paraphrase-device-map",
        type=str,
        choices=["none", "auto"],
        default="none",
        help="Use 'auto' only if you need HF sharding; small Parrot T5 fits on one device with 'none'.",
    )
    g3.add_argument(
        "--paraphrase-sent-interval",
        type=int,
        default=1,
        help="NLTK sentence windows per paraphrase call (default 1).",
    )
    g3.add_argument(
        "--paraphrase-num-beams",
        type=int,
        default=5,
        help="Beam width for generate (default 5; set 1 to disable beam search).",
    )
    g3.add_argument(
        "--paraphrase-max-length",
        type=int,
        default=64,
        help="max_length for paraphrase generate (model card suggests short outputs).",
    )
    g3.add_argument(
        "--paraphrase-do-sample",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, use sampling (with top_p/top_k); default false for beam search.",
    )
    g3.add_argument(
        "--paraphrase-top-p",
        type=float,
        default=0.9,
        help="top_p when --paraphrase-do-sample is true.",
    )
    g3.add_argument(
        "--paraphrase-top-k",
        type=int,
        default=None,
        help="Optional top_k when sampling.",
    )

    g4 = p.add_argument_group("back_translation (local NLLB by default, or HTTP via `translate`)")
    g4.add_argument(
        "--bt-backend",
        type=str,
        choices=["nllb", "online"],
        default="nllb",
        help="Local NLLB (no 500-char API cap; needs GPU VRAM/RAM). 'online': translate package/MyMemory.",
    )
    g4.add_argument(
        "--bt-source-lang",
        type=str,
        default="en",
        help="[online only] ISO-style short codes for `translate`. e.g. en, zh.",
    )
    g4.add_argument(
        "--bt-intermediate-lang",
        type=str,
        default="zh",
        help="[online only] Pivot language short code.",
    )
    g4.add_argument(
        "--bt-nllb-model",
        type=str,
        default=DEFAULT_BT_NLLB_MODEL,
        help="HF seq2seq model id/path (Facebook NLLB-200 checkpoints).",
    )
    g4.add_argument(
        "--bt-nllb-device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Device for NLLB (default: cuda if available, else cpu).",
    )
    g4.add_argument(
        "--bt-nllb-src",
        type=str,
        default="eng_Latn",
        help="FLORES source lang for NLLB (default English).",
    )
    g4.add_argument(
        "--bt-nllb-pivot",
        type=str,
        default="zho_Hans",
        help="FLORES pivot language (default zh-Hans ↔ en round-trip).",
    )
    g4.add_argument(
        "--bt-nllb-max-chunk-chars",
        type=int,
        default=420,
        help="Chunk long texts for NLLB (sentence-aware; hard-split if one sentence exceeds this).",
    )
    g4.add_argument(
        "--bt-nllb-gen-max-length",
        type=int,
        default=512,
        help="generation max_length for each NLLB translate call.",
    )
    g4.add_argument(
        "--bt-nllb-num-beams",
        type=int,
        default=4,
        help="Beam search width for local NLLB.",
    )

    return p.parse_args()


def free_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_attack_and_aux_models(
    args: argparse.Namespace,
    lm_device: str,
) -> Dict[str, Any]:
    """Load attack-specific models once; sets ``attack_editors`` for the WM pipeline."""
    extra_hold: Dict[str, Any] = {}

    if args.attack == "context_synonym":
        bert_dev = args.bert_device or lm_device
        print(
            f"Loading BERT MLM for context_synonym: {args.bert_masked_lm!r} ({bert_dev})",
            flush=True,
        )
        tokenizer = BertTokenizer.from_pretrained(args.bert_masked_lm)
        model = BertForMaskedLM.from_pretrained(args.bert_masked_lm)
        model = model.to(bert_dev).eval()
        editor = ContextAwareSynonymSubstitution(
            ratio=float(args.synonym_ratio),
            tokenizer=tokenizer,
            model=model,
            device=bert_dev,
        )
        extra_hold["bert_tokenizer"] = tokenizer
        extra_hold["bert_model"] = model
        extra_hold["attack_editors"] = [TruncatePromptTextEditor(), editor]
        return extra_hold

    if args.attack == "word_deletion":
        r = float(args.deletion_ratio)
        if not 0.0 <= r <= 1.0:
            raise ValueError(f"deletion-ratio must be in [0, 1], got {r}")
        print(f"word_deletion: deletion_ratio={r}", flush=True)
        extra_hold["attack_editors"] = [
            TruncatePromptTextEditor(),
            WordDeletion(ratio=r),
        ]
        return extra_hold

    if args.attack == "paraphrase":
        para_dev = args.paraphrase_device
        if para_dev is None:
            para_dev = "cuda" if torch.cuda.is_available() else "cpu"

        gen_kw: Dict[str, Any] = {
            "max_length": int(args.paraphrase_max_length),
            "num_beams": max(1, int(args.paraphrase_num_beams)),
            "num_return_sequences": 1,
            "do_sample": bool(args.paraphrase_do_sample),
        }
        if args.paraphrase_do_sample:
            gen_kw["top_p"] = float(args.paraphrase_top_p)
            if args.paraphrase_top_k is not None:
                gen_kw["top_k"] = int(args.paraphrase_top_k)

        print(
            f"Loading paraphrase (Parrot T5): model={args.paraphrase_model!r} "
            f"device_map={args.paraphrase_device_map} placement_device={para_dev!r}",
            flush=True,
        )
        tok = AutoTokenizer.from_pretrained(args.paraphrase_model)

        if para_dev == "cpu":
            para_model = AutoModelForSeq2SeqLM.from_pretrained(
                args.paraphrase_model,
                torch_dtype=torch.float32,
                device_map={"": "cpu"},
            )
            edit_device = "cpu"
        elif args.paraphrase_device_map == "auto":
            dt = torch.float16 if torch.cuda.is_available() else torch.float32
            para_model = AutoModelForSeq2SeqLM.from_pretrained(
                args.paraphrase_model,
                device_map="auto",
                torch_dtype=dt,
            )
            edit_device = str(next(para_model.parameters()).device)
        else:
            dt = torch.float16 if para_dev == "cuda" and torch.cuda.is_available() else torch.float32
            para_model = AutoModelForSeq2SeqLM.from_pretrained(
                args.paraphrase_model,
                torch_dtype=dt,
            ).to(para_dev)
            edit_device = para_dev

        para_model.eval()
        editor = ParrotT5Paraphraser(
            tokenizer=tok,
            model=para_model,
            device=edit_device,
            sent_interval=int(args.paraphrase_sent_interval),
            **gen_kw,
        )
        extra_hold["paraphrase_tokenizer"] = tok
        extra_hold["paraphrase_model"] = para_model
        extra_hold["attack_editors"] = [TruncatePromptTextEditor(), editor]
        return extra_hold

    if args.attack == "back_translation":
        if args.bt_backend == "online":
            src = args.bt_source_lang.strip().lower()
            piv = args.bt_intermediate_lang.strip().lower()
            if src == piv:
                raise ValueError(
                    "back_translation: --bt-source-lang and --bt-intermediate-lang must differ.",
                )
            print(
                f"back_translation (online): {src}->{piv}->{src} (`translate`/MyMemory; max ~500 chars/query)",
                flush=True,
            )
            editor = BackTranslationTextEditor(
                translate_to_intermediary=Translator(from_lang=src, to_lang=piv).translate,
                translate_to_source=Translator(from_lang=piv, to_lang=src).translate,
            )
            extra_hold["attack_editors"] = [TruncatePromptTextEditor(), editor]
            return extra_hold

        n_src = args.bt_nllb_src.strip()
        n_piv = args.bt_nllb_pivot.strip()
        if n_src == n_piv:
            raise ValueError(
                "back_translation: --bt-nllb-src and --bt-nllb-pivot must differ.",
            )
        nllb_dev = args.bt_nllb_device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        dt = torch.float16 if nllb_dev == "cuda" and torch.cuda.is_available() else torch.float32
        print(
            f"back_translation (local NLLB): model={args.bt_nllb_model!r} device={nllb_dev} "
            f"{n_src!r}->{n_piv!r}->{n_src!r}",
            flush=True,
        )
        # AutoTokenizer wraps NLLB as generic TokenizersBackend and skips lang-prefix plumbing;
        # NllbTokenizer sets src-lang special tokens so generate() receives correct encoder input.
        bt_tok = NllbTokenizer.from_pretrained(args.bt_nllb_model, src_lang=n_src)
        bt_model = AutoModelForSeq2SeqLM.from_pretrained(
            args.bt_nllb_model,
            torch_dtype=dt,
        ).to(nllb_dev).eval()
        editor = LocalNLLBBackTranslationEditor(
            bt_tok,
            bt_model,
            device=nllb_dev,
            source_lang=n_src,
            pivot_lang=n_piv,
            max_chunk_chars=int(args.bt_nllb_max_chunk_chars),
            generation_max_length=int(args.bt_nllb_gen_max_length),
            num_beams=int(args.bt_nllb_num_beams),
        )
        extra_hold["bt_nllb_tokenizer"] = bt_tok
        extra_hold["bt_nllb_model"] = bt_model
        extra_hold["attack_editors"] = [TruncatePromptTextEditor(), editor]
        return extra_hold

    raise ValueError(f"Unhandled attack: {args.attack}")


def dispose_attack_resources(extra_hold: Dict[str, Any]) -> None:
    for k in (
        "bert_model",
        "bert_tokenizer",
        "paraphrase_model",
        "paraphrase_tokenizer",
        "bt_nllb_model",
        "bt_nllb_tokenizer",
        "attack_editors",
    ):
        if k in extra_hold:
            del extra_hold[k]
    extra_hold.clear()
    free_memory()


def evaluate_algorithm(
    spec: Dict[str, object],
    dataset: C4Dataset,
    transformers_config: TransformersConfig,
    unwm_source: str,
    wm_text_editor_list: List[Any],
) -> Dict[str, object]:
    name = str(spec["name"])
    reverse = bool(spec["reverse"])
    load_name, config_path = _wm_spec_load_params(spec)
    print(f"\n=== {name} (reverse={reverse}) ===", flush=True)
    started = time.time()

    watermark = AutoWatermark.load(
        load_name,
        algorithm_config=config_path,
        transformers_config=transformers_config,
    )

    wm_pipe = WatermarkedTextDetectionPipeline(
        dataset=dataset,
        text_editor_list=wm_text_editor_list,
        show_progress=True,
        return_type=DetectionPipelineReturnType.SCORES,
    )
    unwm_pipe = UnWatermarkedTextDetectionPipeline(
        dataset=dataset,
        text_editor_list=[],
        text_source_mode=unwm_source,
        show_progress=True,
        return_type=DetectionPipelineReturnType.SCORES,
    )

    wm_scores = [float(s) for s in wm_pipe.evaluate(watermark)]
    unwm_scores = [float(s) for s in unwm_pipe.evaluate(watermark)]

    calc = DynamicThresholdSuccessRateCalculator(
        labels=METRIC_LABELS, rule="best", reverse=reverse,
    )
    detailed = calc.calculate_detailed(wm_scores, unwm_scores)
    metrics = detailed["metrics"]
    threshold = detailed["threshold"]
    subset_metrics = detailed["subset_metrics"]
    elapsed = time.time() - started
    print(f"{name}: {metrics}  threshold={threshold:.6g}  (elapsed {elapsed:.1f}s)", flush=True)

    del watermark, wm_pipe, unwm_pipe
    free_memory()

    return {
        "metrics": metrics,
        "threshold": threshold,
        "subset_metrics": subset_metrics,
        "elapsed_seconds": round(elapsed, 2),
        "wm_scores": wm_scores,
        "unwm_scores": unwm_scores,
        "reverse": reverse,
    }


def attack_config_dict(args: argparse.Namespace, lm_device: str) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "attack": args.attack,
        "seed": args.seed,
    }
    if args.attack == "context_synonym":
        base["synonym_ratio"] = args.synonym_ratio
        base["bert_masked_lm"] = args.bert_masked_lm
        base["bert_device"] = args.bert_device or lm_device
    if args.attack == "word_deletion":
        base["deletion_ratio"] = args.deletion_ratio
    if args.attack == "paraphrase":
        para_dev = args.paraphrase_device or ("cuda" if torch.cuda.is_available() else "cpu")
        base["paraphrase_model"] = args.paraphrase_model
        base["paraphrase_device"] = para_dev
        base["paraphrase_device_map"] = args.paraphrase_device_map
        base["paraphrase_sent_interval"] = args.paraphrase_sent_interval
        base["paraphrase_num_beams"] = args.paraphrase_num_beams
        base["paraphrase_max_length"] = args.paraphrase_max_length
        base["paraphrase_do_sample"] = args.paraphrase_do_sample
        base["paraphrase_top_p"] = args.paraphrase_top_p
        base["paraphrase_top_k"] = args.paraphrase_top_k
    if args.attack == "back_translation":
        base["bt_backend"] = args.bt_backend
        base["bt_source_lang"] = args.bt_source_lang
        base["bt_intermediate_lang"] = args.bt_intermediate_lang
        if args.bt_backend == "nllb":
            base["bt_nllb_model"] = args.bt_nllb_model
            base["bt_nllb_device"] = args.bt_nllb_device or (
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            base["bt_nllb_src"] = args.bt_nllb_src
            base["bt_nllb_pivot"] = args.bt_nllb_pivot
            base["bt_nllb_max_chunk_chars"] = args.bt_nllb_max_chunk_chars
            base["bt_nllb_gen_max_length"] = args.bt_nllb_gen_max_length
            base["bt_nllb_num_beams"] = args.bt_nllb_num_beams
    return base


def write_report(
    path: str,
    args: argparse.Namespace,
    lm_device: str,
    results: Dict[str, Dict[str, object]],
    failures: Dict[str, str],
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    atk_cfg = attack_config_dict(args, lm_device)

    if out.suffix.lower() == ".json":
        payload = {
            "attack_config": atk_cfg,
            "config": {
                "runs": args.runs,
                "model_path": args.model_path,
                "dataset_path": args.dataset_path,
                "unwm_source": args.unwm_source,
                "max_new_tokens": args.max_new_tokens,
                "min_length": args.min_length,
                "dtype": args.dtype,
            },
            "results": {
                name: {
                    "metrics": r["metrics"],
                    "threshold": r["threshold"],
                    "subset_metrics": r["subset_metrics"],
                    "elapsed_seconds": r["elapsed_seconds"],
                    "reverse": r["reverse"],
                    **(
                        {"wm_scores": r["wm_scores"], "unwm_scores": r["unwm_scores"]}
                        if args.save_scores else {}
                    ),
                }
                for name, r in results.items()
            },
            "failures": failures,
        }
        out.write_text(json.dumps(payload, indent=2))
        return

    with out.open("w") as f:
        f.write("# Watermark Detection Under Attack\n\n")
        f.write(f"- Attack: `{args.attack}`\n")
        f.write(f"- Attack details: `{json.dumps(atk_cfg, ensure_ascii=False)}`\n")
        f.write(f"- Runs per algorithm: `{args.runs}`\n")
        f.write(f"- Model: `{args.model_path}` (dtype=`{args.dtype}`, device=`{lm_device}`)\n")
        f.write(f"- Dataset: `{args.dataset_path}`\n")
        f.write(f"- Unwatermarked source: `{args.unwm_source}`\n")
        f.write(
            "Watermarked path: truncate prompt → attack → detect. "
            "Unwatermarked path: unchanged from baseline detection script.\n\n"
        )

        for name, r in results.items():
            f.write(f"### {name}\n\n")
            sm = r["subset_metrics"]
            f.write("| Text | TPR | TNR | FPR | FNR | F1 | ACC |\n")
            f.write("|:-----|:---:|:---:|:---:|:---:|:---:|:---:|\n")
            for row_label, sm_key in (
                ("Unwatermarked", "unwatermarked"),
                ("Watermarked", "watermarked"),
            ):
                row_m = sm[sm_key]
                cells = " | ".join(_format_metric_cell(row_m[k]) for k in METRIC_LABELS)
                f.write(f"| {row_label} | {cells} |\n")
            f.write(
                f"\n*Best-F1 threshold:* `{float(r['threshold']):.6g}` — "
                f"*elapsed:* {r['elapsed_seconds']:.1f}s\n\n"
            )

        if failures:
            f.write("\n## Failures\n\n")
            for name, msg in failures.items():
                f.write(f"- **{name}**: `{msg}`\n")

        if args.save_scores:
            f.write("\n## Raw scores\n\n")
            for name, r in results.items():
                f.write(f"### {name}\n\n")
                f.write(f"- watermarked (attacked): `{r['wm_scores']}`\n")
                f.write(f"- unwatermarked: `{r['unwm_scores']}`\n\n")


def main() -> None:
    args = parse_args()
    if args.attack == "paraphrase":
        # Avoid transformers background Thread-auto_conversion → HF discussions (403 when disabled).
        os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")
    _set_global_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]

    extra_hold = prepare_attack_and_aux_models(args, lm_device=device)
    wm_editors = extra_hold["attack_editors"]

    print(f"Loading shared LM: {args.model_path} (dtype={args.dtype}, device={device})", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=dtype if device == "cuda" else torch.float32,
    ).to(device).eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    min_length = args.min_length if args.min_length is not None else max(args.max_new_tokens - 30, 30)
    transformers_config = TransformersConfig(
        model=model,
        tokenizer=tokenizer,
        vocab_size=model.config.vocab_size,
        device=device,
        max_new_tokens=args.max_new_tokens,
        min_length=min_length,
        do_sample=True,
        no_repeat_ngram_size=4,
    )

    dataset = C4Dataset(args.dataset_path, max_samples=args.runs)
    print(f"Loaded C4 with {dataset.prompt_nums} prompts / {dataset.natural_text_nums} natural texts.", flush=True)

    spec_by_name = {s["name"]: s for s in ALGORITHM_SPECS}
    selected = args.algorithms or [s["name"] for s in ALGORITHM_SPECS]

    results: Dict[str, Dict[str, object]] = {}
    failures: Dict[str, str] = {}

    for name in selected:
        if name not in spec_by_name:
            msg = f"unknown algorithm '{name}', valid: {list(spec_by_name)}"
            print(f"!! {msg}", flush=True)
            failures[name] = msg
            continue

        spec = spec_by_name[name]
        try:
            results[name] = evaluate_algorithm(
                spec=spec,
                dataset=dataset,
                transformers_config=transformers_config,
                unwm_source=args.unwm_source,
                wm_text_editor_list=wm_editors,
            )
        except Exception as exc:
            tb = traceback.format_exc()
            print(f"!! {name} failed: {exc}\n{tb}", flush=True)
            failures[name] = f"{type(exc).__name__}: {exc}"
            free_memory()

    write_report(args.report, args, device, results, failures)

    dispose_attack_resources(extra_hold)
    del model, tokenizer
    free_memory()
    print(f"\nReport saved to: {args.report}", flush=True)


if __name__ == "__main__":
    main()
