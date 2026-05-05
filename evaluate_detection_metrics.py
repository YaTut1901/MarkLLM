"""Run watermark detection metrics (TPR/TNR/FPR/FNR/F1/ACC) over multiple algorithms.

The base language model + tokenizer are loaded **once** and shared across every
algorithm via a single ``TransformersConfig``. Each watermark instance (and any
auxiliary detector/embedder it pulls in) is dropped between algorithms with
``gc.collect()`` + ``torch.cuda.empty_cache()`` so we never duplicate the LM in
GPU memory and don't OOM after the first algorithm finishes.

Per algorithm, ``--runs`` prompts from the C4 dataset are used to:
  - generate watermarked text and detect → watermarked-side scores;
  - read the corresponding natural (or generated unwatermarked) C4 text and
    detect → unwatermarked-side scores.
A dynamic-threshold ``DynamicThresholdSuccessRateCalculator`` then picks the
threshold that maximises F1. The report lists **one subsection per algorithm**:
first table row = rates on gold-unwatermarked text (TNR, FPR; other cells N/A
shown as ``—``), second row = rates on gold-watermarked text (TPR, FNR). ``F1``
is the overall value at that threshold; per-row ``ACC`` is accuracy restricted
to that gold class (same as TNR / TPR on that row).
Algorithms whose ``score`` is a p-value (lower → watermark) are flagged with
``reverse=True`` so the threshold search treats them correctly.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluation.dataset import C4Dataset
from evaluation.pipelines.detection import (
    DetectionPipelineReturnType,
    UnWatermarkedTextDetectionPipeline,
    WatermarkedTextDetectionPipeline,
)
from evaluation.tools.success_rate_calculator import DynamicThresholdSuccessRateCalculator
from evaluation.tools.text_editor import TruncatePromptTextEditor
from utils.transformers_config import TransformersConfig
from watermark.auto_watermark import AutoWatermark


METRIC_LABELS: List[str] = ["TPR", "TNR", "FPR", "FNR", "F1", "ACC"]

# ``reverse=True`` ⇒ lower ``score`` indicates watermark presence (p-value style).
# Optional ``load_name`` / ``config_path`` when the registry name or JSON path
# differs from ``name`` (e.g. FairOze → ETHWatermark).
ALGORITHM_SPECS: List[Dict[str, object]] = [
    {"name": "KGW",       "reverse": False},  # z-score
    {"name": "EXP",       "reverse": True},   # p-value
    {"name": "Unbiased",  "reverse": True},   # p-value
    {"name": "SEMSTAMP",  "reverse": False},  # z-score
    {"name": "SynthID",   "reverse": False},  # mean detector score
    {"name": "SegmentWM", "reverse": False},  # z-score
    {
        "name": "FairOzeWM",
        "load_name": "ETHWatermark",
        "config_path": "config/ETHWatermark.json",
        "reverse": False,
    },  # binary detect score (Fairoze-style hash-chain)
]


def _wm_spec_load_params(spec: Dict[str, object]) -> tuple[str, str]:
    load_name = str(spec.get("load_name", spec["name"]))
    config_path = str(spec.get("config_path", f"config/{load_name}.json"))
    return load_name, config_path

DEFAULT_MODEL_PATH = "models/Qwen2.5-Coder-1.5B"
DEFAULT_DATASET_PATH = "dataset/c4/processed_c4.json"
DEFAULT_REPORT_PATH = "detection_metrics_report.md"


def _format_metric_cell(value: object) -> str:
    if value is None:
        return "—"
    return f"{float(value):.3f}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate watermark detection metrics over multiple algorithms "
                    "with a single shared LM (no per-algorithm model reload).",
    )
    p.add_argument(
        "--runs", type=int, default=50,
        help="Number of prompts/runs per algorithm (default: 50).",
    )
    p.add_argument(
        "--algorithms", nargs="+", default=None, metavar="NAME",
        help=f"Subset to run, in order. Default: all of "
             f"{[s['name'] for s in ALGORITHM_SPECS]}.",
    )
    p.add_argument(
        "--model-path", type=str, default=DEFAULT_MODEL_PATH,
        help=f"HF model path or repo id (default: {DEFAULT_MODEL_PATH}).",
    )
    p.add_argument(
        "--dataset-path", type=str, default=DEFAULT_DATASET_PATH,
        help=f"C4 jsonl dataset (default: {DEFAULT_DATASET_PATH}).",
    )
    p.add_argument(
        "--report", type=str, default=DEFAULT_REPORT_PATH,
        help=f"Output report (.md or .json) path (default: {DEFAULT_REPORT_PATH}).",
    )
    p.add_argument(
        "--unwm-source", choices=["natural", "generated"], default="natural",
        help="Source of unwatermarked text: 'natural' uses C4 reference text "
             "(fast); 'generated' calls model.generate without a watermark.",
    )
    p.add_argument(
        "--max-new-tokens", type=int, default=200,
        help="max_new_tokens for generation kwargs (default: 200).",
    )
    p.add_argument(
        "--min-length", type=int, default=None,
        help="min_length for generation kwargs (default: max_new_tokens - 30).",
    )
    p.add_argument(
        "--dtype", choices=["float16", "bfloat16", "float32"], default="float16",
        help="Torch dtype to load the model in (default: float16).",
    )
    p.add_argument(
        "--save-scores", action="store_true",
        help="Also persist raw watermarked / unwatermarked scores in the report.",
    )
    return p.parse_args()


def free_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def evaluate_algorithm(
    spec: Dict[str, object],
    dataset: C4Dataset,
    transformers_config: TransformersConfig,
    unwm_source: str,
) -> Dict[str, object]:
    """Instantiate one watermark, run both pipelines, compute metrics, free aux models."""
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
        text_editor_list=[TruncatePromptTextEditor()],
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


def write_report(
    path: str,
    args: argparse.Namespace,
    results: Dict[str, Dict[str, object]],
    failures: Dict[str, str],
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.suffix.lower() == ".json":
        payload = {
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
                    **({"wm_scores": r["wm_scores"], "unwm_scores": r["unwm_scores"]}
                       if args.save_scores else {}),
                }
                for name, r in results.items()
            },
            "failures": failures,
        }
        out.write_text(json.dumps(payload, indent=2))
        return

    with out.open("w") as f:
        f.write("# Watermark Detection Metrics Report\n\n")
        f.write(f"- Runs per algorithm: `{args.runs}`\n")
        f.write(f"- Model: `{args.model_path}` (dtype=`{args.dtype}`)\n")
        f.write(f"- Dataset: `{args.dataset_path}`\n")
        f.write(f"- Unwatermarked source: `{args.unwm_source}`\n")
        f.write(f"- max_new_tokens: `{args.max_new_tokens}`, "
                f"min_length: `{args.min_length}`\n\n")
        f.write(
            "Per algorithm: threshold maximises overall F1; **Unwatermarked** row = "
            "TNR/FPR (and class ACC) on clean gold; **Watermarked** row = TPR/FNR "
            "on watermarked gold. **F1** is the same overall value in both rows. "
            "Cells not defined for that row show —.\n\n"
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
                f.write(f"- watermarked: `{r['wm_scores']}`\n")
                f.write(f"- unwatermarked: `{r['unwm_scores']}`\n\n")


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]

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
    print(f"Loaded C4 with {dataset.prompt_nums} prompts / "
          f"{dataset.natural_text_nums} natural texts.", flush=True)

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
            )
        except Exception as exc:  # noqa: BLE001 — keep going across algorithms
            tb = traceback.format_exc()
            print(f"!! {name} failed: {exc}\n{tb}", flush=True)
            failures[name] = f"{type(exc).__name__}: {exc}"
            free_memory()

    write_report(args.report, args, results, failures)
    print(f"\nReport saved to: {args.report}", flush=True)


if __name__ == "__main__":
    main()
