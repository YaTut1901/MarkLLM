"""Compare: no watermark vs segment-watermark vs ETHWatermark (Fairoze)."""

import argparse
import hashlib
import sys, json, time, secrets
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList, LogitsProcessor
from eth_account import Account
from eth_account.messages import encode_defunct

from utils.transformers_config import TransformersConfig
from watermark.auto_watermark import AutoWatermark

sys.path.insert(0, str(Path(__file__).parent / "segment-watermark"))
from wm import RSDecoder, get_pvalue_segment_based
from wm.generator import RSGenerator

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "models/Qwen2.5-Coder-1.5B"
PROMPT = json.loads(open("dataset/c4/processed_c4.json").readline())["prompt"]
WM_PARAMS = dict(seed=42, salt_key=35317, ngram=3, gamma=0.25, delta=8.0, seeding="hash")

SCHEME_CHOICES = ("baseline", "segment-520", "segment-128", "fairoze")
DEFAULT_GEN_KWARGS = dict(do_sample=True, temperature=0.8, top_p=0.95, no_repeat_ngram_size=4)

# Minimum tokens for reliable payload recovery.
# segment-520: ~6.5 tok/seg for m=8 (EIP-2098 + ngram=3, gamma=0.25, delta=8)
SCHEME_TOKENS = {
    "baseline": 600,
    "segment-520": 650,   # 100 segments × 6.5 tokens/segment
    "segment-128": 1200,   # 22 segments × ~55 tokens/segment
    "fairoze": 600,
}

# RS params per scheme: (k, n, m) where k=message segments, n=codeword segments,
# m=bits/segment. Error correction capacity = (n-k)/2.
RS_PARAMS = {
    # EIP-2098 compact (512 bits, k=64) + Варіант 2: trade RS parity for more tok/seg.
    # (64, 100, 8): corrects 18 byte errors, default ~650 tok (~6.5/seg) with ngram=3.
    # Rationale: empirically (65, 120, 8) at 50 tok/seg saw ~15 byte errors; increasing
    # tok/seg to 55 should drive error rate below 18, keeping recovery reliable.
    "segment-520": (64, 100, 8),
    "segment-128": (16, 22, 8),    # can correct 3 errors
}


class SegmentWMLogitsProcessor(LogitsProcessor):
    """Thin adapter: wraps RSGenerator.logits_processor for model.generate()."""
    def __init__(self, rs_gen):
        self.g = rs_gen

    def __call__(self, input_ids, scores):
        ngram_tokens = input_ids[:, -self.g.ngram:]
        return self.g.logits_processor(scores, ngram_tokens)


def run_segment_wm_detection(text, tokenizer, segments_num, gf_segments_num, segment_bit, model_vocab_size):
    det = RSDecoder(tokenizer, **WM_PARAMS, segments_num=segments_num,
                    gf_segments_num=gf_segments_num, segment_bit=segment_bit)
    det.vocab_size = model_vocab_size
    scores, ntoks, _ = det.get_aggregate_scores([text])
    payloads = det.get_decoded_payload(scores)
    zscores, pvalues = get_pvalue_segment_based(scores, ntoks)
    return payloads[0], zscores[0], ntoks[0]


def parse_args():
    p = argparse.ArgumentParser(
        description="Run selected watermark generations and write watermark_comparison_report.md.",
    )
    p.add_argument(
        "--only",
        nargs="+",
        choices=SCHEME_CHOICES,
        metavar="SCHEME",
        help=f"Run only these schemes (order preserved). Choices: {', '.join(SCHEME_CHOICES)}. Default: all.",
    )
    p.add_argument(
        "--max-tokens", type=int, default=None,
        help="Override max_new_tokens for all schemes. Default: per-scheme values "
             "(baseline/fairoze=600, segment-128=1200, segment-520=650).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    run_order = list(args.only) if args.only else list(SCHEME_CHOICES)

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True,
                                                  torch_dtype=torch.float16).to(DEVICE).eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    encoded_prompt = tokenizer(PROMPT, return_tensors="pt", add_special_tokens=True).to(DEVICE)
    results = {}

    def generate(label, max_new_tokens, logits_processor=None):
        print(f"\n--- {label} (max_new_tokens={max_new_tokens}) ---")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        kw = {**DEFAULT_GEN_KWARGS, "max_new_tokens": max_new_tokens}
        if logits_processor:
            kw["logits_processor"] = LogitsProcessorList([logits_processor])
        t0 = time.time()
        out = model.generate(**encoded_prompt, **kw)
        elapsed = time.time() - t0
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        prompt_text = tokenizer.decode(encoded_prompt.input_ids[0], skip_special_tokens=True)
        generated = text[len(prompt_text):]
        n_tok = len(tokenizer.encode(generated, add_special_tokens=False))
        print(f"  {n_tok} tokens in {elapsed:.1f}s")
        results[label] = {"text": generated, "tokens": n_tok, "time": round(elapsed, 1)}
        return generated

    def tokens_for(scheme):
        return args.max_tokens if args.max_tokens else SCHEME_TOKENS[scheme]

    for scheme in run_order:
        if scheme == "baseline":
            generate("Baseline (no watermark)", tokens_for(scheme))

        elif scheme == "segment-520":
            eth_acc = Account.create()
            prefix_hash = hashlib.sha256(PROMPT.encode("utf-8")).digest()
            msg = encode_defunct(primitive=prefix_hash)
            sig = eth_acc.sign_message(msg)
            sig_bytes = bytes(sig.signature)  # standard 65-byte: r||s||v
            # EIP-2098 compact: 64 bytes = r(32) || yParityAndS(32),
            # where yParityAndS = (y_parity << 255) | s. eth_account emits low-s,
            # so s's top bit is always 0 and safely repurposed for y_parity.
            r_bytes = sig_bytes[:32]
            s_int = int.from_bytes(sig_bytes[32:64], "big")
            y_parity = sig_bytes[64] - 27  # v ∈ {27, 28} → y_parity ∈ {0, 1}
            assert s_int >> 255 == 0, "s must be low-half (top bit 0) for EIP-2098"
            compact_s = (y_parity << 255) | s_int
            compact_bytes = r_bytes + compact_s.to_bytes(32, "big")  # 64 bytes = 512 bits
            sig_int = int.from_bytes(compact_bytes, "big")
            print(f"  Signer: {eth_acc.address} (EIP-2098 compact, 512 bits)")
            k, n, m = RS_PARAMS["segment-520"]
            label_520 = f"Segment-WM 512-bit ECDSA EIP-2098 (k={k} n={n} m={m})"
            gen_520 = RSGenerator(model, tokenizer, payload=sig_int,
                                  segments_num=k, gf_segments_num=n, segment_bit=m, **WM_PARAMS)
            text_520 = generate(label_520, tokens_for(scheme), SegmentWMLogitsProcessor(gen_520))
            ext_int, z_520, nt_520 = run_segment_wm_detection(
                text_520, tokenizer, k, n, m, model.config.vocab_size)
            ext_bytes = ext_int.to_bytes(65, "big")[-64:]  # safe clamp to 64 bytes
            matching_bytes = sum(1 for a, b in zip(compact_bytes, ext_bytes) if a == b)
            payload_match = ext_int == sig_int
            # Decode EIP-2098: split r, recover y_parity and s from top bit
            ext_r = ext_bytes[:32]
            ext_compact_s = int.from_bytes(ext_bytes[32:64], "big")
            ext_y_parity = ext_compact_s >> 255
            ext_s_int = ext_compact_s & ((1 << 255) - 1)
            ext_s = ext_s_int.to_bytes(32, "big")
            # Recover address; try both parities in case noisy high bit
            recovered, recover_err = None, None
            for v_try in (ext_y_parity + 27, (1 - ext_y_parity) + 27):
                candidate = ext_r + ext_s + bytes([v_try])
                try:
                    recovered = Account.recover_message(msg, signature=candidate)
                    if recovered.lower() == eth_acc.address.lower():
                        break
                except Exception as e:
                    recover_err = str(e)
            addr_match = bool(recovered and recovered.lower() == eth_acc.address.lower())
            print(f"  Detection: z={z_520:.2f}, addr_match={addr_match}, tokens_scored={nt_520}")
            print(f"  Payload match: {payload_match}, matching bytes: {matching_bytes}/64")
            print(f"  Expected:  {eth_acc.address}")
            print(f"  Recovered: {recovered or f'FAILED ({recover_err})'}")
            results[label_520]["detection"] = (
                f"z={z_520:.2f}, addr_match={addr_match}, "
                f"bytes={matching_bytes}/64, scored={nt_520}"
            )
            results[label_520]["expected_addr"] = eth_acc.address
            results[label_520]["recovered_addr"] = recovered or f"FAILED ({matching_bytes}/64 bytes matched)"

        elif scheme == "segment-128":
            eth_acc = Account.create()
            prefix_hash = hashlib.sha256(PROMPT.encode("utf-8")).digest()
            sig = eth_acc.sign_message(encode_defunct(primitive=prefix_hash))
            sig_int = int.from_bytes(sig.signature, "big")
            commitment = sig_int & ((1 << 128) - 1)  # lower 128 bits of signature
            print(f"  Signer: {eth_acc.address}")
            gen_128 = RSGenerator(model, tokenizer, payload=commitment,
                                  segments_num=16, gf_segments_num=22, segment_bit=8, **WM_PARAMS)
            text_128 = generate("Segment-WM 128-bit commitment (k=16 n=22 m=8)",
                                tokens_for(scheme), SegmentWMLogitsProcessor(gen_128))
            ext_128, z_128, nt_128 = run_segment_wm_detection(
                text_128, tokenizer, 16, 22, 8, model.config.vocab_size)
            commit_match = ext_128 == commitment
            print(f"  Detection: z={z_128:.2f}, commit_match={commit_match}, tokens_scored={nt_128}")
            print(f"  Expected:  0x{commitment:032x}")
            print(f"  Recovered: 0x{ext_128:032x}")
            label_128 = "Segment-WM 128-bit commitment (k=16 n=22 m=8)"
            results[label_128]["detection"] = (
                f"z={z_128:.2f}, commit_match={commit_match}, scored={nt_128}"
            )
            results[label_128]["expected_addr"] = eth_acc.address
            results[label_128]["recovered_addr"] = f"commitment 0x{ext_128:032x}"
            results[label_128]["commitment_match"] = commit_match

        elif scheme == "fairoze":
            print("\n--- ETHWatermark (Fairoze hash-chain) ---")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            eth_acc = Account.create()
            print(f"  Signer: {eth_acc.address}")
            tc = TransformersConfig(
                model=model, tokenizer=tokenizer, vocab_size=model.config.vocab_size,
                device=DEVICE, max_new_tokens=tokens_for(scheme),
                do_sample=True, no_repeat_ngram_size=4)
            eth_wm = AutoWatermark.load(
                "ETHWatermark", algorithm_config="config/ETHWatermark.json", transformers_config=tc)
            t0 = time.time()
            eth_text = eth_wm.generate_watermarked_text(PROMPT, private_key=eth_acc.key)
            elapsed = time.time() - t0
            eth_generated = eth_text[len(PROMPT):] if eth_text.startswith(PROMPT) else eth_text
            n_tok = len(tokenizer.encode(eth_generated, add_special_tokens=False))
            print(f"  {n_tok} tokens in {elapsed:.1f}s")
            det = eth_wm.detect_watermark(eth_text)
            addr_match = (
                (det.get("recovered_address", "").lower() == eth_acc.address.lower())
                if det.get("recovered_address") else False
            )
            print(f"  Detection: watermarked={det['is_watermarked']}, address_match={addr_match}")
            results["ETHWatermark (Fairoze)"] = {
                "text": eth_generated, "tokens": n_tok, "time": round(elapsed, 1),
                "detection": f"watermarked={det['is_watermarked']}, addr_match={addr_match}",
                "expected_addr": eth_acc.address,
                "recovered_addr": det.get("recovered_address", "N/A"),
            }

    # ── Write report ─────────────────────────────────────────────────────────
    report = Path("watermark_comparison_report.md")
    with open(report, "w") as f:
        f.write("# Watermark Comparison: Segment-WM vs Fairoze\n\n")
        f.write(f"**Runs**: `{', '.join(run_order)}`\n\n")
        f.write(f"**Model**: `{MODEL_PATH}` | **Prompt**: `{PROMPT[:80]}...`\n\n")
        f.write("| Scheme | Tokens | Time (s) | Detection | Expected address | Recovered address |\n")
        f.write("|--------|--------|----------|-----------|------------------|-------------------|\n")
        for label, r in results.items():
            exp = r.get("expected_addr", "—")
            rec = r.get("recovered_addr", "—")
            f.write(f"| {label} | {r['tokens']} | {r['time']} "
                    f"| {r.get('detection', 'N/A')} | `{exp}` | `{rec}` |\n")
        f.write("\n---\n\n")
        for label, r in results.items():
            f.write(f"## {label}\n\n")
            if r.get("detection"):
                f.write(f"**Detection**: {r['detection']}\n\n")
            if r.get("expected_addr"):
                f.write(f"**Expected address**: `{r['expected_addr']}`\n\n")
            if r.get("recovered_addr"):
                f.write(f"**Recovered address**: `{r['recovered_addr']}`\n\n")
            if "commitment_match" in r:
                f.write(f"**Commitment match**: {r['commitment_match']}\n\n")
            f.write(f"```\n{r['text']}\n```\n\n")
    print(f"\nReport saved to {report}")


if __name__ == "__main__":
    main()
