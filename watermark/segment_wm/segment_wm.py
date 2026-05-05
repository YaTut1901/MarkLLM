# Copyright 2024 THU-BPM MarkLLM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""MarkLLM adapter for segment-watermark RS (RSGenerator + RSDecoder).

The upstream implementation lives under ``segment-watermark/wm`` (hyphenated
folder name); it is loaded via ``sys.path`` like ``compare_watermarks.py``.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Optional, Union

from functools import partial

from eth_account import Account
from eth_account.messages import encode_defunct

from transformers import LogitsProcessor, LogitsProcessorList

from ..base import BaseWatermark, BaseConfig
from utils.transformers_config import TransformersConfig
from visualize.data_for_visualization import DataForVisualization

_SEG_WM_PATH_READY = False


def _ensure_segment_wm_on_path() -> None:
    global _SEG_WM_PATH_READY
    if _SEG_WM_PATH_READY:
        return
    repo_root = Path(__file__).resolve().parents[2]
    seg_dir = repo_root / "segment-watermark"
    p = str(seg_dir)
    if p not in sys.path:
        sys.path.insert(0, p)
    _SEG_WM_PATH_READY = True


def _import_segment_wm_core():
    _ensure_segment_wm_on_path()
    from wm import RSDecoder, get_pvalue_segment_based
    from wm.generator import RSGenerator

    return RSGenerator, RSDecoder, get_pvalue_segment_based


class SegmentWMConfig(BaseConfig):
    """Config for segment-watermark RS scheme."""

    def initialize_parameters(self) -> None:
        self.segments_num = int(self.config_dict["segments_num"])
        self.gf_segments_num = int(self.config_dict["gf_segments_num"])
        self.segment_bit = int(self.config_dict["segment_bit"])
        self.gamma = float(self.config_dict["gamma"])
        self.delta = float(self.config_dict["delta"])
        self.ngram = int(self.config_dict["ngram"])
        self.seed = int(self.config_dict["seed"])
        self.salt_key = int(self.config_dict["salt_key"])
        self.seeding = str(self.config_dict["seeding"])
        self.payload = int(self.config_dict.get("payload", 0))
        self.z_threshold = float(self.config_dict.get("z_threshold", 4.0))
        # Same UTF-8 string used when signing ``encode_defunct(primitive=SHA256(prompt))`` for EIP-2098 recovery.
        raw = self.config_dict.get("signable_prompt")
        if isinstance(raw, str) and raw:
            self.signable_prompt: Optional[str] = raw
        else:
            self.signable_prompt = None

    @property
    def algorithm_name(self) -> str:
        return "SegmentWM"


def _recover_eth_address_from_eip2098_compact_int(ext_int: int, signable_message) -> Optional[str]:
    """Unpack EIP-2098 compact 512-bit payload to 65-byte sig and ``recover_message`` (see ``compare_watermarks``)."""
    try:
        ext_bytes = ext_int.to_bytes(65, "big", signed=False)[-64:]
        ext_r = ext_bytes[:32]
        ext_compact_s = int.from_bytes(ext_bytes[32:64], "big")
        ext_y_parity = ext_compact_s >> 255
        ext_s_int = ext_compact_s & ((1 << 255) - 1)
        ext_s = ext_s_int.to_bytes(32, "big")
        for v_try in (ext_y_parity + 27, (1 - ext_y_parity) + 27):
            candidate = ext_r + ext_s + bytes([v_try])
            try:
                addr = Account.recover_message(signable_message, signature=candidate)
                return str(addr)
            except Exception:
                continue
    except Exception:
        pass
    return None


class SegmentWMLogitsProcessor(LogitsProcessor):
    """Wraps RSGenerator.logits_processor for HuggingFace ``model.generate``."""

    def __init__(self, rs_gen) -> None:
        self.g = rs_gen

    def __call__(self, input_ids, scores):
        ngram_tokens = input_ids[:, -self.g.ngram :]
        return self.g.logits_processor(scores, ngram_tokens)


class SegmentWM(BaseWatermark):
    """RS segment watermark (Kirchenbauer-style logits + Reed–Solomon payload)."""

    def __init__(
        self,
        algorithm_config: str | SegmentWMConfig,
        transformers_config: TransformersConfig | None = None,
        *args,
        **kwargs,
    ) -> None:
        if isinstance(algorithm_config, str):
            self.config = SegmentWMConfig(algorithm_config, transformers_config, **kwargs)
        elif isinstance(algorithm_config, SegmentWMConfig):
            self.config = algorithm_config
        else:
            raise TypeError(
                "algorithm_config must be a path string or a SegmentWMConfig instance"
            )

    def generate_watermarked_text(self, prompt: str, *args, **kwargs) -> str:
        RSGenerator, _, _ = _import_segment_wm_core()
        payload = kwargs.get("payload")
        if payload is None:
            payload = self.config.payload
        payload = int(payload)

        model = self.config.generation_model
        tokenizer = self.config.generation_tokenizer

        rs_gen = RSGenerator(
            model,
            tokenizer,
            payload=payload,
            segments_num=self.config.segments_num,
            gf_segments_num=self.config.gf_segments_num,
            segment_bit=self.config.segment_bit,
            ngram=self.config.ngram,
            seed=self.config.seed,
            seeding=self.config.seeding,
            salt_key=self.config.salt_key,
            gamma=self.config.gamma,
            delta=self.config.delta,
        )
        # Upstream RS encode can return fewer than ``gf_segments_num`` symbols when
        # trailing parity coefficients are zero (notably payload=0). Pad so segment
        # indices in ``logits_processor`` never overflow.
        miss = self.config.gf_segments_num - len(rs_gen.gf_segments)
        if miss > 0:
            rs_gen.gf_segments = list(rs_gen.gf_segments) + [0] * miss
        lp = SegmentWMLogitsProcessor(rs_gen)
        generate_with_watermark = partial(
            model.generate,
            logits_processor=LogitsProcessorList([lp]),
            **self.config.gen_kwargs,
        )
        encoded_prompt = tokenizer(
            prompt, return_tensors="pt", add_special_tokens=True
        ).to(self.config.device)
        encoded = generate_with_watermark(**encoded_prompt)
        return tokenizer.batch_decode(encoded, skip_special_tokens=True)[0]

    def detect_watermark(
        self, text: str, return_dict: bool = True, *args, **kwargs
    ) -> Union[dict, tuple]:
        _, RSDecoder, get_pvalue_segment_based = _import_segment_wm_core()
        tokenizer = self.config.generation_tokenizer
        tok_ids = tokenizer.encode(text, add_special_tokens=False)
        min_len = self.config.ngram + 2
        if len(tok_ids) < min_len:
            empty = {
                "is_watermarked": False,
                "score": 0.0,
                "recovered_address": None,
                "tokens_scored": 0,
                "p_value": 1.0,
            }
            return empty if return_dict else (False, 0.0)

        det = RSDecoder(
            tokenizer,
            ngram=self.config.ngram,
            seed=self.config.seed,
            seeding=self.config.seeding,
            salt_key=self.config.salt_key,
            gamma=self.config.gamma,
            delta=self.config.delta,
            segments_num=self.config.segments_num,
            gf_segments_num=self.config.gf_segments_num,
            segment_bit=self.config.segment_bit,
        )
        det.vocab_size = self.config.vocab_size

        scores, ntoks, _ = det.get_aggregate_scores([text])
        payloads = det.get_decoded_payload(scores)
        zscores, pvalues = get_pvalue_segment_based(scores, ntoks)

        z = float(zscores[0])
        is_wm = bool(z > self.config.z_threshold)
        ext_int = int(payloads[0])
        recovered_address: Optional[str] = None
        if is_wm and self.config.signable_prompt is not None:
            prefix_hash = hashlib.sha256(self.config.signable_prompt.encode("utf-8")).digest()
            msg = encode_defunct(primitive=prefix_hash)
            recovered_address = _recover_eth_address_from_eip2098_compact_int(ext_int, msg)
        out = {
            "is_watermarked": is_wm,
            "score": z,
            "recovered_address": recovered_address,
            "tokens_scored": int(ntoks[0]),
            "p_value": float(pvalues[0]),
        }
        if return_dict:
            return out
        return (is_wm, z)

    def get_data_for_visualization(self, text: str, *args, **kwargs) -> DataForVisualization:
        """Minimal placeholder so notebook cells do not crash; no per-token WM signal."""
        encoded = self.config.generation_tokenizer(
            text, return_tensors="pt", add_special_tokens=False
        )["input_ids"][0].to(self.config.device)
        decoded_tokens = []
        for token_id in encoded:
            decoded_tokens.append(
                self.config.generation_tokenizer.decode(token_id.item())
            )
        highlight_values = [-1] * len(decoded_tokens)
        return DataForVisualization(decoded_tokens, highlight_values)
