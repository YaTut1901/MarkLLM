# ===========================================================================
# eth_watermark.py
# Description: Implementation of ETH publicly detectable watermark algorithm.
#              Adapted from Fairoze's publicly-detectable-watermark, using
#              Ethereum ECDSA (secp256k1) signatures instead of BLS.
#              Optional Reed–Solomon hardening (``max_planted_errors``) matches
#              Fairoze's ``RSCodec(max_planted_errors * 2)`` on the raw signature
#              bytes before OTP masking. Embeds the codeword via a hash chain
#              over fixed-length character segments.
# ===========================================================================

import hashlib
import json
import logging
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator, Union

import torch
from bitstring import BitArray
from eth_account import Account
from eth_account.messages import encode_defunct
from reedsolo import RSCodec
from tqdm import tqdm

from ..base import BaseWatermark, BaseConfig
from utils.transformers_config import TransformersConfig

logger = logging.getLogger(__name__)

# ``eth_account`` raw ECDSA signatures are 65 bytes (r ‖ s ‖ v) for standard secp256k1.
ETH_RAW_SIGNATURE_BYTE_LEN = 65


@contextmanager
def _optional_file_handler_for_logger(
    log: logging.Logger, log_file: str | None,
) -> Iterator[None]:
    """Attach a UTF-8 file handler for the duration of a single watermark run."""
    if not log_file:
        yield
        return
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"),
    )
    # Default/root loggers are often WARNING; ``info`` would be dropped before any
    # handler runs. Temporarily lower this logger so file logging actually records.
    old_level = log.level
    needed_level_bump = log.getEffectiveLevel() > logging.DEBUG
    if needed_level_bump:
        log.setLevel(logging.DEBUG)
    log.addHandler(fh)
    try:
        yield
    finally:
        log.removeHandler(fh)
        fh.close()
        if needed_level_bump:
            log.setLevel(old_level)


def eth_rs_parity_byte_count(max_planted_errors: int) -> int:
    """Parity bytes added by Reed–Solomon (Fairoze-style ``RSCodec(max_planted_errors * 2)``)."""
    if max_planted_errors <= 0:
        return 0
    return max_planted_errors * 2


def eth_encoded_signature_byte_len(max_planted_errors: int) -> int:
    """Byte length after RS encode (raw signature + parity)."""
    return ETH_RAW_SIGNATURE_BYTE_LEN + eth_rs_parity_byte_count(max_planted_errors)


def eth_signature_total_bits(max_planted_errors: int = 0) -> int:
    """Total bits embedded after RS (before OTP); 520 when ``max_planted_errors`` is 0."""
    return eth_encoded_signature_byte_len(max_planted_errors) * 8


def eth_signature_total_segments(bits_per_segment: int, max_planted_errors: int = 0) -> int:
    """Number of text segments for one codeword (``ceil(total_bits / bits_per_segment)``).

    When the codeword bit length is not a multiple of ``bits_per_segment``, the last
    segment embeds only the remainder bits (shorter hash truncation).
    """
    tb = eth_signature_total_bits(max_planted_errors)
    if bits_per_segment <= 0:
        raise ValueError('bits_per_segment must be positive')
    q, r = divmod(tb, bits_per_segment)
    return q if r == 0 else q + 1


def eth_bits_for_signature_segment(
    segment_index: int,
    bits_per_segment: int,
    max_planted_errors: int,
) -> int:
    """Hash output bit count for segment ``segment_index`` (0-based)."""
    tb = eth_signature_total_bits(max_planted_errors)
    b = bits_per_segment
    q, r = divmod(tb, b)
    if r == 0:
        return b
    if segment_index < q:
        return b
    return r


def eth_count_matching_bits(a: str, b: str) -> int:
    """Count positions where two equal-length bitstrings agree."""
    if len(a) != len(b):
        raise ValueError("bitstrings must have equal length")
    return sum(1 for i in range(len(a)) if a[i] == b[i])


def eth_bit_mismatch_cost(computed_bits: str, target_bits: str) -> int:
    """Fairoze-style bit discrepancy: *k* minus matching positions (Hamming distance)."""
    k = len(target_bits)
    if len(computed_bits) != k:
        raise ValueError("computed and target bitstrings must have equal length")
    return k - eth_count_matching_bits(computed_bits, target_bits)


def eth_min_tries_before_force_accept(planted_bit_errors: int, N: int) -> int | None:
    """Minimum segment samples before a forced mismatch accept may be used.

    ``N`` is ``max_planted_errors``. Returns ``None`` when no forced accept is allowed
    (``N == 0`` or budget already exhausted). Otherwise the first ``N // 2`` bit errors
    use ``1``; further errors use ``2, 4, 8, ...`` on successive sub-ranges of
    ``[N // 2, N)`` (dyadic split of the second half), matching the N=16 schedule.
    """
    if N <= 0:
        return None
    if planted_bit_errors >= N:
        return None
    first_half = N // 2
    if planted_bit_errors < first_half:
        return 1
    start = first_half
    remaining = N - first_half
    power = 1
    while start < N:
        chunk = (remaining + 1) // 2
        end = min(start + chunk, N)
        if planted_bit_errors < end:
            return 2**power
        start = end
        remaining = N - start
        power += 1
    return None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class ETHWatermarkConfig(BaseConfig):
    """Config class for ETHWatermark algorithm."""

    def initialize_parameters(self) -> None:
        """Read algorithm-specific parameters from the JSON config dict."""
        self.prefix_char_count = self.config_dict['prefix_char_count']
        self.segment_char_count = self.config_dict['segment_char_count']
        self.bits_per_segment = int(self.config_dict['bits_per_segment'])
        if self.bits_per_segment <= 0:
            raise ValueError('bits_per_segment must be positive')
        mpe = int(self.config_dict.get('max_planted_errors', 0))
        if mpe < 0:
            raise ValueError('max_planted_errors must be non-negative')
        self.max_planted_errors = mpe
        self.total_signature_segments = eth_signature_total_segments(
            self.bits_per_segment, self.max_planted_errors,
        )
        self.top_p = self.config_dict.get('top_p', 0.9)
        self.temperature = self.config_dict.get('temperature', 0.9)
        self.max_retry_seconds = self.config_dict.get('max_retry_seconds', 300)
        self.post_signature_max_chars = self.config_dict.get('post_signature_max_chars', None)
        rp = float(self.config_dict.get('repetition_penalty', 1.0))
        if rp <= 0:
            raise ValueError('repetition_penalty must be positive')
        self.repetition_penalty = rp

        pk = self.config_dict.get('private_key')
        if not pk:
            raise ValueError(
                "ETHWatermark requires 'private_key' in config JSON "
                "(hex string, e.g. from: python -c \"from eth_account import Account; "
                "print(Account.create().key.hex())\")",
            )
        self.private_key = pk

    def log_snapshot(self) -> dict[str, Any]:
        """Serializable view of algorithm + runtime parameters for logging."""
        model = self.generation_model
        model_label = getattr(model, "name_or_path", None) or type(model).__name__
        safe_config = {
            k: v for k, v in self.config_dict.items() if k != 'private_key'
        }
        return {
            **safe_config,
            "total_signature_segments": self.total_signature_segments,
            "signature_total_bits": eth_signature_total_bits(self.max_planted_errors),
            "encoded_signature_bytes": eth_encoded_signature_byte_len(
                self.max_planted_errors,
            ),
            "rs_parity_bytes": eth_rs_parity_byte_count(self.max_planted_errors),
            "device": str(self.device),
            "vocab_size": self.vocab_size,
            "generation_model": model_label,
        }

    @property
    def algorithm_name(self) -> str:
        return 'ETHWatermark'


# ---------------------------------------------------------------------------
# Utilities: crypto helpers + token / character sampling
# ---------------------------------------------------------------------------

class ETHWatermarkUtils:
    """Utility class containing crypto helpers and text generation functions."""

    def __init__(self, config: ETHWatermarkConfig) -> None:
        self.config = config

    @staticmethod
    def token_ends_generation(token_id: int, tokenizer) -> bool:
        """True if *token_id* is an EOS / end-of-turn id for this tokenizer."""
        eos = getattr(tokenizer, "eos_token_id", None)
        if eos is not None:
            if isinstance(eos, (list, tuple)):
                if token_id in eos:
                    return True
            elif token_id == eos:
                return True
        eot = getattr(tokenizer, "eot_id", None)
        if eot is not None and token_id == eot:
            return True
        return False

    # -- Crypto helpers -----------------------------------------------------

    @staticmethod
    def prefix_hash_bytes(prefix_text: str) -> bytes:
        """32-byte SHA-256 digest of the prefix (same *message* as in Fairoze's pipeline)."""
        return hashlib.sha256(prefix_text.encode("utf-8")).digest()

    @staticmethod
    def eth_sign_prefix(prefix_text: str, private_key: str) -> bytes:
        """ECDSA-sign SHA-256(prefix_text) with the given ETH private key.

        Returns the raw 65-byte signature (r ‖ s ‖ v).
        """
        prefix_hash = ETHWatermarkUtils.prefix_hash_bytes(prefix_text)
        signable_message = encode_defunct(primitive=prefix_hash)
        signed = Account.sign_message(signable_message, private_key=private_key)
        return bytes(signed.signature)

    @staticmethod
    def rs_encode_signature(signature_bytes: bytes, max_planted_errors: int) -> bytes:
        """Append Reed–Solomon parity (same parameters as Fairoze ``bytes_to_binary_codeword``)."""
        if max_planted_errors == 0:
            return signature_bytes
        if len(signature_bytes) != ETH_RAW_SIGNATURE_BYTE_LEN:
            raise ValueError(
                f'RS encode expects {ETH_RAW_SIGNATURE_BYTE_LEN}-byte signature, '
                f'got {len(signature_bytes)}'
            )
        nsym = eth_rs_parity_byte_count(max_planted_errors)
        if nsym <= 0:
            raise ValueError(
                'RS enabled but parity byte count is 0; use max_planted_errors >= 8'
            )
        rsc = RSCodec(nsym)
        return bytes(rsc.encode(signature_bytes))

    @staticmethod
    def rs_decode_codeword_ex(
        codeword_bytes: bytes, max_planted_errors: int,
    ) -> tuple[bytes, tuple[int, ...]]:
        """RS decode; returns ``(raw_signature_bytes, errata_byte_indices)``."""
        if max_planted_errors == 0:
            if len(codeword_bytes) != ETH_RAW_SIGNATURE_BYTE_LEN:
                raise ValueError(
                    f'Expected {ETH_RAW_SIGNATURE_BYTE_LEN}-byte signature, '
                    f'got {len(codeword_bytes)}'
                )
            return codeword_bytes, ()
        nsym = eth_rs_parity_byte_count(max_planted_errors)
        if nsym <= 0:
            raise ValueError(
                'RS decode with max_planted_errors > 0 requires nsym = 2 * (mpe // 8) >= 2'
            )
        rsc = RSCodec(nsym)
        decoded_tuple = rsc.decode(codeword_bytes)
        raw = bytes(decoded_tuple[0])
        errata_raw = decoded_tuple[2] if len(decoded_tuple) > 2 else b""
        errata = tuple(int(x) for x in errata_raw) if errata_raw else ()
        return raw, errata

    @staticmethod
    def rs_decode_codeword(codeword_bytes: bytes, max_planted_errors: int) -> bytes:
        """Correct errors and strip parity; returns raw signature bytes."""
        return ETHWatermarkUtils.rs_decode_codeword_ex(
            codeword_bytes, max_planted_errors,
        )[0]

    @staticmethod
    def otp_pad_bytes(message_bytes: bytes, target_byte_length: int) -> bytes:
        """Deterministic one-time pad bytes (Fairoze-style, extended if needed).

        Fairoze XORs the signature with ``SHA512(message)`` where *message* is the
        signed digest (here: 32-byte prefix hash). That yields 64 pad bytes; for
        a 65-byte ECDSA signature we append ``SHA512(message ‖ counter)`` blocks.
        The first 64 bytes match Fairoze exactly when the signature fits in 64 bytes.
        """
        first_block = hashlib.sha512(message_bytes).digest()
        if target_byte_length <= len(first_block):
            return first_block[:target_byte_length]
        pad = bytearray(first_block)
        counter = 0
        while len(pad) < target_byte_length:
            counter += 1
            pad.extend(
                hashlib.sha512(message_bytes + counter.to_bytes(4, "big")).digest()
            )
        return bytes(pad[:target_byte_length])

    @staticmethod
    def mask_signature_with_otp(signature_bytes: bytes, message_bytes: bytes) -> str:
        """XOR signature bytes with the OTP; return the masked bitstring for embedding.

        Same construction as ``crypto.sign_and_encode_openssl`` (without Reed–Solomon):
        ``BitArray(bytes=(sig_byte ^ pad_byte for ...)).bin``.
        """
        pad = ETHWatermarkUtils.otp_pad_bytes(message_bytes, len(signature_bytes))
        masked = bytes(a ^ b for a, b in zip(signature_bytes, pad, strict=True))
        return BitArray(bytes=masked).bin

    @staticmethod
    def unmask_signature_bits(masked_bit_string: str, message_bytes: bytes) -> bytes:
        """Invert ``mask_signature_with_otp`` to recover raw signature bytes."""
        masked_bytes = BitArray(bin=masked_bit_string).bytes
        pad = ETHWatermarkUtils.otp_pad_bytes(message_bytes, len(masked_bytes))
        return bytes(a ^ b for a, b in zip(masked_bytes, pad, strict=True))

    @staticmethod
    def eth_recover_address(prefix_text: str, signature_bytes: bytes) -> str:
        """Recover the signer's ETH address from a prefix and its signature."""
        prefix_hash = ETHWatermarkUtils.prefix_hash_bytes(prefix_text)
        signable_message = encode_defunct(primitive=prefix_hash)
        return Account.recover_message(signable_message, signature=signature_bytes)

    @staticmethod
    def unkeyed_hash_to_bits(input_bytes: bytes, bit_count: int) -> str:
        """SHA-256 hash truncated to the first *bit_count* bits (binary string).

        Same logic as Fairoze's crypto.unkeyed_hash_to_bits.
        """
        assert bit_count <= 256
        return BitArray(bytes=hashlib.sha256(input_bytes).digest()).bin[:bit_count]

    @staticmethod
    def apply_repetition_penalty(
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        penalty: float,
        vocab_size: int,
    ) -> None:
        """Downweight logits for tokens in *input_ids* (HuggingFace-style). Mutates *logits*.

        For each prior token id, if logit > 0 divide by *penalty*, else multiply.
        Applied once per occurrence (same as ``RepetitionPenaltyLogitsProcessor``).
        """
        if penalty == 1.0:
            return
        for batch_idx in range(logits.shape[0]):
            for tid in input_ids[batch_idx]:
                t = int(tid.item())
                if t < 0 or t >= vocab_size:
                    continue
                score = logits[batch_idx, t]
                logits[batch_idx, t] = torch.where(
                    score < 0, score * penalty, score / penalty
                )

    # -- Token sampling with KV cache --------------------------------------

    def sample_one_token(
        self,
        model,
        tokenizer,
        input_ids: torch.Tensor,
        kv_cache,
        attention_mask: torch.Tensor,
        vocab_size: int,
        allow_stop_token: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, object, torch.Tensor]:
        """Sample a single token using nucleus (top-p) sampling.

        Uses the KV cache so only the last token is fed on subsequent calls.
        Returns (token_tensor, updated_input_ids, updated_kv_cache, updated_attention_mask).
        """
        deadline = time.time() + self.config.max_retry_seconds

        while True:
            if time.time() >= deadline:
                raise RuntimeError(
                    f"sample_one_token: exceeded {self.config.max_retry_seconds}s "
                    "(e.g. EOS resampling during prefix or segment)"
                )

            with torch.no_grad():
                if kv_cache is not None:
                    output = model(
                        input_ids[:, -1:],
                        past_key_values=kv_cache,
                        attention_mask=attention_mask,
                    )
                else:
                    output = model(input_ids)

            logits = output.logits[:, -1, :vocab_size].clone()
            self.apply_repetition_penalty(
                logits, input_ids, self.config.repetition_penalty, vocab_size,
            )

            # --- Nucleus sampling ---
            temperature = self.config.temperature
            top_p = self.config.top_p

            scaled_logits = logits / temperature
            sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens whose cumulative probability exceeds top_p,
            # but always keep the most probable token.
            indices_to_remove = cumulative_probs > top_p
            indices_to_remove[..., 1:] = indices_to_remove[..., :-1].clone()
            indices_to_remove[..., 0] = False

            remove_mask = sorted_indices[indices_to_remove]
            logits[..., remove_mask] = float("-inf")

            probs = torch.softmax(logits, dim=-1)
            token = torch.multinomial(probs, num_samples=1).view(1, 1)
            new_input_ids = torch.cat([input_ids, token], dim=-1)
            tid = int(token.view(-1)[0].item())

            if not allow_stop_token and self.token_ends_generation(tid, tokenizer):
                continue

            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))],
                dim=-1,
            )
            return token, new_input_ids, output.past_key_values, attention_mask

    @staticmethod
    def decode_new_token(
        prev_input_ids: torch.Tensor,
        new_input_ids: torch.Tensor,
        tokenizer,
    ) -> str:
        """Decode the characters added by the last token.

        Decodes the full sequence before and after appending the token,
        then returns the difference. This correctly handles subword
        tokenisers that may merge whitespace across token boundaries.
        """
        prev_text = tokenizer.decode(prev_input_ids.squeeze().detach().cpu())
        new_text = tokenizer.decode(new_input_ids.squeeze().detach().cpu())
        return new_text[len(prev_text):]

    def sample_n_characters(
        self,
        char_count: int,
        initial_overflow: str,
        model,
        tokenizer,
        input_ids: torch.Tensor,
        kv_cache,
        attention_mask: torch.Tensor,
        vocab_size: int,
        allow_stop_token: bool,
    ) -> tuple[str, str, torch.Tensor, object, torch.Tensor]:
        """Grow text up to *char_count* characters (may stop early on EOS).

        When ``allow_stop_token`` is True and the sampled token is EOS, stop
        sampling. Overflow past *char_count* is returned for the next segment boundary.
        """
        segment_text = initial_overflow
        overflow_text = ""

        while len(segment_text) < char_count:
            prev_input_ids = input_ids

            token, input_ids, kv_cache, attention_mask = self.sample_one_token(
                model,
                tokenizer,
                input_ids,
                kv_cache,
                attention_mask,
                vocab_size,
                allow_stop_token,
            )

            tid = int(token.view(-1)[0].item())
            if allow_stop_token and self.token_ends_generation(tid, tokenizer):
                break

            token_chars = self.decode_new_token(prev_input_ids, input_ids, tokenizer)
            segment_text += token_chars

            # If we overshot, split off the extra characters
            if len(segment_text) > char_count:
                overflow_text = segment_text[char_count:]
                segment_text = segment_text[:char_count]

        return segment_text, overflow_text, input_ids, kv_cache, attention_mask

    def sample_until_eos(
        self,
        initial_overflow: str,
        model,
        tokenizer,
        input_ids: torch.Tensor,
        kv_cache,
        attention_mask: torch.Tensor,
        vocab_size: int,
        max_chars: int | None = None,
    ) -> tuple[str, torch.Tensor, object, torch.Tensor]:
        """Append tokens until EOS, or until *max_chars* UTF-8 characters (if set).

        After signature embedding, ``max_chars`` caps the tail length (including
        ``initial_overflow``). If ``max_chars`` is None, runs until EOS only.
        Uses ``max_retry_seconds`` as a wall-clock bound if EOS is never sampled
        and no character cap applies.
        """
        out = initial_overflow
        deadline = time.time() + self.config.max_retry_seconds

        while True:
            if max_chars is not None and len(out) >= max_chars:
                break
            if time.time() >= deadline:
                raise RuntimeError(
                    f"sample_until_eos: no EOS within {self.config.max_retry_seconds}s"
                )

            prev_input_ids = input_ids
            token, input_ids, kv_cache, attention_mask = self.sample_one_token(
                model,
                tokenizer,
                input_ids,
                kv_cache,
                attention_mask,
                vocab_size,
                allow_stop_token=True,
            )
            tid = int(token.view(-1)[0].item())
            if self.token_ends_generation(tid, tokenizer):
                break
            token_chars = self.decode_new_token(prev_input_ids, input_ids, tokenizer)
            out += token_chars

        return out, input_ids, kv_cache, attention_mask


# ---------------------------------------------------------------------------
# Main watermark class
# ---------------------------------------------------------------------------

class ETHWatermark(BaseWatermark):
    """Publicly detectable watermark using Ethereum ECDSA signatures.

    Generation:
        1. Generate a prefix of configurable length (plain LM text).
        2. Sign SHA-256(prefix) with the caller-supplied ETH private key.
        3. Optionally Reed–Solomon-encode the signature (``max_planted_errors``).
        4. XOR-mask the codeword with SHA-512(prefix_hash) (Fairoze OTP; length
           matches the codeword).
        5. Embed the masked bitstring into subsequent text via a hash chain.
    Detection:
        Rebuild masked bits, unmask, RS-decode if configured, recover the ETH address.
    """

    def __init__(
        self,
        algorithm_config: str | ETHWatermarkConfig,
        transformers_config: TransformersConfig | None = None,
        *args,
        **kwargs,
    ) -> None:
        if isinstance(algorithm_config, str):
            self.config = ETHWatermarkConfig(algorithm_config, transformers_config)
        elif isinstance(algorithm_config, ETHWatermarkConfig):
            self.config = algorithm_config
        else:
            raise TypeError(
                "algorithm_config must be a path string or an ETHWatermarkConfig instance"
            )
        self.utils = ETHWatermarkUtils(self.config)
        self.eth_address = Account.from_key(self.config.private_key).address

    # ----- generation ------------------------------------------------------

    def generate_watermarked_text(self, prompt: str, *args, **kwargs) -> str:
        """Generate watermarked text with an embedded ETH signature.

        Uses ``private_key`` from ``config/ETHWatermark.json`` by default.
        Optional kwargs ``private_key`` overrides the config value (e.g. for demos).
        """
        private_key = kwargs.get("private_key") or self.config.private_key
        if not private_key:
            raise ValueError(
                "private_key must be set in config/ETHWatermark.json or passed as a keyword argument",
            )

        log_file = kwargs.get("log_file")
        with _optional_file_handler_for_logger(logger, log_file):
            model = self.config.generation_model
            tokenizer = self.config.generation_tokenizer
            vocab_size = self.config.vocab_size
            device = self.config.device

            prefix_char_count = self.config.prefix_char_count
            segment_char_count = self.config.segment_char_count
            bits_per_segment = self.config.bits_per_segment
            max_retry_seconds = self.config.max_retry_seconds

            run_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            logger.info("ETHWatermark generate start run_id=%s", run_id)
            logger.info(
                "ETHWatermark config snapshot: %s",
                json.dumps(self.config.log_snapshot(), default=str, sort_keys=True),
            )

            # Encode the prompt and initialise the KV-cache state
            input_ids = tokenizer.encode(
                prompt, return_tensors="pt", add_special_tokens=True,
            ).to(device)
            attention_mask = torch.ones_like(input_ids)
            kv_cache = None

            # -- Step 1: generate the prefix - unwatermarked text (reject EOS until prefix is done)
            prefix_text, segment_overflow, input_ids, kv_cache, attention_mask = (
                self.utils.sample_n_characters(
                    prefix_char_count,
                    "",
                    model,
                    tokenizer,
                    input_ids,
                    kv_cache,
                    attention_mask,
                    vocab_size,
                    allow_stop_token=False,
                )
            )

            # -- Step 2: sign the prefix and OTP-mask (Fairoze-style) ------------
            signature_bytes = self.utils.eth_sign_prefix(prefix_text, private_key)
            if len(signature_bytes) != ETH_RAW_SIGNATURE_BYTE_LEN:
                raise ValueError(
                    f'Expected {ETH_RAW_SIGNATURE_BYTE_LEN}-byte Ethereum raw signature, '
                    f'got {len(signature_bytes)} bytes'
                )
            total_segments = self.config.total_signature_segments
            max_planted_errors = self.config.max_planted_errors

            message_for_otp = self.utils.prefix_hash_bytes(prefix_text)
            signature_codeword = self.utils.rs_encode_signature(
                signature_bytes, max_planted_errors,
            )
            signature_bits = self.utils.mask_signature_with_otp(
                signature_codeword, message_for_otp,
            )

            # -- Step 3: embed signature bits via hash-chain --------------------
            watermarked_text = prefix_text
            accumulated_hash_bits = ""
            bit_offset = 0
            segment_retry_counts: list[int] = []
            planted_bit_errors = 0

            for segment_index in tqdm(range(total_segments), desc="Embedding signature"):
                k = eth_bits_for_signature_segment(
                    segment_index, bits_per_segment, max_planted_errors,
                )
                target_bits = signature_bits[bit_offset : bit_offset + k]
                bit_offset += k

                # Snapshot KV state so we can roll back on hash mismatch
                saved_input_ids = input_ids
                saved_kv_cache = kv_cache
                saved_attention_mask = attention_mask
                saved_overflow = segment_overflow

                segment_accepted = False
                retry_start_time = time.time()
                attempts_for_segment = 0
                best_match_bits = -1
                best_segment_text = ""
                best_next_overflow = ""
                best_input_ids = input_ids
                best_kv_cache = kv_cache
                best_attention_mask = attention_mask
                best_computed_hash_bits = ""

                while not segment_accepted:
                    elapsed = time.time() - retry_start_time
                    if best_match_bits >= 0 and elapsed >= max_retry_seconds:
                        if max_planted_errors <= 0:
                            raise RuntimeError(
                                f"Segment {segment_index}: could not find a hash match "
                                f"for target bits '{target_bits}' within {max_retry_seconds}s"
                            )
                        cost_timeout = eth_bit_mismatch_cost(
                            best_computed_hash_bits, target_bits,
                        )
                        if planted_bit_errors + cost_timeout <= max_planted_errors:
                            segment_text = best_segment_text
                            next_overflow = best_next_overflow
                            input_ids = best_input_ids
                            kv_cache = best_kv_cache
                            attention_mask = best_attention_mask
                            computed_hash_bits = best_computed_hash_bits
                            planted_bit_errors += cost_timeout
                            segment_accepted = True
                            logger.info(
                                "Segment %d: timeout force accept bit_mismatch_cost=%d "
                                "total_planted_bit_errors=%d",
                                segment_index + 1,
                                cost_timeout,
                                planted_bit_errors,
                            )
                            break
                        raise RuntimeError(
                            f"Segment {segment_index}: {max_retry_seconds}s elapsed, "
                            f"no exact match and insufficient bit-error budget "
                            f"(planted_bit_errors={planted_bit_errors}, "
                            f"best_possible_cost={cost_timeout}, N={max_planted_errors})"
                        )

                    attempts_for_segment += 1

                    # Restore KV state to the boundary before this segment
                    input_ids = saved_input_ids
                    kv_cache = saved_kv_cache
                    attention_mask = saved_attention_mask

                    # Generate one segment (no stop token until all segments are done)
                    segment_text, next_overflow, input_ids, kv_cache, attention_mask = (
                        self.utils.sample_n_characters(
                            segment_char_count,
                            saved_overflow,
                            model,
                            tokenizer,
                            input_ids,
                            kv_cache,
                            attention_mask,
                            vocab_size,
                            allow_stop_token=False,
                        )
                    )

                    # Compute hash chain: H(prefix ‖ accumulated_hash_bits ‖ segment)
                    hash_input = (
                        prefix_text.encode("utf-8")
                        + accumulated_hash_bits.encode("utf-8")
                        + segment_text.encode("utf-8")
                    )
                    computed_hash_bits = self.utils.unkeyed_hash_to_bits(hash_input, k)

                    matches = eth_count_matching_bits(computed_hash_bits, target_bits)
                    if matches > best_match_bits:
                        best_match_bits = matches
                        best_segment_text = segment_text
                        best_next_overflow = next_overflow
                        best_input_ids = input_ids
                        best_kv_cache = kv_cache
                        best_attention_mask = attention_mask
                        best_computed_hash_bits = computed_hash_bits

                    if computed_hash_bits == target_bits:
                        segment_accepted = True
                        break

                    min_tries = eth_min_tries_before_force_accept(
                        planted_bit_errors, max_planted_errors,
                    )
                    if min_tries is not None and attempts_for_segment >= min_tries:
                        cost = k - best_match_bits
                        if planted_bit_errors + cost <= max_planted_errors:
                            segment_text = best_segment_text
                            next_overflow = best_next_overflow
                            input_ids = best_input_ids
                            kv_cache = best_kv_cache
                            attention_mask = best_attention_mask
                            computed_hash_bits = best_computed_hash_bits
                            planted_bit_errors += cost
                            segment_accepted = True
                            logger.info(
                                "Segment %d: force accept after %d attempt(s) "
                                "bit_mismatch_cost=%d total_planted_bit_errors=%d",
                                segment_index + 1,
                                attempts_for_segment,
                                cost,
                                planted_bit_errors,
                            )
                            break

                retries = attempts_for_segment - 1
                segment_retry_counts.append(retries)
                logger.info(
                    "Segment %d of %d embedded: bits_this_segment=%d attempts=%d retries=%d "
                    "target_bits=%s planted_bit_errors_total=%d",
                    segment_index + 1,
                    total_segments,
                    k,
                    attempts_for_segment,
                    retries,
                    target_bits,
                    planted_bit_errors,
                )

                # Segment accepted — advance the chain
                accumulated_hash_bits += computed_hash_bits
                watermarked_text += segment_text
                segment_overflow = next_overflow

            logger.info(
                "Signature embedding finished: total_segments=%d total_attempts=%d "
                "total_retries=%d planted_bit_errors=%d per_segment_retries=%s",
                total_segments,
                sum(r + 1 for r in segment_retry_counts),
                sum(segment_retry_counts),
                planted_bit_errors,
                segment_retry_counts,
            )

            tail_text, input_ids, kv_cache, attention_mask = self.utils.sample_until_eos(
                segment_overflow,
                model,
                tokenizer,
                input_ids,
                kv_cache,
                attention_mask,
                vocab_size,
                max_chars=self.config.post_signature_max_chars,
            )
            watermarked_text += tail_text

            logger.info(
                "ETHWatermark generate finished run_id=%s output_char_len=%d",
                run_id,
                len(watermarked_text),
            )
            return watermarked_text

    def _try_recover_at_segments(
        self,
        prefix_text: str,
        signature_region: str,
        segment_char_count: int,
        bits_per_segment: int,
        total_segments: int,
        max_planted_errors: int,
    ) -> tuple[str, tuple[int, ...]] | None:
        """Rebuild hash chain; return ``(address, rs_errata_byte_indices)`` or None."""
        accumulated_hash_bits = ""
        for seg_idx in range(total_segments):
            k = eth_bits_for_signature_segment(
                seg_idx, bits_per_segment, max_planted_errors,
            )
            seg_start = seg_idx * segment_char_count
            segment_text = signature_region[seg_start : seg_start + segment_char_count]
            if len(segment_text) < segment_char_count:
                return None
            hash_input = (
                prefix_text.encode('utf-8')
                + accumulated_hash_bits.encode('utf-8')
                + segment_text.encode('utf-8')
            )
            computed_hash_bits = self.utils.unkeyed_hash_to_bits(hash_input, k)
            accumulated_hash_bits += computed_hash_bits

        try:
            message_for_otp = self.utils.prefix_hash_bytes(prefix_text)
            masked_codeword = self.utils.unmask_signature_bits(
                accumulated_hash_bits, message_for_otp,
            )
            recovered_signature, rs_errata = self.utils.rs_decode_codeword_ex(
                masked_codeword, max_planted_errors,
            )
            addr = self.utils.eth_recover_address(prefix_text, recovered_signature)
            return addr, rs_errata
        except Exception:
            return None

    # ----- detection -------------------------------------------------------

    def detect_watermark(
        self, text: str, return_dict: bool = True, *args, **kwargs,
    ) -> Union[dict, tuple]:
        """Extract the embedded signature and recover the signer's ETH address.

        Tries every character rotation of *text* as a potential prefix start.
        Returns ``{"is_watermarked": bool, "recovered_address": str | None}``.

        Optional keyword argument ``log_file``: append detection logs to this path.
        """
        log_file = kwargs.get("log_file")
        with _optional_file_handler_for_logger(logger, log_file):
            return self._detect_watermark_impl(text, return_dict, **kwargs)

    def _detect_watermark_impl(
        self, text: str, return_dict: bool, **kwargs,
    ) -> Union[dict, tuple]:
        run_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        logger.info("ETHWatermark detect start run_id=%s text_char_len=%d", run_id, len(text))
        logger.info(
            "ETHWatermark config snapshot: %s",
            json.dumps(self.config.log_snapshot(), default=str, sort_keys=True),
        )

        prefix_char_count = self.config.prefix_char_count
        segment_char_count = self.config.segment_char_count
        bits_per_segment = self.config.bits_per_segment
        total_segments = self.config.total_signature_segments
        max_planted_errors = self.config.max_planted_errors

        min_length = prefix_char_count + segment_char_count * total_segments
        if len(text) < min_length:
            logger.info(
                "Detect abort: text shorter than minimum %d (got %d)",
                min_length,
                len(text),
            )
            if return_dict:
                return {
                    "is_watermarked": False,
                    "score": 0.0,
                    "recovered_address": None,
                }
            return (False, None)

        rotations_tried = 0
        for rotation_offset in range(len(text)):
            rotated_text = text[rotation_offset:] + text[:rotation_offset]

            if len(rotated_text) < min_length:
                continue

            rotations_tried += 1
            prefix_text = rotated_text[:prefix_char_count]
            signature_region = rotated_text[prefix_char_count:]

            result = self._try_recover_at_segments(
                prefix_text,
                signature_region,
                segment_char_count,
                bits_per_segment,
                total_segments,
                max_planted_errors,
            )
            if result is not None:
                recovered_address, rs_errata = result
                logger.info(
                    "Detect success: rotation_offset=%d recovered_address=%s "
                    "rs_codeword_byte_errata=%s rs_errata_count=%d",
                    rotation_offset,
                    recovered_address,
                    rs_errata,
                    len(rs_errata),
                )
                logger.info("ETHWatermark detect finished run_id=%s success=True", run_id)
                if return_dict:
                    return {
                        "is_watermarked": True,
                        "score": 1.0,
                        "recovered_address": recovered_address,
                    }
                return (True, recovered_address)
            logger.debug("Detect miss at rotation_offset=%d", rotation_offset)

        logger.info(
            "Detect failure: tried %d rotation offset(s), no valid signature",
            rotations_tried,
        )
        logger.info("ETHWatermark detect finished run_id=%s success=False", run_id)
        if return_dict:
            return {
                "is_watermarked": False,
                "score": 0.0,
                "recovered_address": None,
            }
        return (False, None)
