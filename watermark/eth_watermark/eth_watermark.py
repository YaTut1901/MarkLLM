# ===========================================================================
# eth_watermark.py
# Description: Implementation of ETH publicly detectable watermark algorithm.
#              Adapted from Fairoze's publicly-detectable-watermark, using
#              Ethereum ECDSA (secp256k1) signatures instead of BLS.
#              Embeds a signature of the generated prefix into subsequent
#              text via a chain of hashes over fixed-length character segments.
# ===========================================================================

import hashlib
import time
from typing import Union

import torch
from bitstring import BitArray
from eth_account import Account
from eth_account.messages import encode_defunct
from tqdm import tqdm

from ..base import BaseWatermark, BaseConfig
from utils.transformers_config import TransformersConfig

# ``eth_account`` raw ECDSA signatures are 65 bytes (r ‖ s ‖ v) for standard secp256k1.
ETH_RAW_SIGNATURE_BYTE_LEN = 65


def eth_signature_total_bits() -> int:
    return ETH_RAW_SIGNATURE_BYTE_LEN * 8


def eth_signature_total_segments(bits_per_segment: int) -> int:
    """Segment count for embedding/detecting one Ethereum raw signature."""
    tb = eth_signature_total_bits()
    if tb % bits_per_segment != 0:
        raise ValueError(
            f'bits_per_segment={bits_per_segment} must divide ETH signature size '
            f'({tb} bits = {ETH_RAW_SIGNATURE_BYTE_LEN} bytes)'
        )
    return tb // bits_per_segment


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class ETHWatermarkConfig(BaseConfig):
    """Config class for ETHWatermark algorithm."""

    def initialize_parameters(self) -> None:
        """Read algorithm-specific parameters from the JSON config dict."""
        self.prefix_char_count = self.config_dict['prefix_char_count']
        self.segment_char_count = self.config_dict['segment_char_count']
        self.bits_per_segment = self.config_dict['bits_per_segment']
        self.total_signature_segments = eth_signature_total_segments(self.bits_per_segment)
        self.top_p = self.config_dict.get('top_p', 0.9)
        self.temperature = self.config_dict.get('temperature', 0.9)
        self.max_retry_seconds = self.config_dict.get('max_retry_seconds', 300)
        self.post_signature_max_chars = self.config_dict.get('post_signature_max_chars', None)
        rp = float(self.config_dict.get('repetition_penalty', 1.0))
        if rp <= 0:
            raise ValueError('repetition_penalty must be positive')
        self.repetition_penalty = rp

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
        3. XOR-mask the signature with SHA-512(prefix_hash) (Fairoze OTP; extended
           to 65 bytes for ECDSA length).
        4. Embed the masked 520-bit string into subsequent text via a hash chain.
    Detection:
        Rebuild masked bits, unmask with the same OTP, recover the ETH address.
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

    # ----- generation ------------------------------------------------------

    def generate_watermarked_text(self, prompt: str, *args, **kwargs) -> str:
        """Generate watermarked text with an embedded ETH signature.

        The caller **must** pass ``private_key`` as a keyword argument
        (hex string or bytes of the ETH private key used for signing).
        """
        private_key = kwargs.get("private_key")
        if private_key is None:
            raise ValueError("private_key must be provided as a keyword argument")

        model = self.config.generation_model
        tokenizer = self.config.generation_tokenizer
        vocab_size = self.config.vocab_size
        device = self.config.device

        prefix_char_count = self.config.prefix_char_count
        segment_char_count = self.config.segment_char_count
        bits_per_segment = self.config.bits_per_segment
        max_retry_seconds = self.config.max_retry_seconds

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

        message_for_otp = self.utils.prefix_hash_bytes(prefix_text)
        signature_bits = self.utils.mask_signature_with_otp(
            signature_bytes, message_for_otp,
        )

        # -- Step 3: embed signature bits via hash-chain --------------------
        watermarked_text = prefix_text
        accumulated_hash_bits = ""

        for segment_index in tqdm(range(total_segments), desc="Embedding signature"):
            # Slice the next target chunk from the signature bitstring
            bit_offset = segment_index * bits_per_segment
            target_bits = signature_bits[bit_offset : bit_offset + bits_per_segment]

            # Snapshot KV state so we can roll back on hash mismatch
            saved_input_ids = input_ids
            saved_kv_cache = kv_cache
            saved_attention_mask = attention_mask
            saved_overflow = segment_overflow

            segment_accepted = False
            retry_start_time = time.time()

            while not segment_accepted:
                elapsed = time.time() - retry_start_time
                if elapsed >= max_retry_seconds:
                    raise RuntimeError(
                        f"Segment {segment_index}: could not find a hash match "
                        f"for target bits '{target_bits}' within {max_retry_seconds}s"
                    )

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
                computed_hash_bits = self.utils.unkeyed_hash_to_bits(
                    hash_input, bits_per_segment,
                )

                if computed_hash_bits == target_bits:
                    segment_accepted = True

            # Segment accepted — advance the chain
            accumulated_hash_bits += computed_hash_bits
            watermarked_text += segment_text
            segment_overflow = next_overflow

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

        return watermarked_text

    def _try_recover_at_segments(
        self,
        prefix_text: str,
        signature_region: str,
        segment_char_count: int,
        bits_per_segment: int,
        total_segments: int,
    ) -> str | None:
        """Rebuild hash chain for *total_segments* segments; return address or None."""
        accumulated_hash_bits = ""
        for seg_idx in range(total_segments):
            seg_start = seg_idx * segment_char_count
            segment_text = signature_region[seg_start : seg_start + segment_char_count]
            if len(segment_text) < segment_char_count:
                return None
            hash_input = (
                prefix_text.encode('utf-8')
                + accumulated_hash_bits.encode('utf-8')
                + segment_text.encode('utf-8')
            )
            computed_hash_bits = self.utils.unkeyed_hash_to_bits(
                hash_input, bits_per_segment,
            )
            accumulated_hash_bits += computed_hash_bits

        try:
            message_for_otp = self.utils.prefix_hash_bytes(prefix_text)
            recovered_signature = self.utils.unmask_signature_bits(
                accumulated_hash_bits, message_for_otp,
            )
            return self.utils.eth_recover_address(prefix_text, recovered_signature)
        except Exception:
            return None

    # ----- detection -------------------------------------------------------

    def detect_watermark(
        self, text: str, return_dict: bool = True, *args, **kwargs,
    ) -> Union[dict, tuple]:
        """Extract the embedded signature and recover the signer's ETH address.

        Tries every character rotation of *text* as a potential prefix start.
        Returns ``{"is_watermarked": bool, "recovered_address": str | None}``.
        """
        prefix_char_count = self.config.prefix_char_count
        segment_char_count = self.config.segment_char_count
        bits_per_segment = self.config.bits_per_segment
        total_segments = self.config.total_signature_segments

        min_length = prefix_char_count + segment_char_count * total_segments
        if len(text) < min_length:
            if return_dict:
                return {"is_watermarked": False, "recovered_address": None}
            return (False, None)

        # Try each rotation of the text as a potential watermark start
        for rotation_offset in range(len(text)):
            rotated_text = text[rotation_offset:] + text[:rotation_offset]

            if len(rotated_text) < min_length:
                continue

            prefix_text = rotated_text[:prefix_char_count]
            signature_region = rotated_text[prefix_char_count:]

            recovered_address = self._try_recover_at_segments(
                prefix_text,
                signature_region,
                segment_char_count,
                bits_per_segment,
                total_segments,
            )
            if recovered_address is not None:
                if return_dict:
                    return {
                        "is_watermarked": True,
                        "recovered_address": recovered_address,
                    }
                return (True, recovered_address)

        if return_dict:
            return {"is_watermarked": False, "recovered_address": None}
        return (False, None)
