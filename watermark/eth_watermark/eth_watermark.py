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
        self.signature_byte_length = self.config_dict['signature_byte_length']
        self.top_p = self.config_dict.get('top_p', 0.9)
        self.temperature = self.config_dict.get('temperature', 0.9)
        self.max_retry_seconds = self.config_dict.get('max_retry_seconds', 300)

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

    # -- Token sampling with KV cache --------------------------------------

    def sample_one_token(
        self,
        model,
        tokenizer,
        input_ids: torch.Tensor,
        kv_cache,
        attention_mask: torch.Tensor,
        vocab_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, object, torch.Tensor]:
        """Sample a single token using nucleus (top-p) sampling.

        Uses the KV cache so only the last token is fed on subsequent calls.
        Returns (token_tensor, updated_input_ids, updated_kv_cache, updated_attention_mask).
        """
        with torch.no_grad():
            if kv_cache is not None:
                output = model(
                    input_ids[:, -1:],
                    past_key_values=kv_cache,
                    attention_mask=attention_mask,
                )
            else:
                output = model(input_ids)

        logits = output.logits[:, -1, :vocab_size]

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
        token = torch.multinomial(probs, num_samples=1)

        # Append the sampled token to the running sequence
        token = token.view(1, 1)
        input_ids = torch.cat([input_ids, token], dim=-1)
        kv_cache = output.past_key_values
        attention_mask = torch.cat(
            [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))],
            dim=-1,
        )

        return token, input_ids, kv_cache, attention_mask

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
    ) -> tuple[str, str, torch.Tensor, object, torch.Tensor]:
        """Generate exactly *char_count* characters of text.

        Any characters beyond the required count are returned as *overflow*
        so the next segment can start from them (preserving token boundaries).

        Returns (segment_text, overflow_text, input_ids, kv_cache, attention_mask).
        """
        segment_text = initial_overflow
        overflow_text = ""

        while len(segment_text) < char_count:
            prev_input_ids = input_ids

            token, input_ids, kv_cache, attention_mask = self.sample_one_token(
                model, tokenizer, input_ids, kv_cache, attention_mask, vocab_size,
            )

            token_chars = self.decode_new_token(prev_input_ids, input_ids, tokenizer)
            segment_text += token_chars

            # If we overshot, split off the extra characters
            if len(segment_text) > char_count:
                overflow_text = segment_text[char_count:]
                segment_text = segment_text[:char_count]

        return segment_text, overflow_text, input_ids, kv_cache, attention_mask


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
        signature_byte_length = self.config.signature_byte_length
        max_retry_seconds = self.config.max_retry_seconds

        total_signature_bits = signature_byte_length * 8
        total_segments = total_signature_bits // bits_per_segment

        # Encode the prompt and initialise the KV-cache state
        input_ids = tokenizer.encode(
            prompt, return_tensors="pt", add_special_tokens=True,
        ).to(device)
        attention_mask = torch.ones_like(input_ids)
        kv_cache = None

        # -- Step 1: generate the prefix (unwatermarked LM text) ------------
        prefix_text, segment_overflow, input_ids, kv_cache, attention_mask = (
            self.utils.sample_n_characters(
                prefix_char_count, "", model, tokenizer,
                input_ids, kv_cache, attention_mask, vocab_size,
            )
        )

        # -- Step 2: sign the prefix and OTP-mask (Fairoze-style) ------------
        signature_bytes = self.utils.eth_sign_prefix(prefix_text, private_key)
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

                # Generate one segment of text
                segment_text, next_overflow, input_ids, kv_cache, attention_mask = (
                    self.utils.sample_n_characters(
                        segment_char_count, saved_overflow, model, tokenizer,
                        input_ids, kv_cache, attention_mask, vocab_size,
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

        return watermarked_text

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
        signature_byte_length = self.config.signature_byte_length
        total_signature_bits = signature_byte_length * 8
        total_segments = total_signature_bits // bits_per_segment

        # Minimum text length: prefix + enough characters for all segments
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

            # Walk fixed-size segments and rebuild the hash chain
            accumulated_hash_bits = ""
            segment_count = 0

            for seg_start in range(0, len(signature_region), segment_char_count):
                if segment_count >= total_segments:
                    break

                segment_text = signature_region[seg_start : seg_start + segment_char_count]
                if len(segment_text) < segment_char_count:
                    break

                hash_input = (
                    prefix_text.encode("utf-8")
                    + accumulated_hash_bits.encode("utf-8")
                    + segment_text.encode("utf-8")
                )
                computed_hash_bits = self.utils.unkeyed_hash_to_bits(
                    hash_input, bits_per_segment,
                )
                accumulated_hash_bits += computed_hash_bits
                segment_count += 1

            if segment_count < total_segments:
                continue

            # Unmask OTP then recover signature bytes and signer address
            try:
                message_for_otp = self.utils.prefix_hash_bytes(prefix_text)
                recovered_signature = self.utils.unmask_signature_bits(
                    accumulated_hash_bits, message_for_otp,
                )
                recovered_address = self.utils.eth_recover_address(
                    prefix_text, recovered_signature,
                )
            except Exception:
                continue

            if return_dict:
                return {"is_watermarked": True, "recovered_address": recovered_address}
            return (True, recovered_address)

        if return_dict:
            return {"is_watermarked": False, "recovered_address": None}
        return (False, None)
