# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""This example demonstrates instantiating vLLM with a custom logits processor
class object.

For a basic example of implementing a custom logits processor, see
the `DummyLogitsProcessor` implementation in `vllm/test_utils.py`.

For testing purposes, a dummy logits processor is employed which, if
`target_token` is passed as a keyword argument to `SamplingParams.extra_args`,
will mask out all tokens except `target_token`.

A batch is constructed with `temperature=0.0` and 50% of requests specifying
`target_token`, and for these requests - and *only* these requests - we
expect the `target_token` to be decoded in each step, yielding an output
similar to that shown below:

Generated Outputs:
------------------------------------------------------------
Prompt:    'Hello, my name is'
Output:    " ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' '"
------------------------------------------------------------
Prompt:    'The president of the United States is'
Output:    " not a racist. He is a racist.\nHe's a racist because he"
------------------------------------------------------------
Prompt:    'The capital of France is'
Output:    ' also also also also also also also also also also also also also
             also also also'
------------------------------------------------------------
Prompt:    'The future of AI is'
Output:    ' in the hands of the people.\n\nThe future of AI is in the'
------------------------------------------------------------
"""

from typing import Any

import torch

from vllm import SamplingParams
from vllm.config import VllmConfig
from vllm.v1.sample.logits_processor import (
    BatchUpdate,
    LogitsProcessor,
    MoveDirectionality,
)
# Hypothetical custom logits processor
class KGWLogitsProcessor(LogitsProcessor):
    """Fake logit processor to support unit testing and examples"""

    @classmethod
    def validate_params(cls, params: SamplingParams):
        extra = getattr(params, "extra_args", None) or {}
        
        if not extra.get("kgw_enable", False):
            return None
        else:
            gamma = float(extra.get("kgw_gamma", 0.5))
            if not (0.0 < gamma < 1.0):
                raise ValueError("kgw_gamma must be in (0, 1)")
        
            delta = float(extra.get("kgw_delta", 2.0))
            if delta < 0:
                raise ValueError("kgw_delta must be >= 0")
        
            window_size = int(extra.get("kgw_window_size", 1))
            if window_size <= 0:
                raise ValueError("kgw_window_size must be > 0")
            
            return dict({
                "gamma": gamma,
                "delta": delta,
                "hash_key": 15485863,
                "prefix_length": window_size,
                "f_scheme": "time",
                "window_scheme": "left"
            })
        
        

    def __init__(
        self, vllm_config: VllmConfig, device: torch.device, is_pin_memory: bool
    ):
        self.req_info: dict[int, dict[str, Any]] = {}
        

    def is_argmax_invariant(self) -> bool:
        return False

    def update_state(self, batch_update: BatchUpdate | None):
        if not batch_update:
            return

        # Process added requests.
        for index, params, prompt_ids, output_ids in batch_update.added:
            assert params is not None
            kgw_params = self.validate_params(params)
            if kgw_params is not None:
                self.req_info[index] = kgw_params | {"output_ids_in_window": output_ids} | {"prompt_ids": prompt_ids}
            else: 
                self.req_info.pop(index, None)

        if self.req_info:
            # Process removed requests.
            for index in batch_update.removed:
                self.req_info.pop(index, None)

            # Process moved requests, unidirectional move (a->b) and swap
            # (a<->b)
            for adx, bdx, direct in batch_update.moved:
                a_val = self.req_info.pop(adx, None)
                b_val = self.req_info.pop(bdx, None)
                if a_val is not None:
                    self.req_info[bdx] = a_val
                if direct == MoveDirectionality.SWAP and b_val is not None:
                    self.req_info[adx] = b_val
        

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.req_info:
            
            return logits

        batch_size = logits.shape[0]
        window_size = next(iter(self.req_info.values()))["prefix_length"]
        gamma = next(iter(self.req_info.values()))["gamma"]
        delta = next(iter(self.req_info.values()))["delta"]

        mask = torch.ones(batch_size, dtype=torch.long, device=logits.device)
        output_ids_matrix = torch.zeros(
            (batch_size, window_size), dtype=torch.long, device=logits.device
        )

        for i in range(batch_size):
            
            req = self.req_info.get(i)
            if req is None:
                mask[i] = 0
                output_ids_matrix[i] = torch.zeros(window_size, dtype=torch.long, device=logits.device)
                continue
            else:
                window_ids = req["prompt_ids"] + req["output_ids_in_window"]
                if(len(window_ids) < window_size):
                    return logits
                window_ids = window_ids[-window_size:]
                # print(f"当前上下文窗口 {window_ids}")
                window_ids = torch.as_tensor(
                    window_ids, dtype=torch.long, device=logits.device
                )
                output_ids_matrix[i] = window_ids
        
        batched_greenlist_ids = [None for _ in range(output_ids_matrix.shape[0])]

        for idx in range(output_ids_matrix.shape[0]):
            output_ids = output_ids_matrix[idx]
            self.rng = torch.Generator(device=logits.device)
            self.rng.manual_seed(15485863)
            self.prf = torch.randperm(logits.shape[-1], device=logits.device, generator=self.rng)

            time_result = 1
            for i in range(0, window_size):
                time_result *= output_ids[-1 - i].item()
            num = self.prf[time_result % logits.shape[-1]]
        
            self.rng.manual_seed((15485863 * int(num)) % logits.shape[-1])
            greenlist_size = int(logits.shape[-1] * gamma)
            vocab_permutation = torch.randperm(logits.shape[-1], device=logits.device, generator=self.rng)
            greenlist_ids = vocab_permutation[:greenlist_size]
            batched_greenlist_ids[idx] = greenlist_ids
            # print(f"seed:{15485863 * int(num)}, num:{int(num)}, vocab_size:{logits.shape[-1]}, greenlist_size:{greenlist_size}")

        green_tokens_mask = torch.zeros_like(logits)
        for idx in range(len(batched_greenlist_ids)):
            if mask[idx] == 0:
                continue
            green_tokens_mask[idx][batched_greenlist_ids[idx]] = 1
        final_mask = green_tokens_mask.bool()
        
        logits[final_mask] = logits[final_mask] + delta
        return logits



