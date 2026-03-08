import gc
import json
import numpy as np
from tqdm import tqdm

import torch
from vllm import LLM, SamplingParams

from watermark.kgw.kgw_logits_processor_for_vllm import KGWLogitsProcessor
from watermark.auto_watermark import AutoWatermarkForVLLM
from utils.transformers_config import TransformersConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


# Clean gpu memory
assert torch.cuda.is_available()
gc.collect()
torch.cuda.empty_cache()
with torch.no_grad():
    torch.cuda.empty_cache()


# Load data
load_range = 500
with open('dataset/c4/processed_c4.json', 'r') as f:
    lines = [json.loads(line) for _, line in zip(range(load_range), f)]


# only implemented KGW in vllm v1 for now
def main(algorithm_name, model_path):

    ''' watermark generation with vLLM v1 segment '''

    model = LLM(
        model=model_path, 
        tokenizer=model_path,
        tokenizer_mode="auto",
        trust_remote_code=True,
        logits_processors=[KGWLogitsProcessor],
        max_model_len=256,
        gpu_memory_utilization=0.9,
        enforce_eager=False,
        dtype="auto",
        disable_custom_all_reduce=False,
        disable_log_stats=False,
        swap_space=32,
        seed=42
    )
    
    prompts = [line['prompt'] for line in lines]

    # only implemented with_watermark_generation in vllm v1 for now
    # without_watermark_generation can be done by legacy code
    outputs = model.generate(
        prompts=prompts,
        sampling_params=SamplingParams(
            n=1, temperature=0.7, seed=42,
            max_tokens=256, min_tokens=16,
            repetition_penalty=1.1,
            extra_args={"kgw_enable": True}
        ),
        use_tqdm=True,
    )
    watermark_text = [output.outputs[0].text for output in outputs]


    '''watermark detection segment '''

    # NOTE: the model settings here should be consistent with the generation segment
    config = AutoConfig.from_pretrained(model_path)
    transformers_config = TransformersConfig(
        model=AutoModelForCausalLM.from_pretrained(model_path),
        tokenizer=AutoTokenizer.from_pretrained(model_path),
        vocab_size=config.vocab_size,
        device="cuda",
        max_new_tokens=256,
        max_length=256,
        do_sample=True,
        no_repeat_ngram_size=4
    )
    watermark = AutoWatermarkForVLLM(algorithm_name=algorithm_name, algorithm_config=f'config/{algorithm_name}.json', transformers_config=transformers_config)
    detect_results = [
        watermark.detect_watermark(text)
        for text in tqdm(watermark_text, desc="Watermark Detection")
    ]
    watermark_detect_results = np.mean([r['is_watermarked'] for r in detect_results])
    print(f"watermark_detect_results: {watermark_detect_results:.3f}")


if __name__ == "__main__":
    model_path = "Qwen/Qwen2.5-1.5B" # "meta-llama/Meta-Llama-3-8B-Instruct"
    method = "KGW" # only implemented KGW in vllm v1 for now
    main(model_path=model_path, algorithm_name=method)
