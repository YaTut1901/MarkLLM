import gc
import json
import argparse
import os, sys
import numpy as np
from tqdm import tqdm

#add MarkLLM root dir to sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import torch
from vllm import LLM, SamplingParams

from watermark.kgw.kgw_logits_processor_for_vllm import KGWLogitsProcessor
from watermark.auto_watermark import AutoWatermarkForVLLM
from utils.transformers_config import TransformersConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from visualize.color_scheme import ColorSchemeForDiscreteVisualization
from visualize.font_settings import FontSettings
from visualize.legend_settings import DiscreteLegendSettings
from visualize.page_layout_settings import PageLayoutSettings
from visualize.visualizer import DiscreteVisualizer


"""
Tested under torch 2.10 + cu128, vllm 0.17
"""


# Clean gpu memory
assert torch.cuda.is_available()
gc.collect()
torch.cuda.empty_cache()
with torch.no_grad():
    torch.cuda.empty_cache()


# Load data, you can customize load range here
load_range = 100
with open('dataset/c4/processed_c4.json', 'r') as f:
    lines = [json.loads(line) for _, line in zip(range(load_range), f)]

# calculate average nll for generated outputs, used for calculating perplexity 
# and analyzing the impact of watermark on generation quality. 
def calc_avg_nll(outputs):
    nll_values = []
    for output in outputs:
        completion = output.outputs[0]
        token_count = len(completion.token_ids)
        if token_count == 0 or completion.cumulative_logprob is None:
            continue
        nll_values.append(-completion.cumulative_logprob / token_count)

    if not nll_values:
        raise ValueError(
            "No valid cumulative_logprob found in vLLM outputs. "
            "Please set SamplingParams(logprobs=1) and ensure your vLLM version supports cumulative_logprob."
        )
    return float(np.mean(nll_values))


# only implemented KGW in vllm v1 for now
# KGW fixed settings: gamma=0.5, delta=2.0, window_size=1, hash_key=15485863, f_scheme="time", window_scheme="left"
# can be customized by modifying KGWLogitsProcessor
def main(algorithm_name, model_path):

    ''' Watermark generation with vLLM v1 segment '''

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
            logprobs=1,
            extra_args={"kgw_enable": True}
        ),
        use_tqdm=True, # To disable tqdm progress bar, set use_tqdm=False 
    )
    watermark_text = [output.outputs[0].text for output in outputs]
    watermark_avg_nll = calc_avg_nll(outputs)
    watermark_ppl = np.exp(watermark_avg_nll)
    # print(f"watermark_avg_nll: {watermark_avg_nll:.3f}")
    print(f"watermark_ppl: {watermark_ppl:.3f}")


    ''' Unwatermark generation with vLLM v1 segment '''

    outputs = model.generate(
        prompts=prompts,
        sampling_params=SamplingParams(
            n=1, temperature=0.7, seed=42,
            max_tokens=256, min_tokens=16,
            repetition_penalty=1.1,
            logprobs=1,
            extra_args={"kgw_enable": False}
        ),
        use_tqdm=True, # To disable tqdm progress bar, set use_tqdm=False 
    )
    unwatermark_text = [output.outputs[0].text for output in outputs]
    unwatermark_avg_nll = calc_avg_nll(outputs)
    unwatermark_ppl = np.exp(unwatermark_avg_nll)
    # print(f"unwatermark_avg_nll: {unwatermark_avg_nll:.3f}")
    print(f"unwatermark_ppl: {unwatermark_ppl:.3f}")



    '''Watermark detection segment '''

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

    detect_results = [
        watermark.detect_watermark(text)
        for text in tqdm(unwatermark_text, desc="Watermark Detection")
    ]
    unwatermark_detect_results = np.mean([r['is_watermarked'] for r in detect_results])
    print(f"unwatermark_detect_results: {unwatermark_detect_results:.3f}")


    '''Visualize segment '''

    # Initialize visualizer
    color_scheme = ColorSchemeForDiscreteVisualization()
    font_settings = FontSettings()
    legend_settings = DiscreteLegendSettings()
    page_layout_settings = PageLayoutSettings()
    visualizer = DiscreteVisualizer(
        color_scheme=color_scheme,
        font_settings=font_settings,
        legend_settings=legend_settings,
        page_layout_settings=page_layout_settings
    )

    nowatermarked_img = visualizer.visualize(
        data=watermark.get_data_for_visualization(text=unwatermark_text[0]),
        show_text=True, visualize_weight=True, display_legend=True
    )
    nowatermarked_img.save(os.path.join(PROJECT_ROOT, f"{algorithm_name}-nowatermark-vllm.png"))
    watermarked_img = visualizer.visualize(
        data=watermark.get_data_for_visualization(text=watermark_text[0]),
        show_text=True, visualize_weight=True, display_legend=True
    )
    watermarked_img.save(os.path.join(PROJECT_ROOT, f"{algorithm_name}-watermark-vllm.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo for MarkvLLM with vLLM")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-1.5B", help="Path to the language model")
    args = parser.parse_args()
    method = "KGW" # only implemented KGW in vllm v1 for now
    main(model_path=args.model_path, algorithm_name=method)
