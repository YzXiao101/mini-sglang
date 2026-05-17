from __future__ import annotations

import argparse
import os
import sys
import time
from contextlib import nullcontext
from random import seed
from typing import ContextManager

import torch
from minisgl.benchmark.json import (
    collect_filtered_json_samples,
    render_json_prompt_ids,
    validate_json_output,
)
from minisgl.core import SamplingParams
from minisgl.llm import LLM
from transformers import AutoTokenizer


def print_len_stats(name: str, lengths: list[int]) -> None:
    if not lengths:
        print(f"{name}: no data")
        return
    arr = sorted(lengths)
    n = len(arr)
    print(
        f"{name}: count={n}, min={arr[0]}, p50={arr[int(0.50*n)]}, "
        f"p90={arr[int(0.90*n)]}, p99={arr[int(0.99*n)]}, max={arr[-1]}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.getenv("MODEL", "Qwen/Qwen2-0.5B"))
    parser.add_argument(
        "--mode",
        choices=["constrained", "unconstrained"],
        default="constrained",
    )
    parser.add_argument("--num-seqs", type=int, default=100)
    parser.add_argument("--max-output-len", type=int, default=4096)
    parser.add_argument("--disable-cuda-graph-for-profile", action="store_true")
    parser.add_argument("--cuda-profiler-range", action="store_true")
    parser.add_argument("--nsight-capture-range", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    seed(0)
    MODEL = args.model
    NUM_SEQS = args.num_seqs
    MAX_OUTPUT_LEN = args.max_output_len
    IGNORE_EOS = False

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    samples = collect_filtered_json_samples(NUM_SEQS)
    prompt_token_ids = [render_json_prompt_ids(tokenizer, sample) for sample in samples]

    assert prompt_token_ids, "No valid json-mode-eval samples found"

    sampling_params = []
    for sample in samples:
        json_schema = sample.json_schema if args.mode == "constrained" else None
        sampling_params.append(
            SamplingParams(
                temperature=0.0,
                top_k=1,
                ignore_eos=IGNORE_EOS,
                max_tokens=MAX_OUTPUT_LEN,
                json_schema=json_schema,
            )
        )

    def profile_range(name: str) -> ContextManager[None]:
        if args.nsight_capture_range:
            return torch.cuda.nvtx.range(name)
        return nullcontext()

    llm: LLM | None = None
    try:
        with profile_range("bench_json:init_llm"):
            llm = (
                LLM(MODEL, cuda_graph_max_bs=0)
                if args.disable_cuda_graph_for_profile
                else LLM(MODEL)
            )

        with profile_range("bench_json:warmup"):
            warmup_result = llm.generate(
                [prompt_token_ids[-1]],
                sampling_params[-1],
            )[0]
        templated_input_preview = tokenizer.decode(
            prompt_token_ids[-1],
            skip_special_tokens=False,
        )
        templated_input_preview = templated_input_preview.replace("\n", "\\n")
        warmup_token_ids = warmup_result["token_ids"]
        warmup_text = warmup_result["text"]
        print(
            "Warmup sample: "
            f"mode={args.mode}, "
            f"input={len(prompt_token_ids[-1])}tok, "
            f"templated_input_preview='{templated_input_preview}', "
            f"output={len(warmup_token_ids)}tok, "
            f"preview='{warmup_text}'"
        )

        capture_range = (
            torch.cuda.nvtx.range(args.nsight_capture_range)
            if args.nsight_capture_range
            else nullcontext()
        )
        with capture_range:
            with profile_range("bench_json:generate"):
                if args.cuda_profiler_range or args.nsight_capture_range:
                    torch.cuda.profiler.start()
                try:
                    t = time.time()
                    bench_results = llm.generate(prompt_token_ids, sampling_params)
                    t = time.time() - t
                    if args.cuda_profiler_range or args.nsight_capture_range:
                        torch.cuda.synchronize(llm.device)
                finally:
                    if args.cuda_profiler_range or args.nsight_capture_range:
                        torch.cuda.profiler.stop()
    finally:
        if llm is not None:
            llm.shutdown()

    output_lens = []
    parse_ok = 0
    schema_ok = 0
    schema_checked = 0
    for sample, result in zip(samples, bench_results):
        token_ids = result["token_ids"]
        output_lens.append(len(token_ids))
        parsed, valid = validate_json_output(result["text"], sample.json_schema)
        parse_ok += int(parsed)
        if valid is not None:
            schema_checked += 1
            schema_ok += int(valid)

    total_output_budget = sum(sp.max_tokens for sp in sampling_params)
    total_output_tokens = sum(output_lens)

    print(f"Mode: {args.mode}")
    print_len_stats("Input length", [len(x) for x in prompt_token_ids])
    print_len_stats("Output length", output_lens)
    print(f"Bench requests: {len(prompt_token_ids)}")
    print(f"Output budget: {total_output_budget}tok, " f"Actual output: {total_output_tokens}tok")
    print(f"JSON parse: {parse_ok}/{len(bench_results)}")
    print(f"Schema valid: {schema_ok}/{schema_checked}")
    throughput = total_output_tokens / t if t > 0 else 0.0
    print(f"Total: {total_output_tokens}tok, Time: {t:.2f}s, " f"Throughput: {throughput:.2f}tok/s")
    if args.nsight_capture_range:
        # Avoid late CUDA graph wrapper destructors aborting before Nsight finalizes.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
