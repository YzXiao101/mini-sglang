from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from random import seed

import torch
from minisgl.benchmark.json import (
    collect_filtered_json_samples,
    render_json_prompt_ids,
    validate_json_output,
)
from minisgl.core import SamplingParams
from minisgl.env import ENV
from minisgl.llm import LLM
from transformers import AutoTokenizer

_PROFILE_ROW_LIMIT = 200


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
    parser.add_argument(
        "--mode",
        choices=["constrained", "unconstrained"],
        default="constrained",
    )
    parser.add_argument("--num-seqs", type=int, default=100)
    parser.add_argument("--max-output-len", type=int, default=4096)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--disable-cuda-graph-for-profile", action="store_true")
    args = parser.parse_args()
    if args.profile and args.profile_dir is None:
        parser.error("--profile requires --profile-dir")
    return args


def _run_bench(
    llm: LLM,
    prompt_token_ids: list[list[int]],
    sampling_params: list[SamplingParams],
    args: argparse.Namespace,
) -> tuple[list[dict[str, str | list[int]]], float]:
    t = time.time()
    if not args.profile:
        return llm.generate(prompt_token_ids, sampling_params), time.time() - t

    output_dir = args.profile_dir
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_stack=False,
        profile_memory=False,
    ) as prof:
        bench_results = llm.generate(prompt_token_ids, sampling_params)
        torch.cuda.synchronize(llm.device)
    elapsed = time.time() - t

    output_dir.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(output_dir / "trace.json"))
    events = prof.key_averages()
    (output_dir / "cpu_table.txt").write_text(
        events.table(sort_by="self_cpu_time_total", row_limit=_PROFILE_ROW_LIMIT)
    )
    (output_dir / "cuda_table.txt").write_text(
        events.table(sort_by="self_cuda_time_total", row_limit=_PROFILE_ROW_LIMIT)
    )
    (output_dir / "meta.txt").write_text(
        json.dumps(
            {
                "mode": args.mode,
                "model": llm.engine.model.__class__.__name__,
                "model_path": llm.tokenizer.name_or_path,
                "seed": 0,
                "num_seqs": args.num_seqs,
                "bench_requests": len(prompt_token_ids),
                "max_output_len": args.max_output_len,
                "overlap_enabled": not bool(ENV.DISABLE_OVERLAP_SCHEDULING),
                "cuda_graph_max_bs": llm.engine.graph_runner.max_graph_bs,
                "profile_row_limit": _PROFILE_ROW_LIMIT,
                "profile_dir": str(output_dir),
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"Profile artifacts saved to: {output_dir}")
    return bench_results, elapsed


def main() -> None:
    args = parse_args()

    seed(0)
    # NOTE: Using a small, unaligned model makes the diff easier to observe
    MODEL = "Qwen/Qwen2-0.5B"
    IGNORE_EOS = False

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    samples = collect_filtered_json_samples(args.num_seqs)
    prompt_token_ids = [render_json_prompt_ids(tokenizer, sample) for sample in samples]

    assert prompt_token_ids, "No valid json-mode-eval samples found"

    sampling_params = [
        SamplingParams(
            temperature=0.0,
            top_k=1,
            ignore_eos=IGNORE_EOS,
            max_tokens=args.max_output_len,
            json_schema=sample.json_schema if args.mode == "constrained" else None,
        )
        for sample in samples
    ]
    llm_kwargs = {}
    if args.disable_cuda_graph_for_profile:
        llm_kwargs["cuda_graph_max_bs"] = 0
    llm = LLM(MODEL, **llm_kwargs)

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

    bench_results, t = _run_bench(llm, prompt_token_ids, sampling_params, args)

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


if __name__ == "__main__":
    main()
