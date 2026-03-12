from __future__ import annotations

import argparse
import asyncio
import os
import random
from pathlib import Path

from minisgl.benchmark.client import (
    BenchmarkTrace,
    benchmark_trace,
    get_model_name,
    process_benchmark_results,
    read_qwen_trace,
    scale_traces,
)
from minisgl.benchmark.wildchat import collect_filtered_wildchat_prompts
from minisgl.utils import init_logger
from openai import AsyncOpenAI as OpenAI
from transformers import AutoTokenizer

logger = init_logger(__name__)

URL = "https://media.githubusercontent.com/media/alibaba-edu/qwen-bailian-usagetraces-anon/refs/heads/main/qwen_traceA_blksz_16.jsonl"


def download_qwen_trace(url: str) -> str:
    dir = Path(os.path.dirname(__file__))
    # download the file if not exists
    file_path = dir / "qwen_traceA_blksz_16.jsonl"
    if not file_path.exists():
        import urllib.request

        logger.info(f"Downloading trace from {url} to {file_path}...")
        urllib.request.urlretrieve(url, file_path)
        logger.info("Download completed.")
    return str(file_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark MiniSGL with Qwen trace replay.")
    parser.add_argument("--real-prompts", action="store_true")
    parser.add_argument("--disable-ignore-eos", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    return parser.parse_args()


async def main():
    args = parse_args()
    random.seed(42)  # reproducibility
    PORT = 1919
    N = 1000
    SCALES = [0.4, 0.5, 0.6, 0.7, 0.8, 1.6]  # from fast to slow
    async with OpenAI(base_url=f"http://127.0.0.1:{PORT}/v1", api_key="") as client:
        MODEL = await get_model_name(client)
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        TRACES = read_qwen_trace(download_qwen_trace(URL), tokenizer, n=N, dummy=True)
        if args.real_prompts:
            prompts = collect_filtered_wildchat_prompts(N)
            if len(prompts) == 0:
                raise ValueError("No WildChat prompts available.")

            TRACES = [
                BenchmarkTrace(
                    timestamp=trace.timestamp,
                    message=prompts[i % len(prompts)],
                    output_length=args.max_new_tokens,
                )
                for i, trace in enumerate(TRACES)
            ]

        logger.info(f"Start benchmarking with {len(TRACES)} requests using model {MODEL}...")
        for scale in SCALES:
            traces = scale_traces(TRACES, scale)
            results = await benchmark_trace(
                client,
                traces,
                MODEL,
                ignore_eos=not args.disable_ignore_eos,
            )
            process_benchmark_results(results)
        logger.info("Benchmarking completed.")


if __name__ == "__main__":
    asyncio.run(main())
