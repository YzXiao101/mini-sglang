from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import time
from pathlib import Path

from minisgl.benchmark.client import (
    benchmark_trace,
    get_model_name,
    process_benchmark_results,
    read_qwen_trace,
    scale_traces,
)
from minisgl.utils import init_logger
from openai import AsyncOpenAI as OpenAI
from transformers import AutoTokenizer

logger = init_logger(__name__)

URL = "https://media.githubusercontent.com/media/alibaba-edu/qwen-bailian-usagetraces-anon/refs/heads/main/qwen_traceA_blksz_16.jsonl"
ESTIMATE_METRICS_PATH_ENV = "MINISGL_ESTIMATE_METRICS_PATH"


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
    parser.add_argument(
        "--prompt-mode",
        choices=["dummy", "random", "real"],
        default="dummy",
        help="Prompt source mode.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=4096,
        help="Only takes effect when --prompt-mode=real.",
    )
    return parser.parse_args()


def _resolve_phase_path() -> str | None:
    if not (path := os.getenv(ESTIMATE_METRICS_PATH_ENV)):
        return None
    file_path = Path(path).expanduser()
    return str(file_path.with_name(f"{file_path.stem}.phase.jsonl"))


def _append_phase_manifest(path: str | None, event: str, **fields) -> None:
    if not path:
        return
    file_path = Path(path).expanduser()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    record = {"ts_ns": time.time_ns(), "event": event, **fields}
    with file_path.open("a") as f:
        f.write(json.dumps(record) + "\n")


async def main():
    args = parse_args()
    phase_path = _resolve_phase_path()
    random.seed(42)  # reproducibility
    PORT = 1919
    N = 1000
    SCALES = [0.4, 0.5, 0.6, 0.7, 0.8, 1.6]  # from fast to slow
    async with OpenAI(base_url=f"http://127.0.0.1:{PORT}/v1", api_key="") as client:
        MODEL = await get_model_name(client)
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        TRACES = read_qwen_trace(
            download_qwen_trace(URL),
            tokenizer,
            n=N,
            prompt_mode=args.prompt_mode,
            max_new_tokens=args.max_new_tokens,
        )

        logger.info(f"Start benchmarking with {len(TRACES)} requests using model {MODEL}...")
        for scale in SCALES:
            start_ts_ns = time.time_ns()
            _append_phase_manifest(
                phase_path,
                "bench_qwen_scale_start",
                scale=scale,
                prompt_mode=args.prompt_mode,
                max_new_tokens=args.max_new_tokens,
                model=MODEL,
                start_ts_ns=start_ts_ns,
            )
            traces = scale_traces(TRACES, scale)
            results = await benchmark_trace(
                client,
                traces,
                MODEL,
                ignore_eos=args.prompt_mode != "real",
            )
            process_benchmark_results(results)
            _append_phase_manifest(
                phase_path,
                "bench_qwen_scale_end",
                scale=scale,
                prompt_mode=args.prompt_mode,
                max_new_tokens=args.max_new_tokens,
                model=MODEL,
                start_ts_ns=start_ts_ns,
                end_ts_ns=time.time_ns(),
            )
        logger.info("Benchmarking completed.")


if __name__ == "__main__":
    asyncio.run(main())
