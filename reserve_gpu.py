from __future__ import annotations

import argparse
import os
import subprocess
import time
from datetime import datetime
from typing import Dict


MIB = 1024**2
GIB = 1024**3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wait for an authorized GPU and temporarily reserve part of its memory."
    )
    parser.add_argument(
        "--gpus",
        default="all",
        help="Physical GPU indices, for example 0,1,2,3; default: all",
    )
    parser.add_argument(
        "--reserve-gb",
        type=float,
        default=19.9,
        help="GPU memory to reserve in GiB",
    )
    parser.add_argument(
        "--leave-free-gb",
        type=float,
        default=1,
        help="Additional memory that must remain free after reservation",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=30.0,
        help="Seconds between checks",
    )
    parser.add_argument(
        "--max-hold-minutes",
        type=float,
        default=180.0,
        help="Maximum reservation duration; use 0 only when server policy permits",
    )
    return parser.parse_args()


def query_free_memory_mib() -> Dict[int, int]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    free_memory: Dict[int, int] = {}
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        index_text, memory_text = line.split(",", maxsplit=1)
        free_memory[int(index_text.strip())] = int(memory_text.strip())
    return free_memory


def parse_gpu_indices(value: str, available: Dict[int, int]) -> list[int]:
    if value.strip().lower() == "all":
        return sorted(available)
    try:
        indices = sorted({int(item.strip()) for item in value.split(",")})
    except ValueError as exc:
        raise SystemExit(f"Invalid --gpus value: {value!r}") from exc
    missing = [index for index in indices if index not in available]
    if missing:
        raise SystemExit(f"GPU indices are unavailable: {missing}")
    return indices


def reserve_memory(physical_gpu: int, reserve_gb: float, hold_minutes: float) -> None:
    # CUDA_VISIBLE_DEVICES must be set before importing torch.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    import torch

    if not torch.cuda.is_available():
        raise SystemExit(f"Physical GPU {physical_gpu} is not available to PyTorch")

    total_bytes = int(reserve_gb * GIB)
    block_bytes = 256 * MIB
    blocks = []
    allocated_bytes = 0

    try:
        while allocated_bytes < total_bytes:
            current_bytes = min(block_bytes, total_bytes - allocated_bytes)
            blocks.append(torch.empty(current_bytes, dtype=torch.uint8, device="cuda:0"))
            allocated_bytes += current_bytes
            print(
                f"Reserved {allocated_bytes / GIB:.2f}/{reserve_gb:.2f} GiB "
                f"on physical GPU {physical_gpu}",
                flush=True,
            )

        print(
            f"Reservation active. PID={os.getpid()}, physical GPU={physical_gpu}, "
            f"reserved={allocated_bytes / GIB:.2f} GiB. Press Ctrl+C to release.",
            flush=True,
        )
        started_at = time.monotonic()
        while hold_minutes <= 0 or time.monotonic() - started_at < hold_minutes * 60:
            time.sleep(10)
    except KeyboardInterrupt:
        print("Release requested.", flush=True)
    finally:
        blocks.clear()
        torch.cuda.empty_cache()
        print(f"Released GPU {physical_gpu} reservation.", flush=True)


def main() -> None:
    args = parse_args()
    if args.reserve_gb <= 0 or args.leave_free_gb < 0:
        raise SystemExit("Invalid reservation size")
    if args.poll_seconds <= 0 or args.max_hold_minutes < 0:
        raise SystemExit("Invalid polling or hold duration")

    required_mib = int((args.reserve_gb + args.leave_free_gb) * 1024)
    while True:
        try:
            free_memory = query_free_memory_mib()
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            raise SystemExit(f"Unable to query GPUs with nvidia-smi: {exc}") from exc
        gpu_indices = parse_gpu_indices(args.gpus, free_memory)
        status = ", ".join(
            f"GPU {index}: {free_memory[index] / 1024:.1f} GiB free"
            for index in gpu_indices
        )
        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {status}", flush=True)

        candidates = [index for index in gpu_indices if free_memory[index] >= required_mib]
        if candidates:
            selected_gpu = max(candidates, key=free_memory.get)
            reserve_memory(selected_gpu, args.reserve_gb, args.max_hold_minutes)
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
