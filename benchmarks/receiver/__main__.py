"""Run with python -m benchmarks.receiver from the pecSC root."""

import argparse
from pathlib import Path

from .artifacts import MODELS
from .runner import run


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare M0–M3 receiver cost on matched synthetic inputs")
    for name in MODELS:
        parser.add_argument(f"--{name.lower()}-manifest", type=Path, required=True)
    parser.add_argument("--vocab-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--source-length", type=positive_int, default=32)
    parser.add_argument("--decode-steps", type=positive_int, default=24)
    parser.add_argument("--batch-size", type=positive_int, default=1)
    parser.add_argument("--warmup", type=positive_int, default=5)
    parser.add_argument("--repeats", type=positive_int, default=30)
    parser.add_argument("--cpu-threads", type=positive_int, default=1)
    parser.add_argument("--input-seed", type=int, default=1001)
    args = parser.parse_args(argv)
    for row in run(args):
        print(f"{row['model']}: {row['receiver_parameters']:,} parameters, "
              f"{row['receiver_checkpoint_bytes']:,} bytes, "
              f"{row['median_batch_latency_ms']:.3f} ms median per batch")
    print(f"Reports: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
