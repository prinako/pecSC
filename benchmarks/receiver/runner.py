"""Measure receiver cost with fixed-work autoregressive decoding."""

import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import statistics
from time import perf_counter

import torch

from student import Student
from utils.experiment_manifest import file_identity, git_identity
from .artifacts import MODELS, load_specs

RECEIVER_COMPONENTS = ("channel_decoder", "decoder", "dense")
REPO = Path(__file__).resolve().parents[2]


def load_receiver(spec, arch, device):
    # Student constructs the same three RX modules as DeepSC, at either depth.
    receiver = Student(spec.layers, arch["vocab_size"], arch["vocab_size"],
                       arch["max_length"], arch["max_length"], arch["d_model"],
                       spec.heads, arch["dff"], dropout=0.0)
    checkpoint = torch.load(spec.checkpoint["path"], map_location="cpu", weights_only=True)
    metadata = checkpoint.get("meta", {})
    if metadata.get("epoch") != spec.manifest["selected_checkpoint"]["epoch"]:
        raise ValueError(f"{spec.name} checkpoint epoch disagrees with its manifest")
    if spec.name == "M0":
        expected = {"num_layers": spec.layers, "num_heads": spec.heads,
                    "d_model": arch["d_model"], "dff": arch["dff"],
                    "max_len": arch["max_length"], "vocab_size": arch["vocab_size"]}
        if metadata.get("architecture") != expected:
            raise ValueError("M0 checkpoint architecture disagrees with its manifest")
    if spec.name == "M3" and metadata.get("method") != "liu_rx_v1":
        raise ValueError("M3 checkpoint metadata must identify liu_rx_v1")
    for component in RECEIVER_COMPONENTS:
        getattr(receiver, component).load_state_dict(checkpoint[component], strict=True)
    return receiver.to(device).eval().requires_grad_(False)


def fixed_step_decode(receiver, received, source_mask, prefix, causal_mask, steps):
    """Perform exactly steps greedy updates, without EOS stopping or KV caching."""
    memory = receiver.channel_decoder(received)
    tokens = prefix
    for _ in range(steps):
        length = tokens.shape[1]
        features = receiver.decoder(tokens, memory, causal_mask[:, :length, :length], source_mask)
        next_token = receiver.dense(features[:, -1, :]).argmax(dim=-1, keepdim=True)
        tokens = torch.cat((tokens, next_token), dim=1)
    return tokens


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def measure(receivers, received, prefix, steps, warmup, repeats):
    device = received.device
    source_mask = torch.zeros((received.shape[0], 1, received.shape[1]), device=device)
    length = prefix.shape[1] + steps - 1
    causal = torch.ones((1, length, length), device=device).triu_(diagonal=1)

    def decode(name):
        return fixed_step_decode(receivers[name], received, source_mask, prefix, causal, steps)

    for name in MODELS:
        for _ in range(warmup):
            decode(name)
    synchronize(device)

    samples = []
    for i in range(repeats):
        # Rotate model order to reduce consistent first/last position bias.
        offset = i % len(MODELS)
        for name in MODELS[offset:] + MODELS[:offset]:
            synchronize(device)
            start = perf_counter()
            decode(name)
            synchronize(device)
            elapsed_ms = (perf_counter() - start) * 1000
            samples.append({"model": name, "repeat": i + 1, "batch_latency_ms": elapsed_ms})
    return samples


def summarize(specs, receivers, samples, batch_size, decode_steps):
    counts = {name: sum(p.numel() for p in rx.parameters()) for name, rx in receivers.items()}
    results = []
    for spec in specs:
        values = sorted(s["batch_latency_ms"] for s in samples if s["model"] == spec.name)
        median = statistics.median(values)
        results.append({
            "model": spec.name, "receiver_layers": spec.layers, "receiver_heads": spec.heads,
            "receiver_parameters": counts[spec.name],
            "receiver_parameter_reduction_pct_vs_m0": 100 * (1 - counts[spec.name] / counts["M0"]),
            "receiver_checkpoint_bytes": spec.checkpoint["size_bytes"],
            "selected_epoch": spec.manifest["selected_checkpoint"]["epoch"],
            "selection_metric": spec.manifest["selected_checkpoint"]["selection_metric"],
            "batch_size": batch_size, "decode_steps": decode_steps, "samples": len(values),
            "median_batch_latency_ms": median,
            "p95_batch_latency_ms": values[math.ceil(0.95 * len(values)) - 1],
            "examples_per_second_at_median": batch_size * 1000 / median,
        })
    return results


def write_csv(path, rows):
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    if args.output_dir.exists():
        raise FileExistsError(f"Choose a fresh output directory: {args.output_dir}")
    paths = {name: getattr(args, name.lower() + "_manifest") for name in MODELS}
    specs, arch, ids, vocab_identity = load_specs(paths, args.vocab_file)
    if args.source_length > arch["max_length"] or args.decode_steps + 2 > arch["max_length"]:
        raise ValueError("Source length or decoding budget exceeds the position capacity")
    device = torch.device("cuda:0" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    torch.set_num_threads(args.cpu_threads)
    receivers = {spec.name: load_receiver(spec, arch, device) for spec in specs}
    generator = torch.Generator(device="cpu").manual_seed(args.input_seed)
    # Synthetic receiver input: 16 real channel coordinates per source position.
    received = torch.randn(args.batch_size, args.source_length, 16, generator=generator,
                           dtype=torch.float32).to(device)
    prefix = torch.tensor([ids["<START>"], ids["<EN>"]], device=device)
    prefix = prefix.expand(args.batch_size, -1)
    samples = measure(receivers, received, prefix, args.decode_steps, args.warmup, args.repeats)
    summary = summarize(specs, receivers, samples, args.batch_size, args.decode_steps)
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_identity(REPO),
        "scope": "receiver-only, synthetic input, fixed-step autoregressive greedy workload",
        "workload": {"source_length": args.source_length, "decode_steps": args.decode_steps,
                     "batch_size": args.batch_size, "warmup_per_model": args.warmup,
                     "repeats_per_model": args.repeats, "input_seed": args.input_seed,
                     "dtype": str(received.dtype), "early_stop": False, "kv_cache": False},
        "runtime": {"device": str(device), "pytorch": str(torch.__version__),
                    "cuda_runtime": torch.version.cuda, "python": platform.python_version(),
                    "machine": platform.machine(), "platform": platform.platform(),
                    "cpu_threads": torch.get_num_threads(),
                    "float32_matmul_precision": torch.get_float32_matmul_precision(),
                    "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
        "vocabulary": vocab_identity,
        "inputs": [{"model": spec.name, "manifest": spec.manifest_identity,
                    "checkpoint": spec.checkpoint,
                    "training_seed": spec.manifest.get("seed"),
                    "channel": spec.manifest.get("channel"),
                    "checkpoint_selection_rule": spec.manifest.get("checkpoint_selection_rule"),
                    "selected_checkpoint": spec.manifest["selected_checkpoint"]} for spec in specs],
        "implementation_files": {name: file_identity(REPO / name) for name in (
            "benchmarks/receiver/__main__.py", "benchmarks/receiver/artifacts.py",
            "benchmarks/receiver/runner.py", "student.py", "models/transceiver.py")},
        "summary": summary,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(args.output_dir / "summary.csv", summary)
    write_csv(args.output_dir / "latency_samples.csv", samples)
    (args.output_dir / "benchmark_manifest.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return summary
