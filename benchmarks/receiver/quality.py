"""Paired autoregressive reconstruction evaluation for the M0–M3 receivers."""

import argparse
import csv
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import re
import statistics

import torch
from torch.utils.data import DataLoader, Subset

from dataset_multilingual import EurParallelDatasetBPE, collate_parallelBPE
from models.transceiver import DeepSC
from models.tx_model import Transmitter
from train_multi_vocab_one_student import load_teacher_checkpoints
from utils.bpe_utils import SeqtoTextBPE, greedy_decode, load_bpe
from utils.experiment_manifest import file_identity, git_identity, runtime_identity
from utils.model_utils import snr_to_noise
from .__main__ import positive_int
from .artifacts import MODELS, identity_key, load_specs, read_manifest, verify_file
from .quality_metrics import METRICS, QualityScorer
from .runner import REPO, load_receiver, write_csv

NORMALIZATION = "batch_unit_rms_v1"
PREDICTION_COLUMNS = ("sample_id", "model", "split", "channel", "snr_db", "trial_seed",
                      "checkpoint_sha256", "reference", "hypothesis")


def selection_warning(spec):
    """Confirm that the declared selection actually matches the epoch log."""
    selected = spec.manifest["selected_checkpoint"]
    metric = selected["selection_metric"]
    if metric != "validation_cross_entropy" and not (
        spec.name == "M1" and metric == "validation_objective_loss"
    ):
        return f"{spec.name}: checkpoint was not selected by validation CE"
    path = Path(spec.manifest_identity["path"]).parent / "training.csv"
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            rows = [(int(row["epoch"]), float(row["val_ce"])) for row in csv.DictReader(stream)]
        epochs = [epoch for epoch, _ in rows]
        if (not rows or epochs != list(range(1, len(rows) + 1))
                or not all(math.isfinite(value) for _, value in rows)):
            raise ValueError("incomplete or nonfinite epoch rows")
        epoch, value = min(rows, key=lambda row: row[1])
        if epoch != selected["epoch"] or not math.isclose(
            value, selected["selection_value"], rel_tol=1e-9, abs_tol=1e-10
        ):
            return f"{spec.name}: selected checkpoint disagrees with the training.csv validation CE minimum"
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return f"{spec.name}: could not verify validation CE selection from training.csv ({exc})"
    return None


def protocol_warnings(specs):
    """Flag controls that prevent a matched loss comparison."""
    baseline = specs[1].manifest
    warnings = []
    for spec in specs[1:]:
        manifest = spec.manifest
        if manifest["channel"].get("normalization") != NORMALIZATION:
            warnings.append(f"{spec.name}: training normalization is not recorded as {NORMALIZATION}")
        warning = selection_warning(spec)
        if warning:
            warnings.append(warning)
        for key in ("optimizer", "channel", "batch_size", "epoch_budget"):
            if key not in manifest or manifest[key] != baseline.get(key):
                warnings.append(f"{spec.name}: {key} differs from M1")
        for key in ("dropout", "grad_clip", "init_student_from_teacher"):
            if manifest.get("cli_arguments", {}).get(key) != baseline.get("cli_arguments", {}).get(key):
                warnings.append(f"{spec.name}: {key} differs from M1")
        if ("label_smoothing" not in manifest["loss"] or
                manifest["loss"]["label_smoothing"] != baseline["loss"].get("label_smoothing")):
            warnings.append(f"{spec.name}: label smoothing differs from M1")
    return warnings


def load_dataset(args, teacher, arch):
    """Check the selected split against the hashed publication dataset manifest."""
    root = args.data_dir
    recorded = teacher["dataset"]
    identities = {key: verify_file(root / filename, recorded[key]) for key, filename in (
        ("dataset_manifest", "dataset_manifest.json"), ("split_ids", "split_ids.json"),
        ("tokenizer_model", "tokenizer_bpe.model"))}
    manifest = read_manifest(root / "dataset_manifest.json")
    if not manifest.get("record_ids_disjoint") or manifest["duplicate_statistics"]["cross_split_overlaps"] != {
        "train_val": 0, "train_test": 0, "val_test": 0
    }:
        raise ValueError("Dataset manifest does not establish split isolation")
    split_record = manifest["output_files"]["en_en"][args.split]
    if args.split == "val" and identity_key(split_record) != identity_key(recorded["val"]):
        raise ValueError("Validation identity differs from Teacher manifest")
    identities[args.split] = verify_file(root / f"{args.split}_en_en.pkl", split_record)
    tokens, tokenizer = load_bpe(str(args.vocab_file), str(root / "tokenizer_bpe.model"))
    if tokenizer.get_piece_size() != arch["vocab_size"] or any(
        tokens.get(tokenizer.id_to_piece(i)) != i for i in range(arch["vocab_size"])
    ):
        raise ValueError("SentencePiece and checkpoint vocabulary disagree")
    dataset = EurParallelDatasetBPE("en_en", args.split, str(root))
    expected = manifest["record_counts"]["accepted_after_filtering"]["en_en"][args.split]
    if len(dataset) != expected or expected == 0:
        raise ValueError("Empty dataset or split count differs from the manifest")
    count = min(len(dataset), args.max_examples or len(dataset))
    for source, target in dataset.data[:count]:
        if (source != target or not 3 <= len(source) <= arch["max_length"]
                or list(source[:2]) != [tokens["<START>"], tokens["<EN>"]]
                or source[-1] != tokens["<END>"]
                or any(type(i) is not int or not 0 <= i < arch["vocab_size"] for i in source)):
            raise ValueError("Invalid English reconstruction record")
    loader = DataLoader(Subset(dataset, range(count)), batch_size=args.batch_size,
                        shuffle=False, num_workers=0, collate_fn=collate_parallelBPE,
                        generator=torch.Generator().manual_seed(0))
    return loader, tokens, tokenizer, identities, count, expected


def build_transmitter(spec, arch, device):
    selected = spec.manifest["selected_checkpoint"]
    encoder = selected["encoder"]
    encoder_path = Path(spec.manifest_identity["path"]).parent / Path(encoder["path"]).name
    encoder_identity = verify_file(encoder_path, encoder)
    teacher = DeepSC(spec.layers, arch["vocab_size"], arch["vocab_size"], arch["max_length"],
                     arch["max_length"], arch["d_model"], spec.heads, arch["dff"], dropout=0.0)
    load_teacher_checkpoints(teacher, encoder_path, spec.checkpoint["path"], "cpu",
                             expected_architecture={
                                 "num_layers": spec.layers, "num_heads": spec.heads,
                                 "max_len": arch["max_length"], "d_model": arch["d_model"],
                                 "dff": arch["dff"], "vocab_size": arch["vocab_size"],
                             })
    transmitter = Transmitter(teacher.encoder, teacher.channel_encoder)
    return transmitter.to(device).eval().requires_grad_(False), encoder_identity


@torch.inference_mode()
def decode_trial(transmitter, receivers, loader, decoder, tokens, device, channel, snr, seed, max_length):
    """One channel draw per batch; every receiver sees the identical tensor."""
    torch.manual_seed(seed)
    references = []
    hypotheses = {name: [] for name in MODELS}
    noise = snr_to_noise(snr)
    pad = tokens["<PAD>"]
    for source, target in loader:
        source = source.to(device)
        mask = (source == pad).unsqueeze(1).float()
        received = transmitter(source, mask, channel, noise)[3]
        if not torch.isfinite(received).all():
            raise FloatingPointError(f"Nonfinite received tensor: SNR {snr}, trial {seed}")
        references.extend(decoder.sequence_to_text(row) for row in target.tolist())
        for name, receiver in receivers.items():
            output = greedy_decode(received, receiver, source, mask, max_length,
                                   tokens["<START>"], pad, tokens["<END>"], tokens["<EN>"],
                                   channel, snr, device)
            hypotheses[name].extend(decoder.sequence_to_text(row) for row in output.cpu().tolist())
    return references, hypotheses


def summarize(rows, snrs):
    """Aggregate channel trials within one fixed Student training seed."""
    summary, differences = [], []
    for snr in snrs:
        group = [row for row in rows if row["snr_db"] == snr]
        base = {row["trial_seed"]: row for row in group if row["model"] == "M1"}
        for name in MODELS:
            trials = [row for row in group if row["model"] == name]
            result = {"model": name, "snr_db": snr, "examples_per_trial": trials[0]["examples"],
                      "trials": len(trials)}
            for metric in METRICS:
                values = [row[metric] for row in trials]
                result[f"{metric}_mean"] = statistics.mean(values)
                result[f"{metric}_sd"] = statistics.stdev(values) if len(values) > 1 else None
            summary.append(result)
            if name != "M1":
                for row in trials:
                    differences.append({"model": name, "baseline": "M1", "snr_db": snr,
                                        "trial_seed": row["trial_seed"], **{
                                            f"delta_{metric}": row[metric] - base[row["trial_seed"]][metric]
                                            for metric in METRICS}})
    return summary, differences


def load_semantic_model(args):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(args.semantic_model, revision=args.semantic_revision,
                               device=args.semantic_device, trust_remote_code=False).eval()


def run(args, semantic_factory=load_semantic_model):
    if args.output_dir.exists():
        raise FileExistsError(f"Choose a fresh output directory: {args.output_dir}")
    paths = {name: getattr(args, name.lower() + "_manifest") for name in MODELS}
    specs, arch, _, vocab_identity = load_specs(paths, args.vocab_file)
    warnings = protocol_warnings(specs)
    if args.split == "test" and warnings:
        raise ValueError("Held-out evaluation requires matched controls:\n- " + "\n- ".join(warnings))
    for warning in warnings:
        print(f"Exploratory validation: {warning}", flush=True)
    max_length = (args.max_decode_steps + 2 if args.max_decode_steps is not None else arch["max_length"])
    if not 3 <= max_length <= arch["max_length"]:
        raise ValueError("Decoding budget exceeds checkpoint positional capacity")

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    device = torch.device("cuda:0" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if (device.type == "cuda" or args.semantic_device == "cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    torch.set_num_threads(args.cpu_threads)
    torch.use_deterministic_algorithms(True)
    transmitter, encoder_identity = build_transmitter(specs[0], arch, device)
    receivers = {spec.name: load_receiver(spec, arch, device) for spec in specs}
    loader, tokens, tokenizer, data, count, available = load_dataset(args, specs[0].manifest, arch)
    decoder = SeqtoTextBPE(tokens, tokens["<END>"], tokenizer_model=str(args.data_dir / "tokenizer_bpe.model"))
    scorer = QualityScorer(tokenizer, semantic_factory(args), args.semantic_batch_size)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    checkpoints = {spec.name: spec.checkpoint["sha256"] for spec in specs}
    with (args.output_dir / "predictions.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=PREDICTION_COLUMNS)
        writer.writeheader()
        for snr in args.snr_db:
            for seed in args.trial_seeds:
                print(f"Decoding {args.split}: {snr:g} dB, trial {seed}, {count:,} examples", flush=True)
                refs, hyps = decode_trial(transmitter, receivers, loader, decoder, tokens, device,
                                          args.channel, snr, seed, max_length)
                scores = scorer.score_trial(refs, hyps)
                for name in MODELS:
                    group = {"model": name, "split": args.split, "channel": args.channel,
                             "snr_db": snr, "trial_seed": seed, "checkpoint_sha256": checkpoints[name]}
                    writer.writerows({**group, "sample_id": i, "reference": ref, "hypothesis": hyp}
                                     for i, (ref, hyp) in enumerate(zip(refs, hyps[name])))
                    rows.append({**group, "examples": len(refs), **scores[name]})
    summary, differences = summarize(rows, args.snr_db)
    write_csv(args.output_dir / "trial_metrics.csv", rows)
    write_csv(args.output_dir / "quality_summary.csv", summary)
    write_csv(args.output_dir / "paired_differences.csv", differences)

    packages = {}
    for package in ("sacrebleu", "rouge-score", "sentencepiece", "sentence-transformers"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    report = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_identity(REPO),
        "protocol": {"split": args.split, "partial": count < available, "examples": count,
                     "available_examples": available, "sample_order": "stored split row order; zero-based sample_id",
                     "channel": args.channel, "normalization": NORMALIZATION, "snr_db": args.snr_db,
                     "trial_seeds": args.trial_seeds, "training_seed": specs[1].manifest["seed"],
                     "batch_size": args.batch_size, "max_decode_steps": max_length - 2,
                     "decoder": "BPE greedy; [START, EN] prefix; EOS stopping",
                     "trial_policy": "reset Torch seed per SNR/trial; shared received tensor across models",
                     "aggregation": "corpus scores per trial, then mean and sample SD across channel trials"},
        "protocol_warnings": warnings,
        "runtime": {**runtime_identity(), "receiver_device": str(device),
                    "cpu_threads": torch.get_num_threads(), "deterministic_algorithms": True,
                    "metric_packages": packages},
        "semantic_encoder": {"model": args.semantic_model, "revision": args.semantic_revision,
                             "device": args.semantic_device, "batch_size": args.semantic_batch_size},
        "metric_definitions": scorer.signatures(),
        "data": data, "vocabulary": vocab_identity, "teacher_encoder": encoder_identity,
        "inputs": [{"model": s.name, "manifest": s.manifest_identity, "checkpoint": s.checkpoint,
                    "selected_checkpoint": s.manifest["selected_checkpoint"]} for s in specs],
        "implementation_files": {name: file_identity(REPO / name) for name in (
            "benchmarks/receiver/quality.py", "benchmarks/receiver/quality_metrics.py",
            "benchmarks/receiver/artifacts.py", "benchmarks/receiver/runner.py",
            "utils/bpe_utils.py", "utils/model_utils.py", "models/tx_model.py",
            "models/transceiver.py", "student.py", "dataset_multilingual.py")},
    }
    (args.output_dir / "quality_manifest.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Paired M0–M3 reconstruction metrics over SNR")
    for name in MODELS:
        parser.add_argument(f"--{name.lower()}-manifest", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--vocab-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--channel", choices=["Rayleigh", "AWGN", "Rician"], default="Rayleigh")
    parser.add_argument("--snr-db", type=float, nargs="+", required=True)
    parser.add_argument("--trial-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--batch-size", type=positive_int, default=8)
    parser.add_argument("--max-decode-steps", type=positive_int)
    parser.add_argument("--max-examples", type=positive_int, help="Validation-only smoke subset")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--cpu-threads", type=positive_int, default=1)
    parser.add_argument("--semantic-model", default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    parser.add_argument("--semantic-revision", help="Model commit SHA; required for held-out evaluation")
    parser.add_argument("--semantic-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--semantic-batch-size", type=positive_int, default=32)
    args = parser.parse_args(argv)
    if len(set(args.snr_db)) != len(args.snr_db) or not all(map(math.isfinite, args.snr_db)):
        parser.error("--snr-db requires distinct finite values")
    if len(set(args.trial_seeds)) != len(args.trial_seeds) or not all(0 <= s < 2**32 for s in args.trial_seeds):
        parser.error("--trial-seeds requires distinct integers in [0, 2**32)")
    if args.split == "test":
        if args.max_examples is not None:
            parser.error("--max-examples is available only for validation")
        if not re.fullmatch(r"[0-9a-fA-F]{40}", args.semantic_revision or ""):
            parser.error("--split test requires a pinned 40-character --semantic-revision")
        if Path(args.semantic_model).exists():
            parser.error("--split test requires a model repository ID so the revision pins its weights")
    return args


def main(argv=None):
    args = parse_args(argv)
    run(args)
    print(f"Quality reports: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
