import csv
from copy import deepcopy
import json

import pytest
import torch

from student import Student
from utils.experiment_manifest import file_identity
from .__main__ import main
from .artifacts import load_specs


@pytest.fixture
def artifacts(tmp_path):
    vocab_path = tmp_path / "vocab_bpe.json"
    vocab_path.write_text(json.dumps({"vocab_size": 16, "token_to_idx": {"<START>": 1, "<EN>": 4}}))
    vocab = file_identity(vocab_path)
    arch = {"layers": 2, "heads": 2, "d_model": 8, "dff": 16,
            "max_length": 8, "vocab_size": 16}
    paths = {}
    teacher = None
    for index in range(4):
        name = f"M{index}"
        directory = tmp_path / name
        directory.mkdir()
        layers = 2 if index == 0 else 1
        model = Student(layers, 16, 16, 8, 8, 8, 2, 16, dropout=0.0)
        meta = {"epoch": 1}
        if index == 0:
            meta["architecture"] = dict(num_layers=2, num_heads=2, d_model=8, dff=16,
                                        max_len=8, vocab_size=16)
        if index == 3:
            meta["method"] = "liu_rx_v1"
        checkpoint = directory / "selected.pth"
        torch.save({key: getattr(model, key).state_dict()
                    for key in ("channel_decoder", "decoder", "dense")} | {"meta": meta}, checkpoint)
        selected = file_identity(checkpoint) | {
            "path": "/original/training/server/selected.pth", "epoch": 1,
            "selection_metric": "validation_cross_entropy",
        }
        manifest = {
            "model_type": "Teacher" if index == 0 else "Student",
            "architecture": deepcopy(arch), "seed": 42, "tokenizer_vocab": vocab,
            "dataset": {key: vocab for key in
                        ("dataset_manifest", "split_ids", "tokenizer_model", "train", "val")},
            "selected_checkpoint": selected,
        }
        if index == 0:
            selected["encoder"] = {"sha256": "0" * 64, "size_bytes": 1}
            teacher = manifest
        else:
            manifest["architecture"].update(teacher_layers=2, teacher_heads=2,
                                             student_layers=1, student_heads=2)
            weights = [(1, 0, 0), (.6, .3, .1), (.7, .2, .1)][index - 1]
            manifest["loss"] = dict(zip(("alpha", "beta", "gamma"), weights))
            manifest["method"] = "liu_rx_v1" if index == 3 else "receiver_kd"
            manifest["teacher_checkpoints"] = {
                "encoder": teacher["selected_checkpoint"]["encoder"],
                "decoder": teacher["selected_checkpoint"],
            }
        paths[name] = directory / "manifest.json"
        paths[name].write_text(json.dumps(manifest))
    return paths, vocab_path


def test_synthetic_end_to_end_report_and_no_overwrite(tmp_path, artifacts):
    paths, vocab = artifacts
    output = tmp_path / "report"
    argv = [item for name, path in paths.items() for item in (f"--{name.lower()}-manifest", str(path))]
    argv += ["--vocab-file", str(vocab), "--output-dir", str(output), "--device", "cpu",
             "--source-length", "4", "--decode-steps", "3", "--warmup", "1", "--repeats", "2"]
    main(argv)
    manifest = json.loads((output / "benchmark_manifest.json").read_text())
    rows = manifest["summary"]
    assert [row["model"] for row in rows] == ["M0", "M1", "M2", "M3"]
    assert rows[0]["receiver_parameters"] > rows[1]["receiver_parameters"]
    assert len({row["receiver_parameters"] for row in rows[1:]}) == 1
    for row in rows:
        assert row["median_batch_latency_ms"] > 0
        assert row["samples"] == 2
        path = paths[row["model"]].parent / "selected.pth"
        assert row["receiver_checkpoint_bytes"] == path.stat().st_size
    with (output / "latency_samples.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 8
    assert manifest["workload"]["early_stop"] is False
    with pytest.raises(FileExistsError):
        main(argv)


def test_rejects_mismatched_teacher(artifacts):
    paths, vocab = artifacts
    manifest = json.loads(paths["M3"].read_text())
    manifest["teacher_checkpoints"]["encoder"]["sha256"] = "f" * 64
    paths["M3"].write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="different Teacher checkpoint"):
        load_specs(paths, vocab)


def test_rejects_modified_checkpoint_bytes(artifacts):
    paths, vocab = artifacts
    with (paths["M2"].parent / "selected.pth").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="Hash or size"):
        load_specs(paths, vocab)
