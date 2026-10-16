import json
import pickle
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from data_perprocess.preprocess_text_mu_bpe import (
    iter_unique_training_sentences,
    make_split_ids,
    normalized_content_overlaps,
    validate_disjoint_ids,
    validate_split_ratios,
)
from dataset_multilingual import EurParallelDatasetBPE
from metrics.metrics_bpe import summarize_reconstruction_groups
from main_multi_vocab import save_model
from models.transceiver import DeepSC
from models.rx_model import Receiver
from models.tx_model import Transmitter
from train_multi_vocab_one_student import (
    STUDENT_NUM_HEADS,
    STUDENT_NUM_LAYERS,
    build_student_receiver,
    configure_trainability,
    load_teacher_checkpoints,
    setup_seed,
    train,
    validate_loss_configuration,
    validate_epoch,
)
from utils.kd_utils import feature_distillation_loss_cosine_normalized
from utils.experiment_manifest import sha256_file, write_manifest
from utils.publication_data import resolve_training_inputs


def records():
    return [
        {"id": index, "source": f"sentence {index}", "target": f"sentence {index}"}
        for index in range(20)
    ]


@pytest.mark.parametrize("train,val", [(0, .15), (.7, 0), (.9, .2), (-.1, .2)])
def test_invalid_split_ratios(train, val):
    with pytest.raises(ValueError):
        validate_split_ratios(train, val, 20)


def test_duplicate_record_ids_rejected():
    broken = records() + [{"id": 0, "source": "other", "target": "other"}]
    with pytest.raises(ValueError, match="Duplicate IDs"):
        make_split_ids(broken, .7, .15, 48)


def test_cross_split_id_detection():
    with pytest.raises(ValueError, match="across dataset splits"):
        validate_disjoint_ids({1, 2}, {2, 3}, {4})


def test_deterministic_grouped_split_has_no_content_leakage():
    data = records()
    data[19]["source"] = data[0]["source"].upper()
    first = make_split_ids(data, .7, .15, 48)
    second = make_split_ids(data, .7, .15, 48)
    assert first == second
    overlap = normalized_content_overlaps(
        data, dict(zip(("train", "val", "test"), first))
    )
    assert overlap == {"train_val": 0, "train_test": 0, "val_test": 0}


def test_cross_split_normalized_content_leakage_is_detected():
    data = [
        {"id": 1, "source": "Same text", "target": "Same text"},
        {"id": 2, "source": " same   text ", "target": "same text"},
        {"id": 3, "source": "other", "target": "other"},
    ]
    overlap = normalized_content_overlaps(
        data, {"train": {1}, "val": {2}, "test": {3}}
    )
    assert overlap["train_val"] == 1


def test_tokenizer_corpus_uses_train_ids_only():
    data = [
        {"id": 1, "source_language": "en", "target_language": "en",
         "source": "Train source", "target": "Train target"},
        {"id": 2, "source_language": "en", "target_language": "en",
         "source": "Held out", "target": "Never include"},
    ]
    sentences = list(iter_unique_training_sentences({"en_en": data}, {1}))
    assert sentences == ["train source", "train target"]


@pytest.fixture(scope="module")
def synthetic_publication_data(tmp_path_factory):
    root = tmp_path_factory.mktemp("publication-data")
    source_dir = root / "source"
    output_dir = root / "output"
    source_dir.mkdir()
    rows = [
        {
            "id": index,
            "source_language": "en",
            "target_language": "en",
            "source": f"synthetic parliament sentence number {index} .",
            "target": f"synthetic parliament sentence number {index} .",
        }
        for index in range(40)
    ]
    (source_dir / "en_en.json").write_text(json.dumps(rows), encoding="utf-8")
    subprocess.run(
        [
            sys.executable,
            "data_perprocess/preprocess_text_mu_bpe.py",
            "--data-dir", str(source_dir),
            "--output-dir", str(output_dir),
            "--vocab-size", "64",
            "--no-byte-fallback",
            "--min-len", "1",
            "--max-len", "32",
            "--train-ratio", "0.70",
            "--val-ratio", "0.15",
            "--seed", "48",
            "--force-retrain-tokenizer",
        ],
        check=True,
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
    )
    return output_dir


def test_explicit_dataset_directory_and_missing_files(tmp_path, synthetic_publication_data):
    with pytest.raises(FileNotFoundError, match="Missing required"):
        resolve_training_inputs(tmp_path, "vocab_bpe.json")
    root, vocab, paths = resolve_training_inputs(
        synthetic_publication_data, "vocab_bpe.json", expected_max_length=35
    )
    assert root == synthetic_publication_data.resolve()
    assert vocab == (synthetic_publication_data / "vocab_bpe.json").resolve()
    assert set(paths) == {
        "train", "val", "dataset_manifest", "tokenizer_model", "split_ids"
    }
    assert len(EurParallelDatasetBPE("en_en", "train", root)) > 0


def test_training_validation_rejects_stale_hash(tmp_path, synthetic_publication_data):
    copied = tmp_path / "tampered"
    shutil.copytree(synthetic_publication_data, copied)
    with (copied / "train_en_en.pkl").open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="size mismatch|hash mismatch"):
        resolve_training_inputs(copied, "vocab_bpe.json", expected_max_length=35)


def test_training_validation_rejects_vocab_tokenizer_mismatch(
    tmp_path, synthetic_publication_data
):
    copied = tmp_path / "bad-vocab"
    shutil.copytree(synthetic_publication_data, copied)
    vocab_path = copied / "vocab_bpe.json"
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    vocab["token_to_idx"]["<EN>"] = 5
    vocab_path.write_text(json.dumps(vocab), encoding="utf-8")
    manifest_path = copied / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["vocabulary_json"]["sha256"] = sha256_file(vocab_path)
    manifest["vocabulary_json"]["size_bytes"] = vocab_path.stat().st_size
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="mapping mismatch|Special-token"):
        resolve_training_inputs(copied, "vocab_bpe.json", expected_max_length=35)


def test_training_validation_rejects_encoded_train_val_overlap(
    tmp_path, synthetic_publication_data
):
    copied = tmp_path / "overlap"
    shutil.copytree(synthetic_publication_data, copied)
    with (copied / "train_en_en.pkl").open("rb") as handle:
        train_rows = pickle.load(handle)
    val_path = copied / "val_en_en.pkl"
    with val_path.open("rb") as handle:
        val_rows = pickle.load(handle)
    val_rows[0] = deepcopy(train_rows[0])
    with val_path.open("wb") as handle:
        pickle.dump(val_rows, handle, protocol=pickle.HIGHEST_PROTOCOL)
    manifest_path = copied / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["output_files"]["en_en"]["val"]["sha256"] = sha256_file(val_path)
    manifest["output_files"]["en_en"]["val"]["size_bytes"] = val_path.stat().st_size
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="overlaps"):
        resolve_training_inputs(copied, "vocab_bpe.json", expected_max_length=35)


def test_training_validation_rejects_empty_filtered_split(
    tmp_path, synthetic_publication_data
):
    copied = tmp_path / "empty-val"
    shutil.copytree(synthetic_publication_data, copied)
    val_path = copied / "val_en_en.pkl"
    with val_path.open("wb") as handle:
        pickle.dump([], handle, protocol=pickle.HIGHEST_PROTOCOL)
    manifest_path = copied / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["output_files"]["en_en"]["val"]["sha256"] = sha256_file(val_path)
    manifest["output_files"]["en_en"]["val"]["size_bytes"] = val_path.stat().st_size
    manifest["record_counts"]["accepted_after_filtering"]["en_en"]["val"] = 0
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="empty after filtering"):
        resolve_training_inputs(copied, "vocab_bpe.json", expected_max_length=35)


def test_training_validation_does_not_open_test_pickle(
    tmp_path, synthetic_publication_data
):
    copied = tmp_path / "no-test"
    shutil.copytree(synthetic_publication_data, copied)
    (copied / "test_en_en.pkl").unlink()
    resolve_training_inputs(copied, "vocab_bpe.json", expected_max_length=35)


def tiny_teacher():
    return DeepSC(1, 12, 12, 8, 8, 8, 2, 16, 0.0)


TINY_ARCHITECTURE = {
    "num_layers": 1,
    "num_heads": 2,
    "d_model": 8,
    "dff": 16,
    "max_len": 8,
    "vocab_size": 12,
}


def save_teacher(model, directory):
    return save_model(
        model,
        str(directory),
        epoch=0,
        metadata={
            "experiment_id": "synthetic-teacher",
            "epoch": 1,
            "architecture": TINY_ARCHITECTURE,
        },
    )


def test_teacher_checkpoint_missing_and_incompatible(tmp_path):
    model = tiny_teacher()
    with pytest.raises(FileNotFoundError):
        load_teacher_checkpoints(model, tmp_path / "missing", tmp_path / "missing2", "cpu")
    encoder, decoder = save_teacher(model, tmp_path)
    incompatible = DeepSC(1, 13, 13, 8, 8, 8, 2, 16, 0.0)
    with pytest.raises(RuntimeError, match="incompatible"):
        load_teacher_checkpoints(incompatible, encoder, decoder, "cpu")


def test_teacher_checkpoint_round_trip_freezing_and_student_forward(tmp_path):
    original = tiny_teacher()
    expected = deepcopy(original.state_dict())
    encoder, decoder = save_teacher(original, tmp_path)
    loaded = tiny_teacher()
    for parameter in loaded.parameters():
        nn.init.constant_(parameter, 0.0)
    load_teacher_checkpoints(
        loaded, encoder, decoder, "cpu", expected_architecture=TINY_ARCHITECTURE
    )
    for name, tensor in loaded.state_dict().items():
        assert torch.equal(tensor, expected[name])
    assert not any(parameter.requires_grad for parameter in loaded.parameters())

    transmitter = Transmitter(loaded.encoder, loaded.channel_encoder)
    receiver = Receiver(loaded.channel_decoder, loaded.decoder, loaded.dense)
    args = SimpleNamespace(max_len=8, d_model=8, dff=16, dropout=0.0)
    setup_seed(42)
    student = build_student_receiver(12, args, torch.device("cpu"))
    configure_trainability(transmitter, receiver, student)
    source = torch.tensor([[1, 4, 5, 2]])
    target_input = source[:, :-1]
    from utils.train_utils import create_masks
    source_mask, look_ahead_mask = create_masks(source, target_input, 0)
    with torch.no_grad():
        *_, noisy = transmitter(source, source_mask, "AWGN", 0.0)
        logits, _, _ = student(noisy, target_input, look_ahead_mask, source_mask)
    assert logits.shape == (1, 3, 12)
    assert not any(parameter.requires_grad for parameter in transmitter.parameters())
    assert all(parameter.requires_grad for parameter in student.parameters())


def test_teacher_checkpoint_rejects_architecture_metadata_mismatch(tmp_path):
    encoder, decoder = save_teacher(tiny_teacher(), tmp_path)
    expected = {**TINY_ARCHITECTURE, "num_heads": 4}
    with pytest.raises(ValueError, match="metadata mismatch"):
        load_teacher_checkpoints(tiny_teacher(), encoder, decoder, "cpu", expected)


def test_teacher_freezing_and_student_trainability():
    transmitter, teacher, student = nn.Linear(2, 2), nn.Linear(2, 2), nn.Linear(2, 2)
    configure_trainability(transmitter, teacher, student)
    assert not any(p.requires_grad for p in transmitter.parameters())
    assert not any(p.requires_grad for p in teacher.parameters())
    assert all(p.requires_grad for p in student.parameters())


def test_ce_and_kd_students_have_identical_seeded_initialization():
    args = SimpleNamespace(max_len=8, d_model=8, dff=16, dropout=0.0)
    setup_seed(42)
    ce_student = build_student_receiver(12, args, torch.device("cpu"))
    setup_seed(42)
    kd_student = build_student_receiver(12, args, torch.device("cpu"))
    assert STUDENT_NUM_LAYERS == 4
    assert STUDENT_NUM_HEADS == 8
    assert ce_student.state_dict().keys() == kd_student.state_dict().keys()
    assert all(
        torch.equal(ce_student.state_dict()[name], kd_student.state_dict()[name])
        for name in ce_student.state_dict()
    )


def test_ce_only_configuration():
    assert validate_loss_configuration(1.0, 0.0, 0.0, 2.0) == "CE-only Student"
    with pytest.raises(ValueError):
        validate_loss_configuration(0.0, 0.0, 0.0, 2.0)


class FailingTeacher(nn.Module):
    def forward(self, *args, **kwargs):
        raise AssertionError("CE-only validation must not execute the Teacher receiver")


def student_loss_args(**overrides):
    values = {
        "alpha": 1.0,
        "beta": 0.0,
        "gamma": 0.0,
        "temperature": 2.0,
        "snr_mode": "fixed",
        "snr_db": 8.0,
        "snr_db_low": 2.0,
        "snr_db_high": 18.0,
        "grad_clip": 1.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def tiny_receiver_pipeline():
    teacher = tiny_teacher()
    transmitter = Transmitter(teacher.encoder, teacher.channel_encoder)
    receiver = Receiver(teacher.channel_decoder, teacher.decoder, teacher.dense)
    model_args = SimpleNamespace(max_len=8, d_model=8, dff=16, dropout=0.0)
    setup_seed(42)
    student = build_student_receiver(12, model_args, torch.device("cpu"))
    configure_trainability(transmitter, receiver, student)
    batches = DataLoader([
        (torch.tensor([1, 4, 5, 2]), torch.tensor([1, 4, 5, 2])),
        (torch.tensor([1, 4, 6, 2]), torch.tensor([1, 4, 6, 2])),
    ], batch_size=2)
    return transmitter, receiver, student, batches


def test_ce_only_has_no_teacher_receiver_or_kd_contribution():
    transmitter, _, student, batches = tiny_receiver_pipeline()
    stats = validate_epoch(
        epoch=0,
        transmitter=transmitter,
        teacher=FailingTeacher(),
        student=student,
        val_loader=batches,
        pad_idx=0,
        channel="AWGN",
        noise_std=0.0,
        device=torch.device("cpu"),
        criterion=None,
        args=student_loss_args(),
    )
    assert stats["kd"] == 0.0
    assert stats["tf"] == 0.0
    assert stats["loss"] == pytest.approx(stats["ce"])


def test_synthetic_student_training_and_kd_validation_smoke():
    transmitter, receiver, student, batches = tiny_receiver_pipeline()
    optimizer = torch.optim.Adam(
        student.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-8, weight_decay=5e-4
    )
    ce_stats = train(
        transmitter=transmitter,
        teacher=receiver,
        student=student,
        train_loader=batches,
        optimizer=optimizer,
        pad_idx=0,
        channel="AWGN",
        device=torch.device("cpu"),
        criterion=None,
        epoch=0,
        args=student_loss_args(),
    )
    assert ce_stats["kd"] == 0.0
    assert ce_stats["tf"] == 0.0
    kd_args = student_loss_args(alpha=0.6, beta=0.3, gamma=0.1)
    kd_stats = validate_epoch(
        epoch=0,
        transmitter=transmitter,
        teacher=receiver,
        student=student,
        val_loader=batches,
        pad_idx=0,
        channel="AWGN",
        noise_std=0.0,
        device=torch.device("cpu"),
        criterion=None,
        args=kd_args,
    )
    assert kd_stats["kd"] >= 0.0
    assert kd_stats["tf"] >= 0.0
    assert kd_stats["loss"] == pytest.approx(
        0.6 * kd_stats["ce"] + 0.3 * kd_stats["kd"] + 0.1 * kd_stats["tf"],
        rel=1e-6,
    )


def test_normalized_decoder_feature_loss_matches_documented_formula():
    student_features = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [5.0, 5.0]]])
    teacher_features = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [1.0, 1.0]]])
    targets = torch.tensor([[4, 5, 0]])
    loss = feature_distillation_loss_cosine_normalized(
        student_features, teacher_features, targets, pad_idx=0
    )
    assert loss.item() == pytest.approx(0.5)


def test_corpus_bleu_is_grouped_not_sentence_averaged():
    frame = pd.DataFrame([
        {"sample_id": 1, "model": "M1", "channel": "AWGN", "snr_db": 8,
         "trial_seed": 7, "reference": "a b c d", "hypothesis": "a b c d", "checkpoint": "x"},
        {"sample_id": 2, "model": "M1", "channel": "AWGN", "snr_db": 8,
         "trial_seed": 7, "reference": "e f g h", "hypothesis": "e f x y", "checkpoint": "x"},
    ])
    result = summarize_reconstruction_groups(frame)
    assert list(result["sample_count"]) == [2]
    assert 0 < result.loc[0, "corpus_bleu"] < 100


def test_manifest_serialization_and_no_overwrite(tmp_path):
    artifact = tmp_path / "artifact"
    artifact.write_bytes(b"publication")
    output = tmp_path / "manifest.json"
    write_manifest(output, {"hash": sha256_file(artifact), "selected": None})
    assert json.loads(output.read_text())["selected"] is None
    with pytest.raises(FileExistsError):
        write_manifest(output, {})
