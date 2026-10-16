import json
import pickle
from pathlib import Path

import pandas as pd
import pytest
import torch
from torch import nn

from data_perprocess.preprocess_text_mu_bpe import (
    iter_unique_training_sentences,
    make_split_ids,
    normalized_content_overlaps,
    validate_disjoint_ids,
    validate_split_ratios,
)
from dataset_multilingual import EurParallelDatasetBPE
from metrics.metrics_bpe import summarize_reconstruction_groups
from models.transceiver import DeepSC
from train_multi_vocab_one_student import (
    configure_trainability,
    load_teacher_checkpoints,
    validate_loss_configuration,
)
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


def test_explicit_dataset_directory_and_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing required"):
        resolve_training_inputs(tmp_path, "vocab_bpe.json")
    for split in ("train", "val"):
        with (tmp_path / f"{split}_en_en.pkl").open("wb") as handle:
            pickle.dump([([1, 2], [1, 2])], handle)
    (tmp_path / "vocab_bpe.json").write_text("{}", encoding="utf-8")
    root, vocab, paths = resolve_training_inputs(tmp_path, "vocab_bpe.json")
    assert root == tmp_path.resolve()
    assert vocab == (tmp_path / "vocab_bpe.json").resolve()
    assert set(paths) == {"train", "val"}
    assert len(EurParallelDatasetBPE("en_en", "train", tmp_path)) == 1


def tiny_teacher():
    return DeepSC(1, 12, 12, 8, 8, 8, 2, 16, 0.0)


def save_teacher(model, encoder, decoder):
    torch.save({"encoder": model.encoder.state_dict(),
                "channel_encoder": model.channel_encoder.state_dict()}, encoder)
    torch.save({"channel_decoder": model.channel_decoder.state_dict(),
                "decoder": model.decoder.state_dict(),
                "dense": model.dense.state_dict()}, decoder)


def test_teacher_checkpoint_missing_and_incompatible(tmp_path):
    model = tiny_teacher()
    with pytest.raises(FileNotFoundError):
        load_teacher_checkpoints(model, tmp_path / "missing", tmp_path / "missing2", "cpu")
    encoder, decoder = tmp_path / "enc.pth", tmp_path / "dec.pth"
    save_teacher(model, encoder, decoder)
    incompatible = DeepSC(1, 13, 13, 8, 8, 8, 2, 16, 0.0)
    with pytest.raises(RuntimeError, match="incompatible"):
        load_teacher_checkpoints(incompatible, encoder, decoder, "cpu")


def test_teacher_freezing_and_student_trainability():
    transmitter, teacher, student = nn.Linear(2, 2), nn.Linear(2, 2), nn.Linear(2, 2)
    configure_trainability(transmitter, teacher, student)
    assert not any(p.requires_grad for p in transmitter.parameters())
    assert not any(p.requires_grad for p in teacher.parameters())
    assert all(p.requires_grad for p in student.parameters())


def test_ce_only_configuration():
    assert validate_loss_configuration(1.0, 0.0, 0.0, 2.0) == "CE-only Student"
    with pytest.raises(ValueError):
        validate_loss_configuration(0.0, 0.0, 0.0, 2.0)


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
