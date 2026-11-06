import csv
import json
import pickle

import numpy as np
import pytest
import sentencepiece as spm
import torch

from models.transceiver import DeepSC
from utils.experiment_manifest import file_identity
from . import quality
from .quality_metrics import METRICS, QualityScorer
from .test_benchmark import artifacts  # Shared checkpoint fixture.


class SyntheticEmbeddings:
    """Deterministic test double, not an empirical semantic model."""

    def encode(self, texts, **kwargs):
        vectors = np.array([[1, len(text)] for text in texts], dtype=np.float64)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def test_corpus_metrics_and_token_length_penalty():
    class WordIds:
        def encode(self, texts):
            return [text.split() for text in texts]

    scorer = QualityScorer(WordIds(), SyntheticEmbeddings(), 1)
    ref = ["one two three four"]
    scores = scorer.score_trial(ref, {"same": ref, "longer": [ref[0] + " extra"]})
    assert scores["same"] == pytest.approx({
        "bleu": 100, "chrf": 100, "semantic_similarity": 1, "rouge_l": 1, "token_accuracy": 1,
    })
    assert scores["longer"]["token_accuracy"] == pytest.approx(.8)
    assert scores["longer"]["rouge_l"] < 1


@pytest.fixture
def quality_inputs(tmp_path, artifacts):
    paths, vocab_path = artifacts
    data = tmp_path / "bpe"
    data.mkdir()
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(["aa bb", "ab ba", "bb aa", "aa ab", "ba bb"]),
        model_prefix=str(data / "tokenizer_bpe"), model_type="bpe", vocab_size=16,
        pad_id=0, bos_id=1, eos_id=2, unk_id=3,
        pad_piece="<PAD>", bos_piece="<START>", eos_piece="<END>", unk_piece="<UNK>",
        user_defined_symbols=["<EN>", "<PT>", "<ES>", "<FR>"], minloglevel=2,
    )
    sp = spm.SentencePieceProcessor(model_file=str(data / "tokenizer_bpe.model"))
    vocab_path.write_text(json.dumps({"vocab_size": 16, "token_to_idx": {
        sp.id_to_piece(i): i for i in range(16)
    }}))
    sequences = [[1, 4, *sp.encode(text), 2] for text in ("aa bb", "ab ba")]
    assert max(map(len, sequences)) <= 8
    with (data / "val_en_en.pkl").open("wb") as stream:
        pickle.dump([(seq, seq) for seq in sequences], stream)
    (data / "split_ids.json").write_text(json.dumps({"train": [0], "val": [1, 2], "test": [3]}))
    val = file_identity(data / "val_en_en.pkl")
    (data / "dataset_manifest.json").write_text(json.dumps({
        "record_ids_disjoint": True,
        "duplicate_statistics": {"cross_split_overlaps": {"train_val": 0, "train_test": 0, "val_test": 0}},
        "output_files": {"en_en": {"val": val}},
        "record_counts": {"accepted_after_filtering": {"en_en": {"val": 2}}},
    }))
    dataset = {
        "dataset_manifest": file_identity(data / "dataset_manifest.json"),
        "split_ids": file_identity(data / "split_ids.json"),
        "tokenizer_model": file_identity(data / "tokenizer_bpe.model"),
        "train": {"sha256": "a" * 64, "size_bytes": 1},
        "val": val,
    }

    teacher = DeepSC(2, 16, 16, 8, 8, 8, 2, 16, dropout=0.0)
    teacher_dir = paths["M0"].parent
    meta = torch.load(teacher_dir / "selected.pth", weights_only=True)["meta"]
    encoder_path = teacher_dir / "encoder_01.pth"
    torch.save({"encoder": teacher.encoder.state_dict(),
                "channel_encoder": teacher.channel_encoder.state_dict(), "meta": meta}, encoder_path)
    encoder = file_identity(encoder_path)
    for name, path in paths.items():
        manifest = json.loads(path.read_text())
        manifest["dataset"] = dataset
        manifest["tokenizer_vocab"] = file_identity(vocab_path)
        if name == "M0":
            manifest["selected_checkpoint"]["encoder"] = encoder
        else:
            manifest["teacher_checkpoints"]["encoder"] = encoder
            manifest.update(channel={"normalization": quality.NORMALIZATION},
                            optimizer={"name": "Adam"}, batch_size=32, epoch_budget=10)
            manifest["loss"]["label_smoothing"] = .1
            manifest["selected_checkpoint"]["selection_value"] = 2.0
            (path.parent / "training.csv").write_text("epoch,val_ce\n1,2.0\n")
        path.write_text(json.dumps(manifest))

    argv = [item for name, path in paths.items() for item in (f"--{name.lower()}-manifest", str(path))]
    argv += ["--data-dir", str(data), "--vocab-file", str(vocab_path),
             "--output-dir", str(tmp_path / "quality-report"), "--device", "cpu",
             "--max-decode-steps", "3", "--batch-size", "2", "--snr-db", "8", "12",
             "--trial-seeds", "1001", "1002"]
    return paths, argv


def test_quality_validation_end_to_end_shares_channel_draw(tmp_path, quality_inputs, monkeypatch):
    _, argv = quality_inputs
    received_pointers = []
    decode = quality.greedy_decode

    def capture(received, *args):
        received_pointers.append(received.data_ptr())
        return decode(received, *args)

    monkeypatch.setattr(quality, "greedy_decode", capture)
    args = quality.parse_args(argv)
    report = quality.run(args, semantic_factory=lambda _: SyntheticEmbeddings())
    assert report["protocol"]["split"] == "val"
    assert report["protocol"]["examples"] == 2
    assert not report["protocol_warnings"]
    assert len(received_pointers) == 16
    assert all(len(set(received_pointers[i:i+4])) == 1 for i in range(0, 16, 4))
    with (args.output_dir / "predictions.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 32
    with (args.output_dir / "quality_summary.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 8
    assert all(all(row[f"{metric}_mean"] != "" for metric in METRICS) for row in rows)
    assert not (tmp_path / "bpe/test_en_en.pkl").exists()


def test_held_out_rejects_unmatched_checkpoint_selection_before_data_access(quality_inputs):
    paths, argv = quality_inputs
    manifest = json.loads(paths["M2"].read_text())
    manifest["selected_checkpoint"]["selection_metric"] = "validation_objective_loss"
    paths["M2"].write_text(json.dumps(manifest))
    args = quality.parse_args(argv + ["--split", "test", "--semantic-revision", "a" * 40])
    with pytest.raises(ValueError, match="M2: checkpoint was not selected by validation CE"):
        quality.run(args, semantic_factory=lambda _: SyntheticEmbeddings())
    assert not args.output_dir.exists()


def test_held_out_rejects_mislabeled_ce_checkpoint(quality_inputs):
    paths, argv = quality_inputs
    (paths["M2"].parent / "training.csv").write_text("epoch,val_ce\n1,2.0\n2,1.9\n")
    args = quality.parse_args(argv + ["--split", "test", "--semantic-revision", "a" * 40])
    with pytest.raises(ValueError, match="M2: selected checkpoint disagrees"):
        quality.run(args, semantic_factory=lambda _: SyntheticEmbeddings())
