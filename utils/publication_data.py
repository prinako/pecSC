"""Validate immutable publication dataset artifacts before training."""

from __future__ import annotations

import hashlib
import json
import math
import pickle
from pathlib import Path
from typing import Any

import sentencepiece as spm


SPECIAL_TOKEN_IDS = {"<PAD>": 0, "<START>": 1, "<END>": 2, "<UNK>": 3, "<EN>": 4}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid {description}: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must contain a JSON object: {path}")
    return payload


def _verify_identity(path: Path, identity: Any, description: str) -> None:
    if not isinstance(identity, dict):
        raise ValueError(f"Dataset manifest lacks {description} identity")
    expected_hash = identity.get("sha256")
    expected_size = identity.get("size_bytes")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError(f"Dataset manifest has an invalid SHA-256 for {description}")
    if expected_size != path.stat().st_size:
        raise ValueError(f"Publication artifact size mismatch for {description}: {path}")
    if _sha256(path) != expected_hash:
        raise ValueError(f"Publication artifact hash mismatch for {description}: {path}")


def _validate_split_ids(path: Path, manifest: dict[str, Any]) -> None:
    payload = _load_json(path, "split-ID file")
    names = {"train", "val", "test"}
    if set(payload) != names or not all(isinstance(payload[name], list) for name in names):
        raise ValueError("split_ids.json must contain train, val and test ID lists")
    split_sets = {name: set(payload[name]) for name in names}
    if any(len(split_sets[name]) != len(payload[name]) for name in names):
        raise ValueError("Duplicate record IDs found within split_ids.json")
    if (
        split_sets["train"] & split_sets["val"]
        or split_sets["train"] & split_sets["test"]
        or split_sets["val"] & split_sets["test"]
    ):
        raise ValueError("Record IDs overlap across publication splits")
    expected = manifest.get("record_counts", {}).get("split_ids", {})
    actual = {name: len(ids) for name, ids in split_sets.items()}
    if actual != expected:
        raise ValueError(f"Split-ID counts do not match dataset manifest: {actual} != {expected}")


def _validate_tokenizer_and_vocab(
    tokenizer_path: Path, vocab_path: Path, manifest: dict[str, Any]
) -> int:
    vocab = _load_json(vocab_path, "vocabulary JSON")
    token_to_idx = vocab.get("token_to_idx")
    idx_to_token = vocab.get("idx_to_token")
    vocab_size = vocab.get("vocab_size")
    if not isinstance(token_to_idx, dict) or not isinstance(idx_to_token, dict):
        raise ValueError("Vocabulary JSON lacks token_to_idx or idx_to_token mappings")
    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    if vocab_size != tokenizer.get_piece_size():
        raise ValueError(
            f"Tokenizer/vocabulary size mismatch: {tokenizer.get_piece_size()} != {vocab_size}"
        )
    if len(token_to_idx) != vocab_size or len(idx_to_token) != vocab_size:
        raise ValueError("Vocabulary mappings do not contain exactly vocab_size entries")
    for index in range(vocab_size):
        piece = tokenizer.id_to_piece(index)
        if token_to_idx.get(piece) != index or idx_to_token.get(str(index)) != piece:
            raise ValueError(f"Tokenizer/vocabulary mapping mismatch at ID {index}")
    recorded_ids = manifest.get("tokenizer", {}).get("special_token_ids")
    for token, expected_id in SPECIAL_TOKEN_IDS.items():
        if (
            tokenizer.piece_to_id(token) != expected_id
            or token_to_idx.get(token) != expected_id
            or not isinstance(recorded_ids, dict)
            or recorded_ids.get(token) != expected_id
        ):
            raise ValueError(f"Special-token ID mismatch for {token}; expected {expected_id}")
    if manifest.get("tokenizer", {}).get("actual_vocab_size") != vocab_size:
        raise ValueError("Manifest vocabulary size does not match tokenizer/vocabulary")
    return vocab_size


def _validate_training_split(
    path: Path,
    split: str,
    manifest: dict[str, Any],
    vocab_size: int,
    max_length: int,
) -> set[tuple[int, ...]]:
    with path.open("rb") as handle:
        rows = pickle.load(handle)
    if not isinstance(rows, list):
        raise ValueError(f"{split} dataset must contain a list")
    if not rows:
        raise ValueError(f"{split} dataset is empty after filtering")
    expected_count = (
        manifest.get("record_counts", {})
        .get("accepted_after_filtering", {})
        .get("en_en", {})
        .get(split)
    )
    if expected_count != len(rows):
        raise ValueError(f"{split} accepted count does not match manifest: {len(rows)} != {expected_count}")
    sequences = set()
    for row_index, row in enumerate(rows):
        if not isinstance(row, (tuple, list)) or len(row) != 2:
            raise ValueError(f"Invalid {split} record at row {row_index}")
        source, target = row
        if source != target:
            raise ValueError(f"Non-identical EN reconstruction pair in {split} row {row_index}")
        if not isinstance(source, (tuple, list)) or not 3 <= len(source) <= max_length:
            raise ValueError(f"Invalid sequence length in {split} row {row_index}")
        if source[:2] != [SPECIAL_TOKEN_IDS["<START>"], SPECIAL_TOKEN_IDS["<EN>"]] or source[-1] != SPECIAL_TOKEN_IDS["<END>"]:
            raise ValueError(f"Invalid EN sequence framing in {split} row {row_index}")
        if any(not isinstance(token, int) or not 0 <= token < vocab_size for token in source):
            raise ValueError(f"Out-of-range token ID in {split} row {row_index}")
        sequences.add(tuple(source))
    return sequences


def resolve_training_inputs(
    data_dir: str,
    vocab_file: str,
    language: str = "en_en",
    expected_max_length: int | None = None,
):
    """Resolve and fully validate publication artifacts used by training.

    Only train and validation pickle payloads are opened. The held-out test
    pickle is deliberately not read during training validation.
    """
    if language != "en_en":
        raise ValueError("Publication training is restricted to en_en reconstruction")
    root = Path(data_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    required = {
        "train": root / f"train_{language}.pkl",
        "val": root / f"val_{language}.pkl",
        "dataset_manifest": root / "dataset_manifest.json",
        "tokenizer_model": root / "tokenizer_bpe.model",
        "split_ids": root / "split_ids.json",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing required publication dataset artifact(s): " + ", ".join(missing)
        )
    vocab = Path(vocab_file).expanduser()
    if not vocab.is_absolute():
        vocab = root / vocab
    vocab = vocab.resolve()
    if not vocab.is_file():
        raise FileNotFoundError(f"Vocabulary file does not exist: {vocab}")

    manifest = _load_json(required["dataset_manifest"], "dataset manifest")
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported or missing dataset manifest schema_version")
    ratios = manifest.get("split_ratios", {})
    expected_ratios = {"train": 0.70, "validation": 0.15, "test": 0.15}
    if any(not math.isclose(ratios.get(key, -1), value, abs_tol=1e-12) for key, value in expected_ratios.items()):
        raise ValueError(f"Publication split ratios must be 70/15/15, got {ratios}")
    if manifest.get("record_ids_disjoint") is not True:
        raise ValueError("Dataset manifest does not attest disjoint record IDs")
    overlaps = manifest.get("duplicate_statistics", {}).get("cross_split_overlaps")
    if overlaps != {"train_val": 0, "train_test": 0, "val_test": 0}:
        raise ValueError(f"Dataset manifest reports normalized-content leakage: {overlaps}")
    tokenizer_meta = manifest.get("tokenizer", {})
    if tokenizer_meta.get("trained_from") != "training IDs only" or tokenizer_meta.get("reused_existing") is not False:
        raise ValueError("Publication tokenizer must be freshly trained from training IDs only")

    output_files = manifest.get("output_files", {}).get("en_en", {})
    _verify_identity(required["train"], output_files.get("train"), "training split")
    _verify_identity(required["val"], output_files.get("val"), "validation split")
    _verify_identity(required["tokenizer_model"], tokenizer_meta, "tokenizer model")
    _verify_identity(vocab, manifest.get("vocabulary_json"), "vocabulary JSON")
    _verify_identity(required["split_ids"], manifest.get("split_id_file"), "split-ID file")
    _validate_split_ids(required["split_ids"], manifest)
    vocab_size = _validate_tokenizer_and_vocab(required["tokenizer_model"], vocab, manifest)

    max_length = manifest.get("maximum_stored_sequence_length")
    if not isinstance(max_length, int) or max_length < 3:
        raise ValueError("Manifest maximum_stored_sequence_length is invalid")
    if expected_max_length is not None and max_length != expected_max_length:
        raise ValueError(f"Dataset/model positional capacity mismatch: {max_length} != {expected_max_length}")
    train_sequences = _validate_training_split(
        required["train"], "train", manifest, vocab_size, max_length
    )
    val_sequences = _validate_training_split(
        required["val"], "val", manifest, vocab_size, max_length
    )
    if train_sequences & val_sequences:
        raise ValueError("Encoded content overlaps between training and validation splits")
    return root, vocab, required
