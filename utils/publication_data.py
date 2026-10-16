"""Dataset path validation shared by publication training entry points."""

from pathlib import Path


def resolve_training_inputs(data_dir: str, vocab_file: str, language: str = "en_en"):
    root = Path(data_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    required = {
        split: root / f"{split}_{language}.pkl" for split in ("train", "val")
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing required training dataset file(s): " + ", ".join(missing)
        )
    vocab = Path(vocab_file).expanduser()
    if not vocab.is_absolute():
        candidate = root / vocab
        vocab = candidate if candidate.is_file() else vocab.resolve()
    else:
        vocab = vocab.resolve()
    if not vocab.is_file():
        raise FileNotFoundError(f"Vocabulary file does not exist: {vocab}")
    return root, vocab, required
