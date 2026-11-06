"""Resolve the four receiver artifacts from explicit, relocatable manifests."""

from dataclasses import dataclass
import json
from pathlib import Path

from utils.experiment_manifest import file_identity

MODELS = ("M0", "M1", "M2", "M3")
DIMENSIONS = ("d_model", "dff", "max_length", "vocab_size")
DATA_ARTIFACTS = ("dataset_manifest", "split_ids", "tokenizer_model", "train", "val")


@dataclass(frozen=True)
class ReceiverSpec:
    name: str
    layers: int
    heads: int
    manifest: dict
    manifest_identity: dict
    checkpoint: dict


def identity_key(record):
    return record["sha256"], record["size_bytes"]


def verify_file(path, record):
    actual = file_identity(path)
    if identity_key(actual) != identity_key(record):
        raise ValueError(f"Hash or size does not match manifest: {path}")
    return actual


def read_manifest(path):
    return json.loads(path.read_text(encoding="utf-8"))


def selected_checkpoint(path, manifest):
    selected = manifest.get("selected_checkpoint")
    if not selected:
        raise ValueError(f"No selected checkpoint in {path}")
    # The original absolute path may refer to the training machine.
    checkpoint = path.parent / Path(selected["path"]).name
    return verify_file(checkpoint, selected)


def dataset_identity(manifest):
    return {key: identity_key(manifest["dataset"][key]) for key in DATA_ARTIFACTS}


def check_method(name, manifest):
    loss = manifest["loss"]
    method = manifest.get("method", loss.get("method", "receiver_kd"))
    alpha, beta, gamma = (loss[k] for k in ("alpha", "beta", "gamma"))
    valid = {
        "M1": method == "receiver_kd" and (alpha, beta, gamma) == (1, 0, 0),
        "M2": method == "receiver_kd" and alpha > 0 and beta > 0 and gamma > 0,
        "M3": method == "liu_rx_v1" and alpha > 0 and beta > 0 and gamma > 0,
    }
    if not valid[name]:
        raise ValueError(f"{name} manifest does not identify the expected objective")


def load_specs(manifest_paths, vocab_path):
    """Verify loaded checkpoint bytes and common training-artifact identities."""
    teacher_path = manifest_paths["M0"]
    teacher = read_manifest(teacher_path)
    if teacher.get("model_type") != "Teacher":
        raise ValueError("M0 must be a Teacher manifest")
    arch = teacher["architecture"]
    teacher_receiver = selected_checkpoint(teacher_path, teacher)
    teacher_encoder = teacher["selected_checkpoint"]["encoder"]
    vocab_identity = verify_file(vocab_path, teacher["tokenizer_vocab"])
    vocab = read_manifest(vocab_path)
    if vocab["vocab_size"] != arch["vocab_size"]:
        raise ValueError("Teacher and vocabulary sizes differ")
    ids = {key: vocab["token_to_idx"][key] for key in ("<START>", "<EN>")}
    if len(set(ids.values())) != 2 or any(
        type(i) is not int or not 0 <= i < arch["vocab_size"] for i in ids.values()
    ):
        raise ValueError("Invalid START/EN prefix IDs")

    specs = [ReceiverSpec("M0", arch["layers"], arch["heads"], teacher,
                          file_identity(teacher_path), teacher_receiver)]
    student_arch = None
    training_seed = None
    for name in MODELS[1:]:
        path = manifest_paths[name]
        manifest = read_manifest(path)
        check_method(name, manifest)
        shape = manifest["architecture"]
        if any(shape[key] != arch[key] for key in DIMENSIONS):
            raise ValueError(f"{name} and M0 receiver dimensions differ")
        if (shape["teacher_layers"], shape["teacher_heads"]) != (arch["layers"], arch["heads"]):
            raise ValueError(f"{name} records a different Teacher architecture")
        teacher_refs = manifest["teacher_checkpoints"]
        if (identity_key(teacher_refs["encoder"]) != identity_key(teacher_encoder)
                or identity_key(teacher_refs["decoder"]) != identity_key(teacher_receiver)):
            raise ValueError(f"{name} records a different Teacher checkpoint")
        if (identity_key(manifest["tokenizer_vocab"]) != identity_key(vocab_identity)
                or dataset_identity(manifest) != dataset_identity(teacher)):
            raise ValueError(f"{name} records different data or vocabulary artifacts")
        size = (shape["student_layers"], shape["student_heads"])
        if student_arch is None:
            student_arch = size
            training_seed = manifest["seed"]
        if size != student_arch or manifest["seed"] != training_seed:
            raise ValueError("M1–M3 must share the same receiver architecture and training seed")
        specs.append(ReceiverSpec(name, *size, manifest, file_identity(path),
                                  selected_checkpoint(path, manifest)))
    return specs, arch, ids, vocab_identity
