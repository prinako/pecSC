import csv
import random
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from models.rx_model import Receiver
from models.transceiver import DeepSC
from models.tx_model import Transmitter
from train_multi_vocab_one_student import (
    DiagnosticCSVLogger,
    STUDENT_NUM_HEADS,
    STUDENT_NUM_LAYERS,
    build_student_receiver,
    configure_trainability,
    setup_seed,
    train,
)
from utils.model_utils import Channels


def _args():
    return SimpleNamespace(
        alpha=1.0,
        beta=0.0,
        gamma=0.0,
        temperature=2.0,
        snr_mode="fixed",
        snr_db=8.0,
        snr_db_low=2.0,
        snr_db_high=18.0,
        grad_clip=1.0,
    )


def _pipeline(seed=42):
    setup_seed(seed)
    teacher_model = DeepSC(1, 12, 12, 8, 8, 8, 2, 16, 0.0)
    transmitter = Transmitter(teacher_model.encoder, teacher_model.channel_encoder)
    teacher = Receiver(
        teacher_model.channel_decoder, teacher_model.decoder, teacher_model.dense
    )
    model_args = SimpleNamespace(max_len=8, d_model=8, dff=16, dropout=0.0)
    student = build_student_receiver(12, model_args, torch.device("cpu"))
    configure_trainability(transmitter, teacher, student)
    batches = DataLoader(
        [
            (torch.tensor([1, 4, 5, 2]), torch.tensor([1, 4, 5, 2])),
            (torch.tensor([1, 4, 6, 2]), torch.tensor([1, 4, 6, 2])),
        ],
        batch_size=2,
        shuffle=False,
    )
    optimizer = torch.optim.Adam(
        student.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-8,
        weight_decay=5e-4,
    )
    return transmitter, teacher, student, batches, optimizer


def _run(channel, diagnostic_path=None):
    transmitter, teacher, student, batches, optimizer = _pipeline()
    # Separate model initialization from the controlled channel/training stream.
    setup_seed(20261022)
    logger = DiagnosticCSVLogger(diagnostic_path) if diagnostic_path else None
    stats = train(
        transmitter=transmitter,
        teacher=teacher,
        student=student,
        train_loader=batches,
        optimizer=optimizer,
        pad_idx=0,
        channel=channel,
        device=torch.device("cpu"),
        criterion=None,
        epoch=0,
        args=_args(),
        diagnostic_logger=logger,
    )
    rng = {
        "torch": torch.get_rng_state().clone(),
        "numpy": deepcopy(np.random.get_state()),
        "python": random.getstate(),
    }
    return stats, deepcopy(student.state_dict()), deepcopy(optimizer.state_dict()), rng


def _assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            _assert_nested_equal(left_item, right_item)
    else:
        assert left == right


def test_rayleigh_metadata_reuses_original_draw_without_rng_change():
    tx = torch.arange(32, dtype=torch.float32).reshape(1, 2, 16) / 32
    setup_seed(991)
    plain = Channels().Rayleigh(tx, 0.2)
    plain_rng = torch.get_rng_state().clone()

    setup_seed(991)
    diagnosed, metadata = Channels().Rayleigh(tx, 0.2, return_metadata=True)
    diagnosed_rng = torch.get_rng_state().clone()

    assert torch.equal(plain, diagnosed)
    assert torch.equal(plain_rng, diagnosed_rng)
    expected_magnitude = (
        metadata["rayleigh_real"] ** 2 + metadata["rayleigh_imag"] ** 2
    ) ** 0.5
    assert metadata["rayleigh_magnitude"] == pytest.approx(expected_magnitude)
    assert metadata["theoretical_noise_amplification"] == pytest.approx(
        1 / expected_magnitude
    )
    assert metadata["theoretical_noise_power_amplification"] == pytest.approx(
        1 / expected_magnitude ** 2
    )


@pytest.mark.parametrize("channel", ["AWGN", "Rayleigh"])
def test_diagnostic_mode_preserves_training_and_all_rng_states(tmp_path, channel):
    baseline = _run(channel)
    diagnosed = _run(channel, tmp_path / f"{channel.lower()}-diagnostics.csv")

    assert baseline[0] == diagnosed[0]
    _assert_nested_equal(baseline[1], diagnosed[1])
    _assert_nested_equal(baseline[2], diagnosed[2])
    assert torch.equal(baseline[3]["torch"], diagnosed[3]["torch"])
    assert baseline[3]["numpy"][0] == diagnosed[3]["numpy"][0]
    assert np.array_equal(baseline[3]["numpy"][1], diagnosed[3]["numpy"][1])
    assert baseline[3]["numpy"][2:] == diagnosed[3]["numpy"][2:]
    assert baseline[3]["python"] == diagnosed[3]["python"]


def test_controlled_m1_awgn_rayleigh_short_comparison(tmp_path):
    assert STUDENT_NUM_LAYERS == 4
    assert STUDENT_NUM_HEADS == 8

    results = {}
    for channel in ("AWGN", "Rayleigh"):
        path = tmp_path / f"{channel.lower()}.csv"
        stats, _, _, _ = _run(channel, path)
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 1
        row = rows[0]
        assert row["channel"] == channel
        assert row["has_nan"] == "False"
        assert row["has_inf"] == "False"
        assert float(row["ce"]) == pytest.approx(stats["ce"])
        assert float(row["gradient_norm_before_clipping"]) >= 0
        assert float(row["parameter_norm"]) > 0
        assert float(row["max_received_signal_magnitude"]) >= 0
        results[channel] = row

    assert results["AWGN"]["rayleigh_magnitude"] == ""
    assert float(results["AWGN"]["theoretical_noise_amplification"]) == 1.0
    assert float(results["Rayleigh"]["rayleigh_magnitude"]) > 0
    assert float(results["Rayleigh"]["theoretical_noise_amplification"]) == pytest.approx(
        1 / float(results["Rayleigh"]["rayleigh_magnitude"])
    )


def test_diagnostic_logger_refuses_overwrite(tmp_path):
    path = tmp_path / "diagnostics.csv"
    DiagnosticCSVLogger(path)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        DiagnosticCSVLogger(path)
