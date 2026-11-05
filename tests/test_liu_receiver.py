"""Mathematical and active-loop checks for the declared M3 adaptation."""
from copy import deepcopy
import sys

import pytest
import torch
from torch.nn import functional as F

from train_multi_vocab_one_student import (
    configure_method, loss_manifest, parse_args, receiver_loss_terms,
    train, validate_epoch,
)
from utils.liu_kd import METHOD_ID, liu_weights, masked_student_teacher_kl
from test_publication_pipeline import (
    tiny_receiver_pipeline, student_loss_args, synthetic_publication_data,
)


def m3_args(**overrides):
    args = student_loss_args(method="liu-receiver", liu_eta_decoder=0.2,
                             liu_eta_output=0.3, temperature=3.0)
    for key, value in overrides.items():
        setattr(args, key, value)
    configure_method(args)
    return args


def test_reverse_kl_formula_temperature_padding_and_gradients():
    s = torch.tensor([[[2., -1., 0.], [0., 3., -2.], [9., 0., -4.]]],
                     dtype=torch.double, requires_grad=True)
    t = torch.tensor([[[0., 1., -3.], [2., 0., -1.], [-9., 0., 4.]]],
                     dtype=torch.double, requires_grad=True)
    targets = torch.tensor([[4, 2, 0]])
    temperature = 3.
    ps = F.softmax(s[:, :2] / temperature, -1)
    pt = F.softmax(t[:, :2] / temperature, -1)
    expected = temperature**2 * (ps * (ps.log() - pt.log())).sum(-1).mean()
    actual = masked_student_teacher_kl(s, t, targets, 0, temperature)
    torch.testing.assert_close(actual, expected)
    reverse = masked_student_teacher_kl(t, s, targets, 0, temperature)
    assert not torch.isclose(actual, reverse)
    actual.backward()
    assert t.grad is None
    assert torch.isfinite(s.grad).all()
    assert s.grad[:, :2].abs().sum() > 0
    assert torch.count_nonzero(s.grad[:, 2]) == 0


def test_all_pad_and_equal_distributions():
    s = torch.randn(2, 3, 5, requires_grad=True)
    t = torch.randn_like(s, requires_grad=True)
    loss = masked_student_teacher_kl(s, t, torch.zeros(2, 3, dtype=torch.long), 0, 2.)
    assert loss.item() == 0.
    loss.backward()
    assert torch.count_nonzero(s.grad) == 0
    assert t.grad is None
    assert masked_student_teacher_kl(s, s, torch.ones(2, 3), 0, 2.).item() == 0.


@pytest.mark.parametrize("decoder,output,temp", [
    (None, .3, 2), (.2, None, 2), (.2, .3, None),
    (-.1, .3, 2), (.8, .3, 2), (0, 0, 2), (.2, .3, 0),
    (float('nan'), .3, 2), (.2, .3, float('inf')),
])
def test_invalid_configuration(decoder, output, temp):
    with pytest.raises(ValueError):
        liu_weights(decoder, output, temp)


def test_complementary_weights_and_boundaries():
    assert liu_weights(.2, .3, 2) == pytest.approx((.5, .3, .2))
    assert liu_weights(1, 0, 2) == (0, 0, 1)
    assert liu_weights(0, 1, 2) == (0, 1, 0)


@pytest.mark.parametrize("temperature", [0., float('nan'), float('inf')])
def test_invalid_kl_temperature(temperature):
    with pytest.raises(ValueError):
        masked_student_teacher_kl(torch.zeros(1, 2, 3), torch.zeros(1, 2, 3),
                                  torch.ones(1, 2), 0, temperature)


def test_mismatched_features_rejected():
    with pytest.raises(ValueError, match="Matching"):
        masked_student_teacher_kl(torch.zeros(1, 2, 3), torch.zeros(1, 2, 4),
                                  torch.ones(1, 2), 0, 2)


def test_m3_dispatch_and_manifest():
    args = m3_args()
    targets = torch.tensor([[4, 2, 0]])
    logits, teacher_logits = torch.randn(1, 3, 8), torch.randn(1, 3, 8)
    features, teacher_features = torch.randn(1, 3, 5), torch.randn(1, 3, 5)
    ce, kd, feat = receiver_loss_terms(logits, teacher_logits, features,
                                     teacher_features, targets, 0, args)
    torch.testing.assert_close(kd, masked_student_teacher_kl(logits, teacher_logits, targets, 0, 3))
    torch.testing.assert_close(feat, masked_student_teacher_kl(features, teacher_features, targets, 0, 3))
    expected_ce = F.cross_entropy(logits[:, :2].reshape(-1, 8), targets[:, :2].reshape(-1), label_smoothing=.1)
    torch.testing.assert_close(ce, expected_ce)
    meta = loss_manifest(args)
    assert meta['method_id'] == METHOD_ID
    assert meta['kl_direction'] == 'student||teacher'
    assert meta['alpha'] == pytest.approx(.5)


def test_cli_requires_declared_m3_settings_and_preserves_legacy(monkeypatch):
    base = ['train', '--data-dir', '/unused', '--teacher-encoder', '/unused/e',
            '--teacher-decoder', '/unused/d']
    monkeypatch.setattr(sys, 'argv', base)
    args = parse_args()
    assert (args.student_num_layers, args.student_num_heads) == (4, 8)
    assert (args.alpha, args.beta, args.gamma, args.temperature) == (.6, .3, .1, 2.)
    monkeypatch.setattr(sys, 'argv', base + ['--method', 'liu-receiver'])
    with pytest.raises(SystemExit):
        parse_args()
    m3 = base + ['--method', 'liu-receiver', '--liu-eta-decoder', '.2',
                 '--liu-eta-output', '.3', '--temperature', '3']
    monkeypatch.setattr(sys, 'argv', m3)
    assert loss_manifest(parse_args())['method_id'] == METHOD_ID
    monkeypatch.setattr(sys, 'argv', m3 + ['--alpha=.6'])
    with pytest.raises(SystemExit):
        parse_args()
    monkeypatch.setattr(sys, 'argv', base + ['--liu-eta-decoder', '.2'])
    with pytest.raises(SystemExit):
        parse_args()


@pytest.mark.parametrize('channel', ['AWGN', 'Rayleigh'])
def test_active_m3_training_validation_shared_channel_and_freezing(channel):
    transmitter, teacher, student, batches = tiny_receiver_pipeline()
    args = m3_args()
    before_teacher = deepcopy(teacher.state_dict())
    before_tx = deepcopy(transmitter.state_dict())
    before_student = deepcopy(student.state_dict())
    seen = {'tx': [], 'teacher': [], 'student': []}
    handles = [
        transmitter.register_forward_hook(lambda module, inputs, output: seen['tx'].append(output[3])),
        teacher.register_forward_pre_hook(
            lambda module, inputs, kwargs: seen['teacher'].append(kwargs['z_noisy']), with_kwargs=True),
        student.register_forward_pre_hook(lambda module, inputs: seen['student'].append(inputs[0])),
    ]
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-4)
    try:
        training = train(transmitter, teacher, student, batches, optimizer, 0,
                         channel, torch.device('cpu'), None, 0, args)
        validation = validate_epoch(0, transmitter, teacher, student, batches, 0,
                                    channel, .1, torch.device('cpu'), None, args)
    finally:
        for handle in handles:
            handle.remove()
    for stats in (training, validation):
        assert stats['loss'] == pytest.approx(
            args.alpha * stats['ce'] + args.beta * stats['kd'] + args.gamma * stats['tf'], rel=1e-6)
    assert len(seen['tx']) == len(seen['teacher']) == len(seen['student']) == 2
    assert all(tx is teacher_z is student_z for tx, teacher_z, student_z in
               zip(seen['tx'], seen['teacher'], seen['student']))
    for module, before in ((teacher, before_teacher), (transmitter, before_tx)):
        assert not module.training
        assert all(not p.requires_grad and p.grad is None for p in module.parameters())
        assert all(torch.equal(value, module.state_dict()[key]) for key, value in before.items())
    assert any(not torch.equal(value, student.state_dict()[key]) for key, value in before_student.items())
    for module in (student.channel_decoder, student.decoder, student.dense):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())


def test_m3_cli_run_records_selected_checkpoint(tmp_path, synthetic_publication_data):
    """Exercise parsing, model loading, training, selection and provenance together."""
    import json
    from pathlib import Path
    import subprocess
    from main_multi_vocab import save_model
    from models.transceiver import DeepSC
    from utils.experiment_manifest import sha256_file

    vocab = json.loads((synthetic_publication_data / 'vocab_bpe.json').read_text())
    architecture = dict(num_layers=1, num_heads=2, d_model=8, dff=16,
                        max_len=35, vocab_size=vocab['vocab_size'])
    model = DeepSC(1, vocab['vocab_size'], vocab['vocab_size'], 35, 35, 8, 2, 16, 0.)
    (tmp_path / 'teacher').mkdir()
    encoder, decoder = save_model(model, str(tmp_path / 'teacher'), epoch=0,
                                  metadata={'architecture': architecture})
    command = [sys.executable, 'train_multi_vocab_one_student.py',
               '--data-dir', str(synthetic_publication_data),
               '--teacher-encoder', encoder, '--teacher-decoder', decoder,
               '--save-dir', str(tmp_path / 'runs'), '--experiment-id', 'm3-smoke',
               '--method', 'liu-receiver', '--liu-eta-decoder', '.2',
               '--liu-eta-output', '.3', '--temperature', '3', '--max-len', '35',
               '--num-layers', '1', '--num-heads', '2', '--d-model', '8',
               '--dff', '16', '--dropout', '0', '--epochs', '1', '--batch-size', '32',
               '--channel', 'AWGN', '--snr-mode', 'fixed', '--seed', '42']
    subprocess.run(command, cwd=Path(__file__).parents[1], check=True,
                   capture_output=True, text=True)
    run = tmp_path / 'runs' / 'm3-smoke'
    manifest = json.loads((run / 'manifest.json').read_text())
    assert manifest['loss']['method_id'] == METHOD_ID
    assert manifest['loss']['alpha'] == pytest.approx(.5)
    assert manifest['channel']['power_normalization'].startswith('unconditional')
    checkpoint = torch.load(run / 'student_best.pth', weights_only=True)
    assert checkpoint['meta']['loss_configuration'] == manifest['loss']
    assert (run / 'training.csv').is_file()
    # Selected checkpoint provenance must reference the actual output bytes.
    assert sha256_file(run / 'student_best.pth') in json.dumps(manifest)
