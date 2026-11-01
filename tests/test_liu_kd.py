from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from train_multi_vocab_one_student import (
    configure_objective, distillation_terms, objective_manifest,
)
from utils.liu_kd import METHOD_ID, student_teacher_kl, validate_liu_weights


def liu_args(**updates):
    values = dict(method=METHOD_ID, alpha=None, beta=None, gamma=None,
                  liu_decoder_weight=0.1, liu_logit_weight=0.2, temperature=2.0)
    values.update(updates)
    return SimpleNamespace(**values)


def test_kl_direction_matches_independent_probability_formula():
    # Asymmetric distributions distinguish Student||Teacher from its reverse.
    student = torch.tensor([[[2.0, -1.0, 0.3]]], requires_grad=True)
    teacher = torch.tensor([[[-0.5, 1.5, 0.2]]], requires_grad=True)
    target = torch.tensor([[4]])
    result = student_teacher_kl(student, teacher, target, 0, 2.0)
    p = torch.softmax(student.detach().double() / 2.0, -1)
    q = torch.softmax(teacher.detach().double() / 2.0, -1)
    expected = 4 * (p * torch.log(p / q)).sum()
    reverse = 4 * (q * torch.log(q / p)).sum()
    assert result.item() == pytest.approx(expected.item(), rel=1e-6)
    assert abs(result.item() - reverse.item()) > 0.01
    result.backward()
    assert student.grad.abs().sum() > 0
    assert teacher.grad is None


def test_padding_and_token_normalization_are_invariant():
    s = torch.tensor([[[1., 0.], [0., 2.]]], requires_grad=True)
    t = torch.zeros_like(s, requires_grad=True)
    targets = torch.tensor([[4, 2]])
    base = student_teacher_kl(s, t, targets, 0, 1.0)
    padded_s = torch.cat([s, torch.full((1, 1, 2), float('nan'))], dim=1)
    padded_t = torch.cat([t, torch.full((1, 1, 2), float('nan'))], dim=1)
    padded = student_teacher_kl(padded_s, padded_t, torch.tensor([[4, 2, 0]]), 0, 1.0)
    repeated = student_teacher_kl(s.repeat(3, 1, 1), t.repeat(3, 1, 1), targets.repeat(3, 1), 0, 1.0)
    assert padded.item() == pytest.approx(base.item())
    assert repeated.item() == pytest.approx(base.item())
    padded.backward()
    assert torch.isfinite(s.grad).all()
    empty = student_teacher_kl(s, t, torch.zeros_like(targets), 0, 1.0)
    assert empty.item() == 0 and empty.requires_grad


@pytest.mark.parametrize('temperature', [0, -1, float('nan'), float('inf')])
def test_invalid_temperature_rejected(temperature):
    with pytest.raises(ValueError):
        student_teacher_kl(torch.zeros(1, 2, 3), torch.zeros(1, 2, 3),
                           torch.ones(1, 2), 0, temperature)


def test_shape_mismatches_rejected():
    with pytest.raises(ValueError):
        student_teacher_kl(torch.zeros(1, 2, 3), torch.zeros(1, 2, 4), torch.ones(1, 2), 0, 2)
    with pytest.raises(ValueError):
        student_teacher_kl(torch.zeros(1, 2, 3), torch.zeros(1, 2, 3), torch.ones(1, 3), 0, 2)


@pytest.mark.parametrize('weights', [(None, .1), (.1, None), (0, .1), (-.1, .1),
                                     (.5, .5), (.9, .2), (float('nan'), .1)])
def test_invalid_weights_rejected(weights):
    with pytest.raises(ValueError):
        validate_liu_weights(*weights, temperature=2)


def test_method_configuration_and_provenance():
    args = liu_args()
    assert configure_objective(args) == 'Liu et al. receiver-side adaptation'
    assert (args.alpha, args.beta, args.gamma) == pytest.approx((.7, .2, .1))
    record = objective_manifest(args)
    assert record['method'] == METHOD_ID
    assert record['kl_direction'] == 'student||teacher'
    assert record['liu_decoder_weight'] == .1
    with pytest.raises(ValueError):
        configure_objective(liu_args(alpha=.6))
    with pytest.raises(ValueError):
        configure_objective(liu_args(method='receiver_kd'))
    legacy = SimpleNamespace(alpha=None, beta=None, gamma=None, temperature=2)
    configure_objective(legacy)
    assert (legacy.alpha, legacy.beta, legacy.gamma) == (.6, .3, .1)


def test_decoder_and_output_losses_both_send_only_student_gradients():
    args = liu_args()
    configure_objective(args)
    s_output = torch.tensor([[[1., 0., -1.]]], requires_grad=True)
    t_output = torch.zeros_like(s_output, requires_grad=True)
    s_decoder = torch.tensor([[[1., -1.]]], requires_grad=True)
    t_decoder = torch.zeros_like(s_decoder, requires_grad=True)
    kd, feature = distillation_terms(s_output, t_output, s_decoder, t_decoder,
                                     torch.tensor([[2]]), 0, args)
    (args.beta * kd + args.gamma * feature).backward()
    assert s_output.grad.abs().sum() > 0
    assert s_decoder.grad.abs().sum() > 0
    assert t_output.grad is None and t_decoder.grad is None


def test_training_shares_received_tensor_and_keeps_teacher_frozen():
    from models.transceiver import DeepSC
    from models.tx_model import Transmitter
    from models.rx_model import Receiver
    from train_multi_vocab_one_student import build_student_receiver, configure_trainability, train, validate_epoch

    torch.manual_seed(73)
    full = DeepSC(1, 12, 12, 8, 8, 8, 2, 16, 0.0)
    tx = Transmitter(full.encoder, full.channel_encoder)
    teacher = Receiver(full.channel_decoder, full.decoder, full.dense)
    args = liu_args(max_len=8, d_model=8, dff=16, dropout=0.0,
                    student_num_layers=4, student_num_heads=8,
                    snr_mode='fixed', snr_db=8, grad_clip=1)
    configure_objective(args)
    student = build_student_receiver(12, args, torch.device('cpu'))
    configure_trainability(tx, teacher, student)
    before = deepcopy(full.state_dict())
    student_before = deepcopy(student.state_dict())
    observations = []
    teacher.register_forward_pre_hook(lambda module, inputs, kwargs: observations.append(kwargs['z_noisy'].data_ptr()), with_kwargs=True)
    student.register_forward_pre_hook(lambda module, inputs: observations.append(inputs[0].data_ptr()))
    src = torch.tensor([[1, 4, 5, 2], [1, 4, 6, 2]])
    loader = [(src, src)]
    opt = torch.optim.Adam(student.parameters(), lr=1e-4)
    train(tx, teacher, student, loader, opt, 0, 'AWGN', torch.device('cpu'), None, 0, args)
    assert observations[0] == observations[1]
    assert all(torch.equal(before[k], value) for k, value in full.state_dict().items())
    assert all(p.grad is None for p in full.parameters())
    assert any(not torch.equal(student_before[k], value) for k, value in student.state_dict().items())
    stats = validate_epoch(0, tx, teacher, student, loader, 0, 'AWGN', .2, torch.device('cpu'), None, args)
    assert stats['loss'] == pytest.approx(args.alpha * stats['ce'] + args.beta * stats['kd'] + args.gamma * stats['tf'])
    assert observations[2] == observations[3]
