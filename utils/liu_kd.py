"""Liu et al. receiver-only adaptation; see docs/LIU_BASELINE_PLAN.md.

Implements the Student-first KL orientation printed in Eqs. (26)-(28).
Feature-axis softmax and non-PAD token averaging are explicit adaptation choices.
"""

import math

import torch.nn.functional as F


METHOD_ID = "liu_rx_v1"


def validate_liu_weights(decoder_weight, logit_weight, temperature):
    """Require both Table-I receiver signals and a positive hard-label weight."""
    values = (decoder_weight, logit_weight, temperature)
    if any(value is None or not math.isfinite(value) for value in values):
        raise ValueError("Liu decoder/logit weights and temperature must be explicit finite values")
    if decoder_weight <= 0 or logit_weight <= 0:
        raise ValueError("Liu decoder and logit weights must both be positive")
    if decoder_weight + logit_weight >= 1:
        raise ValueError("Liu decoder + logit weights must be less than one")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return 1.0 - decoder_weight - logit_weight


def configure_liu_objective(args):
    """Set the effective CE, output KL, and decoder KL coefficients."""
    args.alpha = validate_liu_weights(
        args.liu_decoder_weight, args.liu_logit_weight, args.temperature
    )
    args.beta = args.liu_logit_weight
    args.gamma = args.liu_decoder_weight
    args.method = METHOD_ID


def liu_distillation_terms(s_logits, t_logits, s_decoder, t_decoder, targets, pad_idx, args):
    return (
        student_teacher_kl(s_logits, t_logits, targets, pad_idx, args.temperature),
        student_teacher_kl(s_decoder, t_decoder, targets, pad_idx, args.temperature),
    )


def liu_objective_manifest(args):
    return {
        "method": METHOD_ID,
        "alpha": args.alpha, "beta": args.beta, "gamma": args.gamma,
        "temperature": args.temperature, "label_smoothing": 0.1,
        "kl_direction": "student||teacher",
        "feature_alignment": "temperature-softmax decoder KL student||teacher",
        "distillation_reduction": "sum final axis; mean non-PAD shifted target tokens",
        "logit_temperature_squared": True,
        "feature_temperature_squared": True,
        "source": "10.1109/TWC.2023.3336941; Table I Student 2; Eqs. 25-28",
        "adaptation": "frozen shared TX; decoder/output transfer only; smoothed causal CE",
        "liu_decoder_weight": args.liu_decoder_weight,
        "liu_logit_weight": args.liu_logit_weight,
    }


def student_teacher_kl(student, teacher, targets, pad_idx, temperature):
    """T^2 * mean_nonpad KL(softmax(Student/T) || softmax(Teacher/T)).

    Representations have shape [B, time, features/classes]. Sum KL along the
    final axis, then average over valid shifted target tokens. Teacher values
    are detached; only the Student receives gradients. PAD rows are excluded
    before softmax so that even nonfinite PAD values cannot contaminate loss.
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if student.ndim != 3 or student.shape != teacher.shape:
        raise ValueError("Liu Student/Teacher tensors must have identical [B,T,D] shapes")
    if targets.ndim != 2 or student.shape[:2] != targets.shape or student.shape[-1] == 0:
        raise ValueError("Liu tensors must match [B,T] targets and have a nonempty feature axis")
    valid = targets != pad_idx
    student_valid = student[valid].float()
    teacher_valid = teacher.detach()[valid].float()
    if student_valid.shape[0] == 0:
        return student_valid.sum()
    student_log = F.log_softmax(student_valid / temperature, dim=-1)
    teacher_log = F.log_softmax(teacher_valid / temperature, dim=-1)
    return (student_log.exp() * (student_log - teacher_log)).sum(-1).mean() * temperature**2
