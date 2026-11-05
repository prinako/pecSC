"""Liu et al. receiver-only adaptation (Eqs. 25–28, Table I Student 2).

This implements the retained decoder/output signals, not an end-to-end or
multi-user reproduction. See docs/LIU_BASELINE_PLAN.md for the protocol.
"""
import math

import torch.nn.functional as F

METHOD_ID = "M3_liu_receiver_v1"


def liu_weights(eta_decoder, eta_output, temperature):
    """Complementary CE weight; numerical values must be chosen on validation."""
    values = (eta_decoder, eta_output, temperature)
    if any(value is None or not math.isfinite(value) for value in values):
        raise ValueError("M3 requires explicit finite decoder/output weights and temperature")
    if min(eta_decoder, eta_output) < 0 or eta_decoder + eta_output > 1:
        raise ValueError("M3 weights must be non-negative and sum to at most one")
    if eta_decoder + eta_output == 0:
        raise ValueError("M3 requires at least one positive distillation weight")
    if temperature <= 0:
        raise ValueError("M3 temperature must be positive")
    return 1.0 - (eta_decoder + eta_output), eta_output, eta_decoder


def masked_student_teacher_kl(student, teacher, targets, pad_idx, temperature):
    """T² KL(Student || detached Teacher), summed over the last axis.

    Inputs are [batch, target_length, feature_or_vocab_width]. The mean is
    over non-PAD shifted target positions, including EOS. Float32 protects
    half precision inputs; float64 is preserved for numerical checks.
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if student.ndim != 3 or student.shape != teacher.shape or student.shape[:2] != targets.shape:
        raise ValueError("Matching [B,T,D] tensors and [B,T] targets are required")
    if student.size(-1) == 0:
        raise ValueError("The distribution axis must be nonempty")
    if student.element_size() < 4:
        student = student.float()
    teacher = teacher.detach().to(student.dtype)
    log_s = F.log_softmax(student / temperature, dim=-1)
    log_t = F.log_softmax(teacher / temperature, dim=-1)
    token_kl = (log_s.exp() * (log_s - log_t)).sum(dim=-1)
    valid = targets.ne(pad_idx)
    return token_kl.masked_fill(~valid, 0).sum() / valid.sum().clamp_min(1) * temperature**2
