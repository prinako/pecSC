import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset_multilingual import EurParallelDatasetBPE, collate_parallelBPE
from dataset import EurDataset, collate_data

from datetime import datetime
import argparse
import csv
import json
import math
import os
import random
import time
import warnings
import numpy as np

from models.transceiver import DeepSC
from student import Student
from teacher import build_teacher
from models.rx_model import Receiver
from models.tx_model import Transmitter
from utils.train_utils import (
    create_masks, 
    loss_function, 
    masked_ce_loss, 
    validate_multi_epoch, 
)
from utils.kd_utils import (
    kd_kl_loss, 
    feature_distillation_loss, 
    feature_distillation_loss_cosine,
    feature_distillation_loss_cosine_normalized,
    logit_distillation_loss_cosine,
)
from utils.model_utils import (
    save_epoch_results,
    snr_to_noise, 
    save_student_receiver
)
from utils.experiment_manifest import (
    base_training_manifest, create_run_directory, file_identity,
    update_selected_checkpoint, write_manifest,
)
from utils.publication_data import resolve_training_inputs
from utils.liu_kd import METHOD_ID, liu_weights, masked_student_teacher_kl





def parse_args():
    parser = argparse.ArgumentParser(description="Receiver-only KD for DeepSC", allow_abbrev=False)

    # files
    parser.add_argument(
        "--data-dir", required=True,
        help="Explicit directory containing train_en_en.pkl and val_en_en.pkl",
    )
    parser.add_argument(
        "--vocab-file", type=str, default="vocab_bpe.json",
        help="Absolute path or path relative to --data-dir",
    )
    parser.add_argument(
        "--teacher-encoder", type=str, required=True,
        help="Teacher checkpoint containing encoder and channel_encoder",
    )
    parser.add_argument(
        "--teacher-decoder", type=str, required=True,
        help="Teacher checkpoint containing channel_decoder, decoder and dense",
    )
    parser.add_argument(
        "--save-dir", type=str, default="./checkpoints/deepsc-Rayleigh/multi_vocab_kd_bpe"
    )
    parser.add_argument("--experiment-id", type=str,
                        help="Unique run directory name (default: generated UTC timestamp)")

    # model config (must match teacher checkpoint)
    parser.add_argument("--max-len", type=int, default=67)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--dff", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)

    # train
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)

    # studennt train
    parser.add_argument("--student_num_layers", type=int, default=4)
    parser.add_argument("--student_num_heads", type=int, default=8)

    # channel
    parser.add_argument("--channel", type=str, default="Rayleigh", choices=["AWGN", "Rayleigh", "Rician"])
    parser.add_argument("--snr-mode", type=str, default="range", choices=["fixed", "range"])
    parser.add_argument("--snr-db", type=float, default=8.0)
    parser.add_argument("--snr-db-low", type=float, default=2.0)
    parser.add_argument("--snr-db-high", type=float, default=18.0)
    parser.add_argument("--val-snr-db", type=float, default=8.0)
    parser.add_argument(
        "--diagnostic-mode",
        action="store_true",
        help=(
            "Write per-training-batch channel and numerical diagnostics to "
            "diagnostics.csv in this run's new experiment directory"
        ),
    )

    # KD
    parser.add_argument("--method", choices=["receiver-kd", "liu-receiver"],
                        default="receiver-kd", help="Existing CE/KD path or explicit M3 adaptation")
    parser.add_argument("--liu-eta-decoder", type=float,
                        help="M3 semantic-decoder distribution weight (validation-selected)")
    parser.add_argument("--liu-eta-output", type=float,
                        help="M3 output distribution weight (validation-selected)")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Required explicitly for M3; existing CE/KD defaults to 2")
    parser.add_argument("--alpha", type=float, default=0.6, help="CE weight")
    parser.add_argument("--beta", type=float, default=0.3, help="KD weight")
    parser.add_argument("--gamma", type=float, default=0.1, help="Decoder cosine feature weight (existing KD only)")
    parser.add_argument("--init-student-from-teacher", action="store_true")
    parser.add_argument('--en', default='en_en', type=str)
    parser.add_argument('--en-pt', default='en_pt', type=str)
    parser.add_argument('--en-es', default='en_es', type=str)
    parser.add_argument('--en-fr', default='en_fr', type=str)

    parser.add_argument('--pt', default='pt_pt', type=str)
    parser.add_argument('--pt-pt', default='pt_en', type=str)
    parser.add_argument('--pt-es', default='pt_es', type=str)
    parser.add_argument('--pt-fr', default='pt_fr', type=str)

    args = parser.parse_args()
    if args.method == "liu-receiver":
        # Reject ambiguous legacy weight flags, even when equal to their defaults.
        import sys
        if any(arg.split("=", 1)[0] in {"--alpha", "--beta", "--gamma"}
               for arg in sys.argv[1:]):
            parser.error("M3 uses --liu-eta-decoder/output, not --alpha/beta/gamma")
    try:
        configure_method(args)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def validate_loss_configuration(alpha: float, beta: float, gamma: float, temperature: float) -> str:
    if min(alpha, beta, gamma) < 0:
        raise ValueError("Loss weights alpha, beta and gamma must be non-negative")
    
    if alpha + beta + gamma <= 0:
        raise ValueError("At least one loss weight must be positive")
    
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    
    return "CE-only Student" if beta == 0 and gamma == 0 else "KD Student"


def configure_method(args):
    """Resolve effective weights before constructing data, models or run folders."""
    if args.method == "liu-receiver":
        args.alpha, args.beta, args.gamma = liu_weights(
            args.liu_eta_decoder, args.liu_eta_output, args.temperature
        )
        return "Liu et al. — receiver-side adaptation"
    if args.liu_eta_decoder is not None or args.liu_eta_output is not None:
        raise ValueError("--liu-eta-* requires --method liu-receiver")
    if args.temperature is None:
        args.temperature = 2.0
    return validate_loss_configuration(args.alpha, args.beta, args.gamma, args.temperature)


def receiver_loss_terms(s_logits, t_logits, s_features, t_features, targets, pad_idx, args):
    """One loss dispatch shared by active training and validation."""
    ce, _ = masked_ce_loss(s_logits, targets, pad_idx, label_smoothing=0.1)
    zero = ce.new_zeros(())
    if getattr(args, "method", "receiver-kd") == "liu-receiver":
        kd = (masked_student_teacher_kl(s_logits, t_logits, targets, pad_idx, args.temperature)
              if args.beta > 0 else zero)
        feat = (masked_student_teacher_kl(s_features, t_features, targets, pad_idx, args.temperature)
                if args.gamma > 0 else zero)
    else:
        kd = (kd_kl_loss(s_logits, t_logits, targets, pad_idx, args.temperature)
              if args.beta > 0 else zero)
        feat = (feature_distillation_loss_cosine_normalized(
            student_feat=s_features, teacher_feat=t_features.detach(),
            targets=targets, pad_idx=pad_idx, eps=1e-8
        ) if args.gamma > 0 else zero)
    return ce, kd, feat


def loss_manifest(args):
    m3 = args.method == "liu-receiver"
    return {
        "method_id": METHOD_ID if m3 else ("M1_ce_only" if args.beta == args.gamma == 0 else "M2_receiver_kd"),
        "alpha": args.alpha, "beta": args.beta, "gamma": args.gamma,
        "temperature": args.temperature, "label_smoothing": 0.1,
        "kl_direction": "student||teacher" if m3 else "teacher||student",
        "feature_alignment": "temperature-softmax decoder KL" if m3 else "normalized decoder cosine squared",
        "distribution_axis": "last (features or vocabulary)",
        "reduction": "sum distribution axis; mean non-PAD shifted target tokens",
        "liu_eta_decoder": args.liu_eta_decoder, "liu_eta_output": args.liu_eta_output,
        "liu_reference": "10.1109/TWC.2023.3336941; Eqs.25-28; Table I Student 2" if m3 else None,
    }


def load_teacher_checkpoints(model, encoder_path, decoder_path, device, expected_architecture=None):
    paths = [os.path.abspath(os.path.expanduser(p)) for p in (encoder_path, decoder_path)]
    
    for path in paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Teacher checkpoint does not exist: {path}")
    
    enc = torch.load(paths[0], map_location=device, weights_only=True)
    dec = torch.load(paths[1], map_location=device, weights_only=True)
    
    expected_enc = {"encoder", "channel_encoder"}
    expected_dec = {"channel_decoder", "decoder", "dense"}

    if not isinstance(enc, dict) or not expected_enc.issubset(enc):
        raise ValueError(f"Teacher encoder checkpoint must contain {sorted(expected_enc)}")
    
    if not isinstance(dec, dict) or not expected_dec.issubset(dec):
        raise ValueError(f"Teacher decoder checkpoint must contain {sorted(expected_dec)}")
    
    unexpected_enc = set(enc) - expected_enc - {"meta"}
    unexpected_dec = set(dec) - expected_dec - {"meta"}
    
    if unexpected_enc or unexpected_dec:
        raise ValueError(
            f"Unexpected Teacher checkpoint components: encoder={sorted(unexpected_enc)}, "
            f"decoder={sorted(unexpected_dec)}"
        )
    
    if enc.get("meta") != dec.get("meta"):
        raise ValueError("Teacher encoder and decoder checkpoint metadata do not match")
    
    if expected_architecture is not None:
        recorded = enc.get("meta", {}).get("architecture")
        if recorded is None:
            raise ValueError(
                "Teacher checkpoint lacks architecture metadata required for a publication run"
            )
        differences = {
            key: (recorded.get(key), value)
            for key, value in expected_architecture.items()
            if recorded.get(key) != value
        }
        if differences:
            raise ValueError(f"Teacher checkpoint architecture metadata mismatch: {differences}")
    
    try:
        model.encoder.load_state_dict(enc["encoder"], strict=True)
        model.channel_encoder.load_state_dict(enc["channel_encoder"], strict=True)
        model.channel_decoder.load_state_dict(dec["channel_decoder"], strict=True)
        model.decoder.load_state_dict(dec["decoder"], strict=True)
        model.dense.load_state_dict(dec["dense"], strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "Teacher checkpoint is incompatible with the configured architecture, "
            "vocabulary size, or positional capacity"
        ) from exc
    
    model.requires_grad_(False)
    model.eval()
    
    return paths


def configure_trainability(transmitter, teacher, student):
    transmitter.requires_grad_(False)
    teacher.requires_grad_(False)
    student.requires_grad_(True)
    transmitter.eval()
    teacher.eval()


def build_student_receiver(num_vocab, args, device):
    """Build the single compact receiver architecture used by CE and KD."""
    return Student(
        args.student_num_layers,
        num_vocab,
        num_vocab,
        args.max_len,
        args.max_len,
        args.d_model,
        args.student_num_heads,
        args.dff,
        args.dropout,
    ).to(device)


def setup_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


DIAGNOSTIC_FIELDS = [
    "epoch", "batch_index", "channel", "snr_db", "noise_std",
    "rayleigh_real", "rayleigh_imag", "rayleigh_magnitude",
    "theoretical_noise_amplification",
    "theoretical_noise_power_amplification", "ce", "loss",
    "gradient_norm_before_clipping", "parameter_norm",
    "max_received_signal_magnitude", "has_nan", "has_inf",
    "nonfinite_components",
]


class DiagnosticCSVLogger:
    """Append-only per-batch diagnostics for one newly created run directory."""

    def __init__(self, path):
        self.path = os.fspath(path)
        if os.path.exists(self.path):
            raise FileExistsError(f"Refusing to overwrite diagnostic file: {self.path}")
        with open(self.path, "x", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=DIAGNOSTIC_FIELDS).writeheader()

    def write(self, row):
        with open(self.path, "a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=DIAGNOSTIC_FIELDS).writerow(row)


def _tensor_nonfinite_kind(value):
    if value is None:
        return False, False
    tensor = value.detach() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    return bool(torch.isnan(tensor).any().item()), bool(torch.isinf(tensor).any().item())


def _parameter_l2_norm(parameters):
    norms = [torch.linalg.vector_norm(parameter.detach()) for parameter in parameters]
    if not norms:
        return 0.0
    return float(torch.linalg.vector_norm(torch.stack(norms)).item())


def _gradient_l2_norm(parameters):
    norms = [
        torch.linalg.vector_norm(parameter.grad.detach())
        for parameter in parameters if parameter.grad is not None
    ]
    if not norms:
        return 0.0
    return float(torch.linalg.vector_norm(torch.stack(norms)).item())


def _diagnostic_row(
    *, epoch, batch_index, channel, snr_db, noise_std, channel_metadata,
    ce, loss, gradient_norm, parameter_norm, z_noisy, student,
):
    components = {
        "ce": ce,
        "loss": loss,
        "received_signal": z_noisy,
        "gradients": [p.grad for p in student.parameters() if p.grad is not None],
        "parameters": list(student.parameters()),
    }
    
    has_nan = False
    has_inf = False
    nonfinite_components = []
    
    for name, values in components.items():
        values = values if isinstance(values, list) else [values]
        component_nan = component_inf = False
        
        for value in values:
            value_nan, value_inf = _tensor_nonfinite_kind(value)
            component_nan = component_nan or value_nan
            component_inf = component_inf or value_inf
        
        if component_nan or component_inf:
            nonfinite_components.append(name)
        
        has_nan = has_nan or component_nan
        has_inf = has_inf or component_inf

    row = {
        "epoch": epoch + 1,
        "batch_index": batch_index + 1,
        "channel": channel,
        "snr_db": float(snr_db),
        "noise_std": float(noise_std),
        "rayleigh_real": channel_metadata.get("rayleigh_real"),
        "rayleigh_imag": channel_metadata.get("rayleigh_imag"),
        "rayleigh_magnitude": channel_metadata.get("rayleigh_magnitude"),
        "theoretical_noise_amplification": channel_metadata[
            "theoretical_noise_amplification"
        ],
        "theoretical_noise_power_amplification": channel_metadata[
            "theoretical_noise_power_amplification"
        ],
        "ce": float(ce.detach().item()),
        "loss": float(loss.detach().item()),
        "gradient_norm_before_clipping": float(gradient_norm),
        "parameter_norm": float(parameter_norm),
        "max_received_signal_magnitude": float(z_noisy.detach().abs().max().item()),
        "has_nan": has_nan,
        "has_inf": has_inf,
        "nonfinite_components": ";".join(nonfinite_components),
    }
    
    scalar_values = [
        row["snr_db"], row["noise_std"], row["ce"], row["loss"],
        row["gradient_norm_before_clipping"], row["parameter_norm"],
        row["max_received_signal_magnitude"],
        row["theoretical_noise_amplification"],
        row["theoretical_noise_power_amplification"],
    ]
    
    for value in (row["rayleigh_real"], row["rayleigh_imag"], row["rayleigh_magnitude"]):
        if value is not None:
            scalar_values.append(value)
    
    row["has_nan"] = row["has_nan"] or any(math.isnan(value) for value in scalar_values)
    row["has_inf"] = row["has_inf"] or any(math.isinf(value) for value in scalar_values)
    
    if row["has_nan"] or row["has_inf"]:
        warnings.warn(
            "Non-finite diagnostic at epoch "
            f"{row['epoch']}, batch {row['batch_index']}: "
            f"components={row['nonfinite_components'] or 'derived scalar metrics'}, "
            f"CE={row['ce']}, grad_norm={row['gradient_norm_before_clipping']}, "
            f"|h|={row['rayleigh_magnitude']}",
            RuntimeWarning,
            stacklevel=2,
        )
    return row

# @torch.no_grad()
# def validate_epoch(
#     epoch,
#     transmitter,
#     teacher: Receiver,
#     student: Student,
#     val_loader,
#     pad_idx,
#     channel,
#     noise_std,
#     device: torch.device,
#     criterion,
#     args,
# ):
#     transmitter.eval()
#     teacher.eval()
#     student.eval()


#     total_loss = 0.0
#     total_ce = 0.0
#     total_kd = 0.0
#     total_feat = 0.0
#     total_ce_ppl = 0.0

#     pbar = tqdm(val_loader)

#     with torch.no_grad():
#         for src, trg in pbar:
#             src = src.to(device)
#             trg = trg.to(device)

#             trg_inp = trg[:, :-1]
#             trg_real = trg[:, 1:]

#             src_mask, look_ahead_mask = create_masks(src, trg_inp, pad_idx)

#             tx_en_out, tx_ch_en_out, Tx_sig, z_noisy = transmitter(
#                 src, 
#                 src_mask, 
#                 channel, 
#                 noise_std
#             )

#             use_kd = args.beta > 0 or args.gamma > 0
#             if use_kd:
#                 t_logits, rx_ch_dec_out, rx_dec_out = teacher(
#                     z_noisy=z_noisy, trg_inp=trg_inp,
#                     look_ahead_mask=look_ahead_mask, src_mask=src_mask
#                 )

#             s_logits, s_ch_dec_out, s_dec_out = student(
#                 z_noisy, 
#                 trg_inp, 
#                 look_ahead_mask, 
#                 src_mask
#             )

#             ce, ce_ppl = masked_ce_loss(
#                 s_logits,
#                 trg_real,
#                 pad_idx,
#                 label_smoothing=0.1
#             )

#             # ce = masked_ce_loss(s_logits, trg_real, pad_idx)
#             # ce_s1 = loss_function(
#             #     s1_logits.contiguous().view(-1, s1_logits.size(-1)),
#             #     trg_real.contiguous().view(-1), 
#             #     pad_idx, 
#             #     criterion
#             # )

#             kd = (kd_kl_loss(s_logits, t_logits, trg_real, pad_idx, args.temperature)
#                   if args.beta > 0 else ce.new_zeros(()))

#             # feat = feature_distillation_loss(
#             #     student_feat=s_ch_dec_out, 
#             #     teacher_feat=rx_ch_dec_out.detach(), 
#             #     targets=src,
#             #     pad_idx=pad_idx,
#             # )

#             feat = (feature_distillation_loss_cosine_normalized(
#                 student_feat=s_dec_out, teacher_feat=rx_dec_out.detach(),
#                 targets=trg_real, pad_idx=pad_idx, eps=1e-8
#             ) if args.gamma > 0 else ce.new_zeros(()))

#             weighted_ce = args.alpha * ce
#             weighted_kd = args.beta * kd
#             weighted_feat = args.gamma * feat

#             loss = weighted_ce + weighted_kd + weighted_feat

#             total_loss += float(loss.item())
#             total_ce += float(ce.item())
#             total_kd += float(kd.item())
#             total_feat += float(feat.item())
#             total_ce_ppl += float(ce_ppl)

#             pbar.set_description(f"Epoch {epoch + 1} Valid")

#             pbar.set_postfix(
#                 Loss=f"{loss.item():.3f}",
#                 CE=f"{weighted_ce.item():.3f}",
#                 CE_PPL=f"{ce_ppl:.3f}",
#                 KD=f"{weighted_kd.item():.3f}",
#                 FEAT=f"{weighted_feat.item():.3f}",
#             )

#     n = max(len(val_loader), 1)

#     return{
#         "loss": total_loss / n,
#         "ce": total_ce / n,
#         "kd": total_kd / n,
#         "tf": total_feat / n,
#     }


# def train(
#     transmitter: Transmitter, 
#     teacher: Receiver, 
#     student: Student, 
#     train_loader: DataLoader, 
#     optimizer: optim.Adam,
#     pad_idx,
#     channel,
#     device: torch.device,
#     criterion,
#     epoch,
#     args,
#     diagnostic_logger=None,
#     ):

#     for p in teacher.parameters():
#         p.requires_grad = False
    
#     for p in transmitter.parameters():
#         p.requires_grad = False
    
#     transmitter.eval()
#     teacher.eval()

#     student.train()

#     total_loss = 0.0
#     total_ce = 0.0
#     total_kd = 0.0
#     total_feat = 0.0
#     total_ce_ppl = 0.0

#     num_batches=0.0

#     pbar = tqdm(train_loader)

#     for batch_index, (src, trg) in enumerate(pbar):
#         # print(batch.shape)
#         src = src.to(device)
#         trg = trg.to(device)

#         trg_inp = trg[:, :-1]
#         trg_real = trg[:, 1:]

#         opt = optimizer

#         opt.zero_grad()

#         src_mask, look_ahead_mask = create_masks(src, trg_inp, pad_idx)

#         if args.snr_mode == "fixed":
#             snr_db = args.snr_db
#         else:
#             snr_db = np.random.uniform(
#                 args.snr_db_low,
#                 args.snr_db_high
#             )

#         noise_std = snr_to_noise(snr_db)

#         with torch.no_grad():
#             tx_result = transmitter(
#                 src,
#                 src_mask,
#                 channel,
#                 noise_std,
#                 return_channel_metadata=diagnostic_logger is not None,
#             )
#             if diagnostic_logger is None:
#                 _, _, _, z_noisy = tx_result
#                 channel_metadata = None
#             else:
#                 _, _, _, z_noisy, channel_metadata = tx_result
            
#             if args.beta > 0 or args.gamma > 0:
#                 t_logits, t_rx_ch, t_rx_dec = teacher(
#                     z_noisy=z_noisy, trg_inp=trg_inp,
#                     look_ahead_mask=look_ahead_mask, src_mask=src_mask
#                 )

#         s_logits, s_rx_ch, s_dec_out = student(
#             z_noisy, 
#             trg_inp, 
#             look_ahead_mask, 
#             src_mask
#         )

#         ce, ce_ppl = masked_ce_loss(
#             s_logits,
#             trg_real,
#             pad_idx,
#             label_smoothing=0.1,
#         )

#         kd = (kd_kl_loss(s_logits, t_logits, trg_real, pad_idx, args.temperature)
#               if args.beta > 0 else ce.new_zeros(()))

#         # feat = feature_distillation_loss(
#         #     student_feat=s_rx_ch,
#         #     teacher_feat=t_rx_ch.detach(),
#         #     targets=src,
#         #     pad_idx=pad_idx,
#         # )

#         feat = (feature_distillation_loss_cosine_normalized(
#             student_feat=s_dec_out, teacher_feat=t_rx_dec.detach(),
#             targets=trg_real, pad_idx=pad_idx, eps=1e-8
#         ) if args.gamma > 0 else ce.new_zeros(()))

#         weighted_ce = args.alpha * ce
#         weighted_kd = args.beta * kd
#         weighted_feat = args.gamma * feat

#         loss = weighted_ce + weighted_kd + weighted_feat

#         # loss = (args.alpha * ce) + (args.beta * kd) + args.gamma * feat

#         loss.backward()

#         if diagnostic_logger is not None:
#             gradient_norm = _gradient_l2_norm(student.parameters())
#             parameter_norm = _parameter_l2_norm(student.parameters())
#             diagnostic_logger.write(_diagnostic_row(
#                 epoch=epoch,
#                 batch_index=batch_index,
#                 channel=channel,
#                 snr_db=snr_db,
#                 noise_std=noise_std,
#                 channel_metadata=channel_metadata,
#                 ce=ce,
#                 loss=loss,
#                 gradient_norm=gradient_norm,
#                 parameter_norm=parameter_norm,
#                 z_noisy=z_noisy,
#                 student=student,
#             ))

#         if args.grad_clip is not None and args.grad_clip > 0:
#             torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
#             # torch.nn.utils.clip_grad_norm_(student_2.parameters(), args.grad_clip)
        
#         opt.step()

#         total_loss += float(loss.item())
#         total_ce += float(ce.item())
#         total_kd += float(kd.item())
#         total_feat += float(feat.item())
#         total_ce_ppl += float(ce_ppl)

#         num_batches += 1

#         pbar.set_description(f"Epoch {epoch + 1} Train")

#         pbar.set_postfix(
#             Loss=f"{loss.item():.3f}",
#             CE=f"{weighted_ce.item():.3f}",
#             CE_PPL=f"{ce_ppl:.3f}",
#             KD=f"{weighted_kd.item():.3f}",
#             TF=f"{weighted_feat.item():.3f}"
#         )

#     n = len(train_loader)

#     return {
#             "loss": total_loss / n,
#             "ce": total_ce / n,
#             "kd": total_kd / n,
#             "tf": total_feat / n,
#             "ce_ppl": total_ce_ppl / n
#         }

# ==============================================
# Was change on 31/10/2026 to:
#===============================================

@torch.no_grad()
def batch_nll_and_tokens(logits, targets, pad_idx):
    """Summed unsmoothed NLL and number of non-padding tokens."""
    nll_sum = torch.nn.functional.cross_entropy(
        logits.detach().float().reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=pad_idx,
        reduction="sum",
        label_smoothing=0.0,
    )

    num_tokens = (targets != pad_idx).sum().item()

    return float(nll_sum.item()), int(num_tokens)


def nll_and_perplexity(total_nll, total_tokens):
    """Convert accumulated token-level NLL to dataset perplexity."""
    if total_tokens == 0:
        return float("nan"), float("nan")

    mean_nll = total_nll / total_tokens

    try:
        perplexity = math.exp(mean_nll)
    except OverflowError:
        perplexity = float("inf")

    return mean_nll, perplexity


def validate_epoch(
    epoch,
    transmitter,
    teacher: Receiver,
    student: Student,
    val_loader,
    pad_idx,
    channel,
    noise_std,
    device: torch.device,
    criterion,
    args,
):
    transmitter.eval()
    teacher.eval()
    student.eval()

    total_loss = 0.0
    total_ce = 0.0
    total_kd = 0.0
    total_feat = 0.0

    total_nll = 0.0
    total_tokens = 0
    num_batches = 0

    pbar = tqdm(val_loader)

    with torch.no_grad():
        for src, trg in pbar:
            src = src.to(device)
            trg = trg.to(device)

            trg_inp = trg[:, :-1]
            trg_real = trg[:, 1:]

            src_mask, look_ahead_mask = create_masks(
                src, trg_inp, pad_idx
            )

            _, _, _, z_noisy = transmitter(
                src,
                src_mask,
                channel,
                noise_std,
            )

            if args.beta > 0 or args.gamma > 0:
                t_logits, _, t_rx_dec = teacher(
                    z_noisy=z_noisy,
                    trg_inp=trg_inp,
                    look_ahead_mask=look_ahead_mask,
                    src_mask=src_mask,
                )

            s_logits, _, s_dec_out = student(
                z_noisy,
                trg_inp,
                look_ahead_mask,
                src_mask,
            )

            ce, kd, feat = receiver_loss_terms(
                s_logits, t_logits if args.beta > 0 or args.gamma > 0 else None,
                s_dec_out, t_rx_dec if args.beta > 0 or args.gamma > 0 else None,
                trg_real, pad_idx, args,
            )

            weighted_ce = args.alpha * ce
            weighted_kd = args.beta * kd
            weighted_feat = args.gamma * feat

            loss = weighted_ce + weighted_kd + weighted_feat

            # Independent reporting metric: unsmoothed token NLL.
            batch_nll, batch_tokens = batch_nll_and_tokens(
                s_logits, trg_real, pad_idx
            )

            total_nll += batch_nll
            total_tokens += batch_tokens

            total_loss += float(loss.item())
            total_ce += float(ce.item())
            total_kd += float(kd.item())
            total_feat += float(feat.item())
            num_batches += 1

            _, running_ppl = nll_and_perplexity(
                total_nll, total_tokens
            )

            pbar.set_description(f"Epoch {epoch + 1} Valid")
            pbar.set_postfix(
                Loss=f"{loss.item():.3f}",
                CE=f"{weighted_ce.item():.3f}",
                PPL=f"{running_ppl:.3f}",
                KD=f"{weighted_kd.item():.3f}",
                FEAT=f"{weighted_feat.item():.3f}",
            )

    if num_batches == 0:
        raise ValueError("Validation loader produced no batches.")

    mean_nll, perplexity = nll_and_perplexity(
        total_nll, total_tokens
    )

    return {
        "loss": total_loss / num_batches,
        "ce": total_ce / num_batches,
        "kd": total_kd / num_batches,
        "tf": total_feat / num_batches,
        "nll": mean_nll,
        "perplexity": perplexity,
        "num_tokens": total_tokens,
    }


def train(
    transmitter: Transmitter,
    teacher: Receiver,
    student: Student,
    train_loader: DataLoader,
    optimizer: optim.Adam,
    pad_idx,
    channel,
    device: torch.device,
    criterion,
    epoch,
    args,
    diagnostic_logger=None,
):
    for parameter in teacher.parameters():
        parameter.requires_grad = False

    for parameter in transmitter.parameters():
        parameter.requires_grad = False

    transmitter.eval()
    teacher.eval()
    student.train()

    total_loss = 0.0
    total_ce = 0.0
    total_kd = 0.0
    total_feat = 0.0

    total_nll = 0.0
    total_tokens = 0
    num_batches = 0

    pbar = tqdm(train_loader)

    for batch_index, (src, trg) in enumerate(pbar):
        src = src.to(device)
        trg = trg.to(device)

        trg_inp = trg[:, :-1]
        trg_real = trg[:, 1:]

        optimizer.zero_grad()

        src_mask, look_ahead_mask = create_masks(
            src, trg_inp, pad_idx
        )

        if args.snr_mode == "fixed":
            snr_db = args.snr_db
        else:
            snr_db = np.random.uniform(
                args.snr_db_low,
                args.snr_db_high,
            )

        noise_std = snr_to_noise(snr_db)

        with torch.no_grad():
            tx_result = transmitter(
                src,
                src_mask,
                channel,
                noise_std,
                return_channel_metadata=diagnostic_logger is not None,
            )

            if diagnostic_logger is None:
                _, _, _, z_noisy = tx_result
                channel_metadata = None
            else:
                _, _, _, z_noisy, channel_metadata = tx_result

            if args.beta > 0 or args.gamma > 0:
                t_logits, _, t_rx_dec = teacher(
                    z_noisy=z_noisy,
                    trg_inp=trg_inp,
                    look_ahead_mask=look_ahead_mask,
                    src_mask=src_mask,
                )

        s_logits, _, s_dec_out = student(
            z_noisy,
            trg_inp,
            look_ahead_mask,
            src_mask,
        )

        ce, kd, feat = receiver_loss_terms(
            s_logits, t_logits if args.beta > 0 or args.gamma > 0 else None,
            s_dec_out, t_rx_dec if args.beta > 0 or args.gamma > 0 else None,
            trg_real, pad_idx, args,
        )

        weighted_ce = args.alpha * ce
        weighted_kd = args.beta * kd
        weighted_feat = args.gamma * feat

        loss = weighted_ce + weighted_kd + weighted_feat

        # Detached metric; it contributes no gradients.
        batch_nll, batch_tokens = batch_nll_and_tokens(
            s_logits, trg_real, pad_idx
        )

        total_nll += batch_nll
        total_tokens += batch_tokens

        loss.backward()

        if diagnostic_logger is not None:
            gradient_norm = _gradient_l2_norm(
                student.parameters()
            )
            parameter_norm = _parameter_l2_norm(
                student.parameters()
            )

            diagnostic_logger.write(
                _diagnostic_row(
                    epoch=epoch,
                    batch_index=batch_index,
                    channel=channel,
                    snr_db=snr_db,
                    noise_std=noise_std,
                    channel_metadata=channel_metadata,
                    ce=ce,
                    loss=loss,
                    gradient_norm=gradient_norm,
                    parameter_norm=parameter_norm,
                    z_noisy=z_noisy,
                    student=student,
                )
            )

        if args.grad_clip is not None and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                student.parameters(),
                args.grad_clip,
            )

        optimizer.step()

        total_loss += float(loss.item())
        total_ce += float(ce.item())
        total_kd += float(kd.item())
        total_feat += float(feat.item())
        num_batches += 1

        _, running_ppl = nll_and_perplexity(
            total_nll, total_tokens
        )

        pbar.set_description(f"Epoch {epoch + 1} Train")
        pbar.set_postfix(
            Loss=f"{loss.item():.3f}",
            CE=f"{weighted_ce.item():.3f}",
            PPL=f"{running_ppl:.3f}",
            KD=f"{weighted_kd.item():.3f}",
            TF=f"{weighted_feat.item():.3f}",
        )

    if num_batches == 0:
        raise ValueError("Training loader produced no batches.")

    mean_nll, perplexity = nll_and_perplexity(
        total_nll, total_tokens
    )

    return {
        "loss": total_loss / num_batches,
        "ce": total_ce / num_batches,
        "kd": total_kd / num_batches,
        "tf": total_feat / num_batches,
        "nll": mean_nll,
        "perplexity": perplexity,
        "num_tokens": total_tokens,
    }


def main():
    args = parse_args()
    model_type = configure_method(args)
    if args.init_student_from_teacher:
        raise ValueError(
            "--init-student-from-teacher is not implemented; publication Students use fresh initialization"
        )
    setup_seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    data_dir, vocab_file, dataset_files = resolve_training_inputs(
        args.data_dir, args.vocab_file, args.en, args.max_len
    )
    args.data_dir, args.vocab_file = str(data_dir), str(vocab_file)
    with open(args.vocab_file, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    
    token_to_idx = vocab["token_to_idx"]
    num_vocab = vocab['vocab_size']
    
    pad_idx = token_to_idx["<PAD>"] if "<PAD>" in token_to_idx else token_to_idx[""]
    
    start_idx = token_to_idx["<START>"]
    end_idx = token_to_idx["<END>"]

    train_set = EurParallelDatasetBPE(args.en, "train", args.data_dir)
    val_set = EurParallelDatasetBPE(args.en, "val", args.data_dir)

    # load dataset
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_parallelBPE,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_parallelBPE,
    )

    deepsc = DeepSC(
        args.num_layers, 
        num_vocab,
        num_vocab,
        args.max_len,
        args.max_len,
        args.d_model,
        args.num_heads,
        args.dff,
        dropout=args.dropout
    ).to(device)

    # deepsc = build_teacher(
    #     num_vocab, 
    #     args.max_len, 
    #     args.num_layers, 
    #     args.d_model, 
    #     args.num_heads, 
    #     args.dff, 
    #     args.dropout, 
    #     device
    # )
    teacher_paths = load_teacher_checkpoints(
        deepsc, args.teacher_encoder, args.teacher_decoder, device,
        expected_architecture={
            "num_layers": args.num_layers, "num_heads": args.num_heads,
            "d_model": args.d_model, "dff": args.dff,
            "max_len": args.max_len, "vocab_size": num_vocab,
        },
    )
    # encoder = deepsc.load_state_dict(torch.load(args.teacher_checkpoint, map_location=device))
    # deep_sc.load_state_dict(torch.load(args.teacher_checkpoint, map_location=device))
    # deep_sc.load_state_dict(torch.load('deepsc_12n.pth', map_location=device))
    # deep_sc.eval()

    transmitter = Transmitter(deepsc.encoder, deepsc.channel_encoder)
    
    receiver = Receiver(
        channel_decoder=deepsc.channel_decoder, 
        decoder=deepsc.decoder, 
        dense=deepsc.dense
    )

    student = build_student_receiver(num_vocab, args, device)
    configure_trainability(transmitter, receiver, student)

    noise_std = snr_to_noise(float(args.val_snr_db))
    # noise_std = np.random.uniform(
    #         snr_to_noise(args.snr_db_low), 
    #         snr_to_noise(args.snr_db_high), 
    #         # size=(1)
    #     )

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer= torch.optim.Adam(
            student.parameters(),
            lr=args.lr,
            betas=(0.9, 0.98),
            eps=1e-8,
            weight_decay = args.weight_decay
        )
    
    best_val = float("inf")
    experiment_id = args.experiment_id or datetime.utcnow().strftime("student-%Y%m%dT%H%M%SZ")
    root_dir = str(create_run_directory(args.save_dir, experiment_id))
    csv_path = os.path.join(root_dir, "training.csv")
    manifest_path = os.path.join(root_dir, "manifest.json")
    diagnostic_logger = (
        DiagnosticCSVLogger(os.path.join(root_dir, "diagnostics.csv"))
        if args.diagnostic_mode else None
    )
    manifest = base_training_manifest(
        experiment_id=experiment_id, model_type=model_type, cli_args=vars(args)
    )
    manifest.update({
        "architecture": {"teacher_layers": args.num_layers, "teacher_heads": args.num_heads,
                         "student_layers": args.student_num_layers,
                         "student_heads": args.student_num_heads, "d_model": args.d_model,
                         "dff": args.dff, "max_length": args.max_len,
                         "vocab_size": num_vocab, "student_initialization": "fresh seeded initialization"
                        },
        "dataset": {name: file_identity(path) for name, path in dataset_files.items()},
        "tokenizer_vocab": file_identity(args.vocab_file),
        "teacher_checkpoints": {"encoder": file_identity(teacher_paths[0]),
                                "decoder": file_identity(teacher_paths[1])},
        "seed": args.seed,
        "channel": {"model": args.channel, "train_snr_mode": args.snr_mode,
                    "train_snr_db": args.snr_db, "train_snr_db_low": args.snr_db_low,
                    "train_snr_db_high": args.snr_db_high,
                    "validation_snr_db": args.val_snr_db,
                    "power_normalization": "unconditional batch RMS (current PowerNormalize)"},
        "optimizer": {"name": "Adam", "lr": args.lr, "betas": [0.9, 0.98],
                      "eps": 1e-8, "weight_decay": args.weight_decay},
        "loss": loss_manifest(args),
        "batch_size": args.batch_size, "epoch_budget": args.epochs,
        "checkpoint_selection_rule": "minimum validation cross-entropy",
        "diagnostics": {
            "enabled": args.diagnostic_mode,
            "file": "diagnostics.csv" if args.diagnostic_mode else None,
            "batch_indices": "one-based",
            "noise_amplification": "1 / |h| (amplitude/std); power column is 1 / |h|^2",
        },
    })
    write_manifest(manifest_path, manifest)

    for epoch in range(args.epochs):
        start_time = time.time()

        train_stats = train(
            transmitter=transmitter, 
            teacher=receiver, 
            student=student, 
            train_loader=train_loader, 
            optimizer=optimizer,
            pad_idx=pad_idx,
            device=device,
            channel=args.channel,
            criterion=criterion,
            epoch=epoch,
            args=args,
            diagnostic_logger=diagnostic_logger,
        )

        val_stats = validate_epoch(
            epoch=epoch,
            transmitter=transmitter, 
            teacher=receiver, 
            student=student, 
            val_loader=val_loader, 
            pad_idx=pad_idx,
            device=device,
            channel=args.channel,
            noise_std=noise_std,
            criterion=criterion,
            args=args
        )

        elapsed = time.time() - start_time

        epoch_metrics = {
            "loss": train_stats["loss"],
            "val_loss": val_stats["loss"],
            "ce": train_stats["ce"],
            "kd": train_stats["kd"],
            "tf": train_stats["tf"],
            "val_ce": val_stats["ce"],
            "val_kd": val_stats["kd"],
            "val_tf": val_stats["tf"],
            "nll": train_stats["nll"],
            "perplexity": train_stats["perplexity"],
            "val_nll": val_stats["nll"],
            "val_perplexity": val_stats["perplexity"],
            "num_tokens": train_stats["num_tokens"],
            "val_num_tokens": val_stats["num_tokens"],
            "alpha": args.alpha,
            "beta": args.beta,
            "gamma": args.gamma,
            "temperature": args.temperature,
            "time": elapsed
        }

        save_epoch_results(
            csv_path=csv_path,
            epoch=epoch,
            metrics=epoch_metrics,
        )

        latest_path = os.path.join(
            root_dir, f"student_{epoch+1:02d}.pth"
        )
        # save the student model by epoch
        save_student_receiver(
            student,
            latest_path,
            meta={
                "epoch": epoch + 1,
                "val_loss": val_stats["loss"],
                "temperature": args.temperature,
                "alpha": args.alpha,
                "beta": args.beta,
                "gamma": args.gamma,
                "channel": args.channel,
                "loss_configuration": loss_manifest(args),
            },
        )

        # save the best student model        
        if val_stats["ce"] < best_val:
            best_val = val_stats["ce"]

            best_path = os.path.join(
                root_dir,
                "student_best.pth"
            )

            save_student_receiver(
                student,
                best_path,
                meta={
                    "epoch": epoch + 1,
                    "val_loss": val_stats["loss"],
                    "val_ce": val_stats["ce"],
                    "temperature": args.temperature,
                    "alpha": args.alpha,
                    "beta": args.beta,
                    "gamma": args.gamma,
                    "channel": args.channel,
                    "loss_configuration": loss_manifest(args),
                },
            )
            update_selected_checkpoint(
                manifest_path, best_path, epoch + 1,
                "validation_cross_entropy", best_val
            )

            print(f"  -> saved best student TR to: {best_path}")

if __name__ == "__main__":
    main()
