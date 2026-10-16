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
import json
import os
import random
import time
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


STUDENT_NUM_LAYERS = 4
STUDENT_NUM_HEADS = 8


def parse_args():
    parser = argparse.ArgumentParser(description="Receiver-only KD for DeepSC")

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
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)

    # channel
    parser.add_argument("--channel", type=str, default="Rayleigh", choices=["AWGN", "Rayleigh", "Rician"])
    parser.add_argument("--snr-mode", type=str, default="range", choices=["fixed", "range"])
    parser.add_argument("--snr-db", type=float, default=8.0)
    parser.add_argument("--snr-db-low", type=float, default=2.0)
    parser.add_argument("--snr-db-high", type=float, default=18.0)
    parser.add_argument("--val-snr-db", type=float, default=8.0)

    # KD
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.6, help="CE weight")
    parser.add_argument("--beta", type=float, default=0.3, help="KD weight")
    parser.add_argument("--gamma", type=float, default=0.1, help="Feature MSE weight")
    parser.add_argument("--init-student-from-teacher", action="store_true")
    parser.add_argument('--en', default='en_en', type=str)
    parser.add_argument('--en-pt', default='en_pt', type=str)
    parser.add_argument('--en-es', default='en_es', type=str)
    parser.add_argument('--en-fr', default='en_fr', type=str)

    parser.add_argument('--pt', default='pt_pt', type=str)
    parser.add_argument('--pt-pt', default='pt_en', type=str)
    parser.add_argument('--pt-es', default='pt_es', type=str)
    parser.add_argument('--pt-fr', default='pt_fr', type=str)

    return parser.parse_args()


def validate_loss_configuration(alpha: float, beta: float, gamma: float, temperature: float) -> str:
    if min(alpha, beta, gamma) < 0:
        raise ValueError("Loss weights alpha, beta and gamma must be non-negative")
    if alpha + beta + gamma <= 0:
        raise ValueError("At least one loss weight must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return "CE-only Student" if beta == 0 and gamma == 0 else "KD Student"


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
        STUDENT_NUM_LAYERS,
        num_vocab,
        num_vocab,
        args.max_len,
        args.max_len,
        args.d_model,
        STUDENT_NUM_HEADS,
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

@torch.no_grad()
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
    total_ce_ppl = 0.0

    pbar = tqdm(val_loader)

    with torch.no_grad():
        for src, trg in pbar:
            src = src.to(device)
            trg = trg.to(device)

            trg_inp = trg[:, :-1]
            trg_real = trg[:, 1:]

            src_mask, look_ahead_mask = create_masks(src, trg_inp, pad_idx)

            tx_en_out, tx_ch_en_out, Tx_sig, z_noisy = transmitter(
                src, 
                src_mask, 
                channel, 
                noise_std
            )

            use_kd = args.beta > 0 or args.gamma > 0
            if use_kd:
                t_logits, rx_ch_dec_out, rx_dec_out = teacher(
                    z_noisy=z_noisy, trg_inp=trg_inp,
                    look_ahead_mask=look_ahead_mask, src_mask=src_mask
                )

            s_logits, s_ch_dec_out, s_dec_out = student(
                z_noisy, 
                trg_inp, 
                look_ahead_mask, 
                src_mask
            )

            ce, ce_ppl = masked_ce_loss(
                s_logits,
                trg_real,
                pad_idx,
                label_smoothing=0.1
            )

            # ce = masked_ce_loss(s_logits, trg_real, pad_idx)
            # ce_s1 = loss_function(
            #     s1_logits.contiguous().view(-1, s1_logits.size(-1)),
            #     trg_real.contiguous().view(-1), 
            #     pad_idx, 
            #     criterion
            # )

            kd = (kd_kl_loss(s_logits, t_logits, trg_real, pad_idx, args.temperature)
                  if args.beta > 0 else ce.new_zeros(()))

            # feat = feature_distillation_loss(
            #     student_feat=s_ch_dec_out, 
            #     teacher_feat=rx_ch_dec_out.detach(), 
            #     targets=src,
            #     pad_idx=pad_idx,
            # )

            feat = (feature_distillation_loss_cosine_normalized(
                student_feat=s_dec_out, teacher_feat=rx_dec_out.detach(),
                targets=trg_real, pad_idx=pad_idx, eps=1e-8
            ) if args.gamma > 0 else ce.new_zeros(()))

            weighted_ce = args.alpha * ce
            weighted_kd = args.beta * kd
            weighted_feat = args.gamma * feat

            loss = weighted_ce + weighted_kd + weighted_feat

            total_loss += float(loss.item())
            total_ce += float(ce.item())
            total_kd += float(kd.item())
            total_feat += float(feat.item())
            total_ce_ppl += float(ce_ppl)

            pbar.set_description(f"Epoch {epoch + 1} Valid")

            pbar.set_postfix(
                Loss=f"{loss.item():.3f}",
                CE=f"{weighted_ce.item():.3f}",
                CE_PPL=f"{ce_ppl:.3f}",
                KD=f"{weighted_kd.item():.3f}",
                FEAT=f"{weighted_feat.item():.3f}",
            )

    n = max(len(val_loader), 1)

    return{
        "loss": total_loss / n,
        "ce": total_ce / n,
        "kd": total_kd / n,
        "tf": total_feat / n,
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
    args
    ):

    for p in teacher.parameters():
        p.requires_grad = False
    
    for p in transmitter.parameters():
        p.requires_grad = False
    
    transmitter.eval()
    teacher.eval()

    student.train()

    total_loss = 0.0
    total_ce = 0.0
    total_kd = 0.0
    total_feat = 0.0
    total_ce_ppl = 0.0

    num_batches=0.0

    pbar = tqdm(train_loader)

    for src, trg in pbar:
        # print(batch.shape)
        src = src.to(device)
        trg = trg.to(device)

        trg_inp = trg[:, :-1]
        trg_real = trg[:, 1:]

        opt = optimizer

        opt.zero_grad()

        src_mask, look_ahead_mask = create_masks(src, trg_inp, pad_idx)

        if args.snr_mode == "fixed":
            snr_db = args.snr_db
        else:
            snr_db = np.random.uniform(
                args.snr_db_low,
                args.snr_db_high
            )

        noise_std = snr_to_noise(snr_db)

        with torch.no_grad():
            _, _, _, z_noisy = transmitter(
                src, 
                src_mask, 
                channel, 
                noise_std
            )
            
            if args.beta > 0 or args.gamma > 0:
                t_logits, t_rx_ch, t_rx_dec = teacher(
                    z_noisy=z_noisy, trg_inp=trg_inp,
                    look_ahead_mask=look_ahead_mask, src_mask=src_mask
                )

        s_logits, s_rx_ch, s_dec_out = student(
            z_noisy, 
            trg_inp, 
            look_ahead_mask, 
            src_mask
        )

        ce, ce_ppl = masked_ce_loss(
            s_logits,
            trg_real,
            pad_idx,
            label_smoothing=0.1,
        )

        kd = (kd_kl_loss(s_logits, t_logits, trg_real, pad_idx, args.temperature)
              if args.beta > 0 else ce.new_zeros(()))

        # feat = feature_distillation_loss(
        #     student_feat=s_rx_ch,
        #     teacher_feat=t_rx_ch.detach(),
        #     targets=src,
        #     pad_idx=pad_idx,
        # )

        feat = (feature_distillation_loss_cosine_normalized(
            student_feat=s_dec_out, teacher_feat=t_rx_dec.detach(),
            targets=trg_real, pad_idx=pad_idx, eps=1e-8
        ) if args.gamma > 0 else ce.new_zeros(()))

        weighted_ce = args.alpha * ce
        weighted_kd = args.beta * kd
        weighted_feat = args.gamma * feat

        loss = weighted_ce + weighted_kd + weighted_feat

        # loss = (args.alpha * ce) + (args.beta * kd) + args.gamma * feat

        loss.backward()
        
        if args.grad_clip is not None and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
            # torch.nn.utils.clip_grad_norm_(student_2.parameters(), args.grad_clip)
        
        opt.step()

        total_loss += float(loss.item())
        total_ce += float(ce.item())
        total_kd += float(kd.item())
        total_feat += float(feat.item())
        total_ce_ppl += float(ce_ppl)

        num_batches += 1

        pbar.set_description(f"Epoch {epoch + 1} Train")

        pbar.set_postfix(
            Loss=f"{loss.item():.3f}",
            CE=f"{weighted_ce.item():.3f}",
            CE_PPL=f"{ce_ppl:.3f}",
            KD=f"{weighted_kd.item():.3f}",
            TF=f"{weighted_feat.item():.3f}"
        )

    n = len(train_loader)

    return {
            "loss": total_loss / n,
            "ce": total_ce / n,
            "kd": total_kd / n,
            "tf": total_feat / n,
            "ce_ppl": total_ce_ppl / n
        }

def main():
    args = parse_args()
    model_type = validate_loss_configuration(
        args.alpha, args.beta, args.gamma, args.temperature
    )
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
    manifest = base_training_manifest(
        experiment_id=experiment_id, model_type=model_type, cli_args=vars(args)
    )
    manifest.update({
        "architecture": {"teacher_layers": args.num_layers, "teacher_heads": args.num_heads,
                         "student_layers": STUDENT_NUM_LAYERS,
                         "student_heads": STUDENT_NUM_HEADS, "d_model": args.d_model,
                         "dff": args.dff, "max_length": args.max_len,
                         "vocab_size": num_vocab, "student_initialization": "fresh seeded initialization"},
        "dataset": {name: file_identity(path) for name, path in dataset_files.items()},
        "tokenizer_vocab": file_identity(args.vocab_file),
        "teacher_checkpoints": {"encoder": file_identity(teacher_paths[0]),
                                "decoder": file_identity(teacher_paths[1])},
        "seed": args.seed,
        "channel": {"model": args.channel, "train_snr_mode": args.snr_mode,
                    "train_snr_db": args.snr_db, "train_snr_db_low": args.snr_db_low,
                    "train_snr_db_high": args.snr_db_high,
                    "validation_snr_db": args.val_snr_db},
        "optimizer": {"name": "Adam", "lr": args.lr, "betas": [0.9, 0.98],
                      "eps": 1e-8, "weight_decay": args.weight_decay},
        "loss": {"alpha": args.alpha, "beta": args.beta, "gamma": args.gamma,
                 "temperature": args.temperature, "label_smoothing": 0.1,
                 "feature_alignment": "normalized decoder cosine squared"},
        "batch_size": args.batch_size, "epoch_budget": args.epochs,
        "checkpoint_selection_rule": "minimum validation cross-entropy",
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
            args=args
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
            "ce_ppl": train_stats["ce_ppl"],
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
                },
            )
            update_selected_checkpoint(
                manifest_path, best_path, epoch + 1,
                "validation_cross_entropy", best_val
            )

            print(f"  -> saved best student TR to: {best_path}")

if __name__ == "__main__":
    main()
