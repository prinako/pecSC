"""Paired, teacher-forced validation for pecSC:kd-exp. No training/test access."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return {'sha256': digest.hexdigest(), 'size_bytes': path.stat().st_size}


def verify(path, expected):
    actual = identity(path)
    if any(actual[key] != expected.get(key) for key in actual):
        raise ValueError(f'Artifact hash/size mismatch: {path}')
    return actual


def signature(manifest):
    """Compare identities, not machine-specific paths or selected epochs."""
    return {
        'architecture': manifest['architecture'],
        'dataset': {k: (v['sha256'], v['size_bytes']) for k, v in manifest['dataset'].items()},
        'vocab': manifest['tokenizer_vocab']['sha256'],
        'teacher': {k: v['sha256'] for k, v in manifest['teacher_checkpoints'].items()},
        'optimizer': manifest['optimizer'], 'channel': manifest['channel'],
        'batch_size': manifest['batch_size'], 'epoch_budget': manifest['epoch_budget'],
        'smoothing': manifest['loss']['label_smoothing'],
        'controls': {k: manifest['cli_arguments'][k] for k in
                     ('dropout', 'grad_clip', 'init_student_from_teacher')},
    }


def preflight(args):
    entries = []
    for seed in args.training_seeds:
        for label, template, weights in [
            ('m1', args.m1_template, (1.0, 0.0, 0.0)),
            ('feature', args.feature_template, (1.0, 0.0, 0.1)),
        ]:
            directory = args.run_root / template.format(seed=seed)
            manifest_path = directory / 'manifest.json'
            m = json.loads(manifest_path.read_text())
            if m['seed'] != seed or m['cli_arguments']['seed'] != seed:
                raise ValueError(f'Seed mismatch: {manifest_path}')
            actual_weights = tuple(m['loss'][k] for k in ('alpha', 'beta', 'gamma'))
            if actual_weights != weights:
                raise ValueError(f'Unexpected {label} weights: {actual_weights}')
            if m['checkpoint_selection_rule'] != 'minimum validation cross-entropy':
                raise ValueError('Run did not select by validation CE')
            selected = m['selected_checkpoint']
            if selected['selection_metric'] != 'validation_cross_entropy':
                raise ValueError('Selected checkpoint metric mismatch')
            # Resolve relative to run directory: original machine paths may have moved.
            checkpoint = directory / Path(selected['path']).name
            verify(checkpoint, selected)
            entries.append({'seed': seed, 'label': label, 'manifest': m,
                            'manifest_path': str(manifest_path.resolve()),
                            'checkpoint': str(checkpoint.resolve())})
    reference = entries[0]['manifest']
    for entry in entries:
        if signature(entry['manifest']) != signature(reference):
            raise ValueError(f'Incompatible run controls: {entry["manifest_path"]}')
    if reference['channel']['model'] != 'Rayleigh':
        raise ValueError('This protocol requires the existing Rayleigh channel')
    if reference['loss']['label_smoothing'] != 0.1:
        raise ValueError('Expected Student label smoothing 0.1')
    if reference['architecture']['student_initialization'] != 'fresh seeded initialization':
        raise ValueError('Unexpected initialization policy')
    if reference['cli_arguments']['init_student_from_teacher']:
        raise ValueError('Teacher initialization is outside this comparison')
    # Hash only validation/metadata inputs, never open train/test payloads.
    paths = {'val': args.data_dir / 'val_en_en.pkl',
             'dataset_manifest': args.data_dir / 'dataset_manifest.json',
             'tokenizer_model': args.data_dir / 'tokenizer_bpe.model',
             'split_ids': args.data_dir / 'split_ids.json'}
    for key, path in paths.items():
        verify(path, reference['dataset'][key])
    vocab_path = args.data_dir / 'vocab_bpe.json'
    verify(vocab_path, reference['tokenizer_vocab'])
    verify(args.teacher_encoder, reference['teacher_checkpoints']['encoder'])
    verify(args.teacher_decoder, reference['teacher_checkpoints']['decoder'])
    data = json.loads(paths['dataset_manifest'].read_text())
    if data['duplicate_statistics']['cross_split_overlaps'] != {
            'train_val': 0, 'train_test': 0, 'val_test': 0}:
        raise ValueError('Dataset manifest reports leakage')
    return entries, reference, data, vocab_path


def run_trial(torch, transmitter, models, loader, trial_seed, noise_std,
              device, pad_idx, make_masks, max_batches=None):
    """One transmitter/channel call per batch, then all receivers on the same Z."""
    import numpy as np
    import torch.nn.functional as F

    random.seed(trial_seed)
    np.random.seed(trial_seed)
    torch.manual_seed(trial_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(trial_seed)
    transmitter.eval().requires_grad_(False)
    for model in models.values():
        model.eval().requires_grad_(False)
    totals = {key: {'ce_sum': 0.0, 'nll_sum': 0.0, 'batch_ce_sum': 0.0,
                    'tokens': 0, 'batches': 0, 'examples': 0} for key in models}
    with torch.inference_mode():
        for batch_index, (src, trg) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            src, trg = src.to(device), trg.to(device)
            prefix, target = trg[:, :-1], trg[:, 1:]
            source_mask, causal_mask = make_masks(src, prefix, pad_idx)
            received = transmitter(src, source_mask, 'Rayleigh', noise_std)[3]
            if not torch.isfinite(received).all():
                raise FloatingPointError(f'Nonfinite channel output in trial {trial_seed}, batch {batch_index}')
            tokens = int((target != pad_idx).sum().item())
            if tokens == 0:
                raise ValueError('Validation batch contains no valid target tokens')
            for key, model in models.items():
                logits = model(received, prefix, causal_mask, source_mask)[0]
                if not torch.isfinite(logits).all():
                    raise FloatingPointError(f'Nonfinite logits: {key}, trial {trial_seed}, batch {batch_index}')
                flat, targets = logits.flatten(0, 1), target.flatten()
                ce = F.cross_entropy(flat, targets, ignore_index=pad_idx,
                                     label_smoothing=0.1, reduction='sum')
                nll = F.cross_entropy(flat, targets, ignore_index=pad_idx,
                                      label_smoothing=0.0, reduction='sum')
                ce_value, nll_value = float(ce.item()), float(nll.item())
                if not math.isfinite(ce_value + nll_value):
                    raise FloatingPointError(f'Nonfinite loss: {key}, trial {trial_seed}')
                acc = totals[key]
                acc['ce_sum'] += ce_value
                acc['nll_sum'] += nll_value
                acc['batch_ce_sum'] += ce_value / tokens
                acc['tokens'] += tokens
                acc['batches'] += 1
                acc['examples'] += src.shape[0]
                del logits, flat, ce, nll
            if (batch_index + 1) % 500 == 0:
                print(f'Trial {trial_seed}: {batch_index + 1} batches', flush=True)
    results = {}
    for key, acc in totals.items():
        if not acc['tokens']:
            raise ValueError('Empty validation loader')
        nll = acc['nll_sum'] / acc['tokens']
        results[key] = {'batch_mean_ce': acc['batch_ce_sum'] / acc['batches'],
                        'token_mean_ce': acc['ce_sum'] / acc['tokens'],
                        'token_mean_nll': nll,
                        'unsmoothed_ppl': math.exp(nll) if nll < 700 else float('inf'),
                        **{k: acc[k] for k in ('tokens', 'batches', 'examples')}}
    return results


def paired_summary(rows, training_seeds, trial_seeds):
    metrics = ('batch_mean_ce', 'token_mean_ce', 'token_mean_nll')
    lookup = {(r['training_seed'], r['trial_seed'], r['model']): r for r in rows}
    if len(lookup) != len(rows):
        raise ValueError('Duplicate model/trial result')
    if len(rows) != len(training_seeds) * len(trial_seeds) * 2:
        raise ValueError('Incomplete paired results')
    differences, per_seed = [], []
    for seed in training_seeds:
        paired = []
        for trial in trial_seeds:
            base, feat = (lookup[(seed, trial, name)] for name in ('m1', 'feature'))
            if any(base[k] != feat[k] for k in ('tokens', 'batches', 'examples')):
                raise ValueError('Paired counts differ')
            row = {'training_seed': seed, 'trial_seed': trial}
            row.update({f'delta_{k}': feat[k] - base[k] for k in metrics})
            differences.append(row)
            paired.append(row)
        summary = {'training_seed': seed, 'trials': len(trial_seeds)}
        for k in metrics:
            summary[f'mean_delta_{k}'] = statistics.mean(r[f'delta_{k}'] for r in paired)
        per_seed.append(summary)
    aggregate = {}
    for k in metrics:
        values = [r[f'mean_delta_{k}'] for r in per_seed]
        aggregate[k] = {'mean_paired_difference': statistics.mean(values),
                        'sample_sd_across_training_seeds': statistics.stdev(values) if len(values) > 1 else None,
                        'training_seeds_favoring_feature': sum(x < 0 for x in values),
                        'training_seed_count': len(values)}
    return differences, per_seed, aggregate


def write_csv(path, rows):
    with Path(path).open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--teacher-encoder', type=Path, required=True)
    parser.add_argument('--teacher-decoder', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, default=Path('results/article_kd/diagnostics'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--training-seeds', nargs='+', type=int, default=[42, 43, 44, 45])
    parser.add_argument('--trial-seeds', nargs='+', type=int, default=[1001, 1002, 1003, 1004, 1005])
    parser.add_argument('--m1-template', default='m1-rayleigh-adam-wd0-10epochs-seed{seed}')
    parser.add_argument('--feature-template', default='ablation-ce1-feature-wd0-10epochs-seed{seed}')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--max-batches', type=int, help='SMOKE TEST ONLY; output marked partial')
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    for seeds in (args.training_seeds, args.trial_seeds):
        if len(set(seeds)) != len(seeds) or any(x < 0 or x >= 2**32 for x in seeds):
            parser.error('Seeds must be unique integers in [0, 2**32)')
    if args.max_batches is not None and args.max_batches < 1:
        parser.error('--max-batches must be positive')
    return args


def main():
    args = parse_args()
    entries, reference, data_manifest, vocab_path = preflight(args)
    print(f'Artifact checks passed for {len(entries)} checkpoints.', flush=True)
    if args.preflight_only:
        return
    import torch
    from torch.utils.data import DataLoader
    from dataset_multilingual import EurParallelDatasetBPE, collate_parallelBPE
    from models.transceiver import DeepSC
    from models.tx_model import Transmitter
    from student import Student
    from utils.model_utils import create_masks, snr_to_noise

    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available()
                          else 'cpu' if args.device == 'auto' else args.device)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    # Must be a fresh directory. No existing runs or outputs are overwritten.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    arch = reference['architecture']
    vocab = json.loads(vocab_path.read_text())
    if vocab['vocab_size'] != arch['vocab_size']:
        raise ValueError('Vocabulary/model size mismatch')
    pad = vocab['token_to_idx']['<PAD>']
    dataset = EurParallelDatasetBPE('en_en', 'val', str(args.data_dir))
    expected_count = data_manifest['record_counts']['accepted_after_filtering']['en_en']['val']
    if len(dataset) != expected_count:
        raise ValueError('Validation count differs from manifest')
    for source, target in dataset.data:
        if source != target or not 3 <= len(source) <= arch['max_length']:
            raise ValueError('Invalid validation reconstruction pair')
        if source[:2] != [vocab['token_to_idx']['<START>'], vocab['token_to_idx']['<EN>']] or source[-1] != vocab['token_to_idx']['<END>']:
            raise ValueError('Invalid validation framing')
        if any(type(token) is not int or not 0 <= token < arch['vocab_size'] for token in source):
            raise ValueError('Invalid validation token')
    # A separate generator keeps DataLoader bookkeeping out of the channel stream.
    loader = DataLoader(dataset, batch_size=reference['batch_size'], shuffle=False,
                        num_workers=0, collate_fn=collate_parallelBPE,
                        generator=torch.Generator().manual_seed(0))
    teacher = DeepSC(arch['teacher_layers'], arch['vocab_size'], arch['vocab_size'],
                     arch['max_length'], arch['max_length'], arch['d_model'],
                     arch['teacher_heads'], arch['dff'], reference['cli_arguments']['dropout'])
    enc = torch.load(args.teacher_encoder, map_location='cpu', weights_only=True)
    expected_arch = {'num_layers': arch['teacher_layers'], 'num_heads': arch['teacher_heads'],
                     'd_model': arch['d_model'], 'dff': arch['dff'],
                     'max_len': arch['max_length'], 'vocab_size': arch['vocab_size']}
    recorded_arch = enc.get('meta', {}).get('architecture', {})
    if any(recorded_arch.get(k) != v for k, v in expected_arch.items()):
        raise ValueError('Teacher encoder architecture metadata mismatch')
    teacher.encoder.load_state_dict(enc['encoder'], strict=True)
    teacher.channel_encoder.load_state_dict(enc['channel_encoder'], strict=True)
    transmitter = Transmitter(teacher.encoder, teacher.channel_encoder).to(device)
    del teacher, enc
    models = {}
    for entry in entries:
        model = Student(arch['student_layers'], arch['vocab_size'], arch['vocab_size'],
                        arch['max_length'], arch['max_length'], arch['d_model'],
                        arch['student_heads'], arch['dff'], reference['cli_arguments']['dropout'])
        checkpoint = torch.load(entry['checkpoint'], map_location='cpu', weights_only=True)
        if checkpoint.get('meta', {}).get('epoch') != entry['manifest']['selected_checkpoint']['epoch']:
            raise ValueError('Student checkpoint epoch metadata mismatch')
        for component in ('channel_decoder', 'decoder', 'dense'):
            getattr(model, component).load_state_dict(checkpoint[component], strict=True)
        models[(entry['seed'], entry['label'])] = model.to(device)
        del checkpoint
    provenance = {'created_utc': datetime.now(timezone.utc).isoformat(),
                  'protocol': 'validation only, teacher forcing, one shared channel draw per batch',
                  'partial': args.max_batches is not None,
                  'max_batches': args.max_batches, 'training_seeds': args.training_seeds,
                  'trial_seeds': args.trial_seeds, 'snr_db': 8.0, 'batch_size': reference['batch_size'],
                  'label_smoothing': 0.1, 'difference_direction': 'feature minus m1; negative favors feature',
                  'runtime': {'pytorch': torch.__version__, 'device': str(device),
                              'cuda_runtime': torch.version.cuda,
                              'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None},
                  'evaluator': identity(__file__),
                  'implementation_files': {str(p): identity(p) for p in map(Path, [
                      'student.py', 'models/transceiver.py', 'models/tx_model.py',
                      'utils/model_utils.py', 'dataset_multilingual.py'])},
                  'inputs': [{**{k: v for k, v in e.items() if k != 'manifest'},
                              'manifest_identity': identity(e['manifest_path']),
                              'recorded_manifest': e['manifest']} for e in entries]}
    try:
        provenance['evaluation_git_commit'] = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
        provenance['evaluation_git_status'] = subprocess.check_output(
            ['git', 'status', '--short'], text=True, stderr=subprocess.DEVNULL).splitlines()
    except (OSError, subprocess.CalledProcessError):
        provenance['evaluation_git_commit'] = None
    (args.output_dir / 'evaluation_manifest.json').write_text(json.dumps(provenance, indent=2))
    rows = []
    for trial in args.trial_seeds:
        results = run_trial(torch, transmitter, models, loader, trial, snr_to_noise(8.0),
                            device, pad, create_masks, args.max_batches)
        trial_rows = [{'training_seed': seed, 'trial_seed': trial, 'model': label, **metrics}
                      for (seed, label), metrics in results.items()]
        write_csv(args.output_dir / f'trial_{trial}.csv', trial_rows)
        rows.extend(trial_rows)
    differences, per_seed, aggregate = paired_summary(rows, args.training_seeds, args.trial_seeds)
    write_csv(args.output_dir / 'trial_metrics.csv', rows)
    write_csv(args.output_dir / 'paired_differences.csv', differences)
    write_csv(args.output_dir / 'per_seed_summary.csv', per_seed)
    summary = {'partial': args.max_batches is not None, 'metrics': aggregate,
               'inference_unit': 'training seed, after averaging channel trials within seed',
               'caution': 'Descriptive validation comparison of fixed selected checkpoints; not an independent test result or automatic significance claim.'}
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(f'Outputs: {args.output_dir.resolve()}', flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()

