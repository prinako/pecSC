"""Repeat DeepSC diagnostics/comparisons without changing the training code."""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
SOURCE_COMMIT = '670f7fb5a5b8b060aa6647aa6307ae1cee647989'
VARIANTS = {
    'reference': (1.0, 0.0, 0.0, 0.0005, 'reference-m1-adam-wd5e4-{epochs}epochs-seed{seed}'),
    'm1': (1.0, 0.0, 0.0, 0.0, 'm1-rayleigh-adam-wd0-{epochs}epochs-seed{seed}'),
    'm2': (0.6, 0.3, 0.1, 0.0, 'm2-rayleigh-adam-wd0-{epochs}epochs-seed{seed}'),
    'feature': (1.0, 0.0, 0.1, 0.0, 'ablation-ce1-feature-wd0-{epochs}epochs-seed{seed}'),
    'logit': (1.0, 0.3, 0.0, 0.0, 'ablation-ce1-logit-wd0-{epochs}epochs-seed{seed}'),
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def dump(path, obj):
    with Path(path).open('x') as f:
        json.dump(obj, f, indent=2, allow_nan=False)


def csv_out(path, rows):
    if not rows:
        return
    with Path(path).open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def git_state(repo):
    def get(*args):
        try:
            return subprocess.check_output(['git', *args], cwd=repo, text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    return {'commit': get('rev-parse', 'HEAD'), 'status': get('status', '--short')}


def read_run(directory):
    directory = Path(directory)
    manifest_path = directory / 'manifest.json'
    csv_path = directory / 'training.csv'
    manifest = json.loads(manifest_path.read_text())
    with csv_path.open(newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f'Empty training CSV: {csv_path}')
    epochs = [int(r['epoch']) for r in rows]
    if epochs != list(range(1, len(rows) + 1)):
        raise ValueError(f'Missing, duplicated or unordered epochs: {csv_path}')
    for r in rows:
        for key in ('ce', 'val_ce', 'loss', 'val_loss', 'kd', 'tf', 'time'):
            if not math.isfinite(float(r[key])):
                raise ValueError(f'Nonfinite {key}: {csv_path}')
        for key in ('alpha', 'beta', 'gamma', 'temperature'):
            if float(r[key]) != float(manifest['loss'][key]):
                raise ValueError(f'CSV/manifest {key} mismatch: {csv_path}')
    best = min(rows, key=lambda r: float(r['val_ce']))
    selected = manifest['selected_checkpoint']
    if selected['selection_metric'] != 'validation_cross_entropy':
        raise ValueError(f'Unexpected checkpoint selection metric: {directory}')
    if selected['epoch'] != int(best['epoch']) or not math.isclose(
            selected['selection_value'], float(best['val_ce']), abs_tol=1e-10, rel_tol=1e-10):
        raise ValueError(f'Selected checkpoint does not match CSV minimum: {directory}')
    return manifest, rows, best


def control_signature(m):
    return {
        'architecture': m['architecture'],
        'dataset': {k: v['sha256'] for k, v in m['dataset'].items()},
        'vocab': m['tokenizer_vocab']['sha256'],
        'teacher': {k: v['sha256'] for k, v in m['teacher_checkpoints'].items()},
        'optimizer': m['optimizer'], 'channel': m['channel'],
        'batch_size': m['batch_size'], 'epoch_budget': m['epoch_budget'],
        'dropout': m['cli_arguments']['dropout'],
        'grad_clip': m['cli_arguments']['grad_clip'],
        'initialization': m['cli_arguments']['init_student_from_teacher'],
        'smoothing': m['loss']['label_smoothing'],
    }


def compare(args):
    root, out = args.run_root, args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    summary, data, identities, epoch_rows = [], {}, [], []
    for directory in sorted(root.iterdir()):
        if not directory.is_dir() or not (directory / 'training.csv').exists():
            continue
        if not (directory / 'manifest.json').exists():
            raise ValueError(f'Missing manifest: {directory}')
        m, rows, best = read_run(directory)
        weights = tuple(float(m['loss'][k]) for k in ('alpha', 'beta', 'gamma'))
        decay = float(m['optimizer']['weight_decay'])
        variant = next((name for name, config in VARIANTS.items()
                        if config[:3] == weights and config[3] == decay), 'other')
        row = {'run': directory.name, 'variant': variant, 'seed': m['seed'],
               'epochs_recorded': len(rows), 'epoch_budget': m['epoch_budget'],
               'complete': len(rows) == m['epoch_budget'], 'best_epoch': int(best['epoch']),
               'best_val_ce': float(best['val_ce']), 'final_train_ce': float(rows[-1]['ce']),
               'seconds_per_epoch': statistics.mean(float(r['time']) for r in rows),
               'alpha': weights[0], 'beta': weights[1], 'gamma': weights[2], 'weight_decay': decay}
        summary.append(row); data[directory.name] = (m, row)
        for r in rows:
            epoch_rows.append({'run':directory.name,'variant':variant,'seed':m['seed'],
                'epoch':int(r['epoch']), **{key:float(r[key]) for key in
                    ('ce','val_ce','loss','val_loss','kd','tf','time')}})
        identities.append({'run': directory.name, 'manifest_sha256': sha(directory / 'manifest.json'),
                           'training_csv_sha256': sha(directory / 'training.csv')})
    if not summary:
        raise ValueError('No training CSV/manifest pairs found')
    pairs, missing = [], []
    for seed in args.seeds:
        base_name = VARIANTS['m1'][4].format(seed=seed, epochs=args.epochs)
        if base_name not in data:
            missing.append(base_name); continue
        base_m, base = data[base_name]
        for variant in ('m2', 'feature', 'logit'):
            name = VARIANTS[variant][4].format(seed=seed, epochs=args.epochs)
            if name not in data:
                missing.append(name); continue
            m, row = data[name]
            if control_signature(m) != control_signature(base_m):
                raise ValueError(f'Unmatched controls: {base_name} versus {name}')
            if not (row['complete'] and base['complete']):
                raise ValueError(f'Incomplete matched run: {name}')
            delta = row['best_val_ce'] - base['best_val_ce']
            pairs.append({'seed': seed, 'variant': variant, 'm1_best_epoch': base['best_epoch'],
                          'candidate_best_epoch': row['best_epoch'],
                          'm1_best_val_ce': base['best_val_ce'],
                          'candidate_best_val_ce': row['best_val_ce'],
                          'delta_val_ce': delta,
                          'relative_change_percent': 100 * delta / base['best_val_ce']})
    aggregates = []
    for variant in ('m2', 'feature', 'logit'):
        group = [r for r in pairs if r['variant'] == variant]
        if not group: continue
        vals = [r['candidate_best_val_ce'] for r in group]
        baselines = [r['m1_best_val_ce'] for r in group]
        aggregates.append({'variant': variant, 'seeds': [r['seed'] for r in group],
                           'n': len(group), 'candidate_mean': statistics.mean(vals),
                           'candidate_sample_sd': statistics.stdev(vals) if len(vals)>1 else None,
                           'matched_m1_mean': statistics.mean(baselines),
                           'mean_delta': statistics.mean(r['delta_val_ce'] for r in group),
                           'relative_mean_change_percent': 100*(statistics.mean(vals)/statistics.mean(baselines)-1),
                           'seeds_favoring_candidate': sum(r['delta_val_ce'] < 0 for r in group)})
    csv_out(out/'runs.csv', summary); csv_out(out/'matched_comparisons.csv', pairs)
    csv_out(out/'epoch_metrics.csv', epoch_rows)
    dump(out/'comparison_summary.json', {'groups': aggregates, 'missing_expected_runs': missing,
          'metric': 'minimum epoch batch-mean smoothed validation CE; selected-checkpoint evidence only',
          'inputs': identities, 'suite_sha256': sha(__file__)})
    print(json.dumps(aggregates, indent=2), flush=True)
    if args.paired_dir:
        from evaluate_paired_validation import paired_summary
        manifest = json.loads((args.paired_dir/'evaluation_manifest.json').read_text())
        if manifest.get('partial') is not False:
            raise ValueError('Refusing to treat partial/smoke evaluation as full evidence')
        with (args.paired_dir/'trial_metrics.csv').open(newline='') as f:
            trial_rows = list(csv.DictReader(f))
        for row in trial_rows:
            for k in ('training_seed', 'trial_seed', 'tokens', 'batches', 'examples'):
                row[k] = int(row[k])
            for k in ('batch_mean_ce', 'token_mean_ce', 'token_mean_nll'):
                row[k] = float(row[k])
                if not math.isfinite(row[k]): raise ValueError('Nonfinite paired metric')
        d, per_seed, a = paired_summary(trial_rows, manifest['training_seeds'], manifest['trial_seeds'])
        csv_out(out/'paired_differences.csv', d); csv_out(out/'per_seed_summary.csv', per_seed)
        dump(out/'paired_summary.json', {'metrics': a, 'independent_unit': 'training seed',
             'input_csv_sha256': sha(args.paired_dir/'trial_metrics.csv'),
             'evaluation_manifest_sha256': sha(args.paired_dir/'evaluation_manifest.json')})
        print('Paired metrics:', json.dumps(a, indent=2), flush=True)
    print(f'Reports: {out.resolve()}')


def diagnostics(args):
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    output, inputs = [], []
    files = [args.input] if args.input.is_file() else sorted(args.input.rglob('diagnostics.csv'))
    if not files: raise ValueError('No diagnostics.csv files found; excluded Git files must exist locally')
    for path in files:
        groups = {}
        with path.open(newline='') as f:
            for row in csv.DictReader(f):
                epoch = int(row['epoch'])
                a = groups.setdefault(epoch, {'n': 0, 'ce': 0., 'grad': 0., 'clips': 0,
                    'grad_max': 0., 'nonfinite': 0, 'deep_n': 0, 'deep_ce': 0.,
                    'other_n': 0, 'other_ce': 0., 'param_first': float(row['parameter_norm']),
                    'param_last': 0., 'last_batch': 0})
                batch = int(row['batch_index'])
                if batch != a['last_batch'] + 1: raise ValueError(f'Batch sequence error: {path}')
                ce, grad, param = (float(row[k]) for k in ('ce','gradient_norm_before_clipping','parameter_norm'))
                if not all(map(math.isfinite, (ce, grad, param))):
                    raise ValueError(f'Nonfinite diagnostic scalar: {path}, epoch {epoch}, batch {batch}')
                a['n']+=1; a['ce']+=ce; a['grad']+=grad; a['clips']+=grad>args.clip
                a['grad_max']=max(a['grad_max'],grad)
                a['nonfinite']+=row['has_nan'].lower()=='true' or row['has_inf'].lower()=='true'
                a['param_last']=param; a['last_batch']=batch
                if row.get('rayleigh_magnitude') not in ('', None):
                    prefix = 'deep' if float(row['rayleigh_magnitude']) < 0.1 else 'other'
                    a[prefix+'_n']+=1; a[prefix+'_ce']+=ce
        for epoch,a in sorted(groups.items()):
            output.append({'file': str(path.resolve()), 'epoch':epoch,'batches':a['n'],
                'ce':a['ce']/a['n'],'gradient_mean':a['grad']/a['n'],'gradient_max':a['grad_max'],
                'clipping_percent':100*a['clips']/a['n'],'parameter_first':a['param_first'],
                'parameter_last_before_step':a['param_last'],'nonfinite_flagged_batches':a['nonfinite'],
                'deep_fade_batches':a['deep_n'],
                'deep_fade_ce':a['deep_ce']/a['deep_n'] if a['deep_n'] else '',
                'other_ce':a['other_ce']/a['other_n'] if a['other_n'] else ''})
        inputs.append({'path':str(path.resolve()),'sha256':sha(path)})
    csv_out(out/'diagnostics_by_epoch.csv',output)
    dump(out/'diagnostic_inputs.json',{'inputs':inputs,'gradient_threshold':args.clip,
         'note':'Scalar means by batch. Recorded parameter norms precede optimizer.step. Partial epochs are not inferred complete.'})
    print(f'Analyzed {len(files)} diagnostic files: {out.resolve()}')


def training_plan(args):
    script = args.repo/'train_multi_vocab_one_student.py'
    if not script.is_file(): raise ValueError('Run --repo with your pecSC root')
    jobs=[]
    for seed in args.seeds:
        for variant in args.variants:
            alpha,beta,gamma,decay,template=VARIANTS[variant]
            run_id=template.format(seed=seed,epochs=args.epochs)
            command=['uv','run','python','-u',str(script),
                '--data-dir',str(args.repo/args.data_dir),'--vocab-file','vocab_bpe.json',
                '--teacher-encoder',str(args.repo/args.teacher_encoder),
                '--teacher-decoder',str(args.repo/args.teacher_decoder),
                '--save-dir',str(args.run_root),'--experiment-id',run_id,
                '--seed',str(seed),'--channel','Rayleigh','--snr-mode','range',
                '--snr-db-low','0','--snr-db-high','20','--val-snr-db','8',
                '--max-len','67','--d-model','128','--dff','512',
                '--num-layers','8','--num-heads','16','--dropout','0.1',
                '--batch-size','32','--num-workers','0','--epochs',str(args.epochs),
                '--lr','1e-4','--weight-decay',str(decay),'--grad-clip','1',
                '--alpha',str(alpha),'--beta',str(beta),'--gamma',str(gamma),
                '--temperature','2','--diagnostic-mode']
            jobs.append({'variant':variant,'seed':seed,'run_id':run_id,'command':command})
    return jobs


def train(args):
    jobs=training_plan(args)
    print(json.dumps({'runs':len(jobs),'epochs_per_run':args.epochs,'jobs':jobs},indent=2),flush=True)
    if not args.execute:
        print('PLAN ONLY. Add --execute to start fresh training.'); return
    # Check all destinations before launching even the first job.
    conflicts=[j['run_id'] for j in jobs if (args.run_root/j['run_id']).exists()]
    if conflicts: raise FileExistsError(f'Existing run(s); no overwrite/resume: {conflicts}')
    args.run_root.mkdir(parents=True,exist_ok=True)
    dump(args.run_root/'suite_plan.json',{'source':git_state(args.repo),
          'suite_sha256':sha(__file__),
          'implementation_sha256': {p:sha(args.repo/p) for p in [
              'train_multi_vocab_one_student.py','student.py','models/transceiver.py',
              'models/tx_model.py','utils/model_utils.py','utils/kd_utils.py',
              'utils/publication_data.py','dataset_multilingual.py']},
          'jobs':jobs,'created_utc':datetime.now(timezone.utc).isoformat()})
    for j in jobs:
        log=args.run_root/(j['run_id']+'.log.gz')
        print('Starting',j['run_id'],'compressed log:',log,flush=True)
        with gzip.open(log,'xb') as f:
            process=subprocess.Popen(j['command'],cwd=args.repo,stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            try:
                while True:
                    block=process.stdout.read(65536)
                    if not block:break
                    f.write(block)
                code=process.wait()
            except BaseException:
                process.terminate(); process.wait(); raise
        if code: raise RuntimeError(f'{j["run_id"]} failed ({code}); stopping sequence. Read {log}')
        print('Completed',j['run_id'],flush=True)


def paired(args):
    script=HERE/'evaluate_paired_validation.py'
    command=['uv','run','python','-u',str(script),'--data-dir',str(args.repo/args.data_dir),
        '--teacher-encoder',str(args.repo/args.teacher_encoder),
        '--teacher-decoder',str(args.repo/args.teacher_decoder),
        '--run-root',str(args.run_root),'--output-dir',str(args.output_dir),
        '--training-seeds',*map(str,args.seeds),'--trial-seeds',*map(str,args.trial_seeds)]
    if args.preflight_only:command.append('--preflight-only')
    if args.max_batches:command+=['--max-batches',str(args.max_batches)]
    env=os.environ.copy();env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    env['PYTHONPATH']=str(args.repo)+os.pathsep+env.get('PYTHONPATH','')
    subprocess.run(command,cwd=args.repo,env=env,check=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=Path.cwd())
    subs=parser.add_subparsers(dest='action',required=True)
    p=subs.add_parser('compare',help='Regenerate tables from saved CSVs/manifests; no GPU')
    p.add_argument('--run-root',type=Path,default=Path('results/article_kd/diagnostics'))
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--paired-dir',type=Path)
    p.add_argument('--seeds',nargs='+',type=int,default=[42,43,44,45]);p.add_argument('--epochs',type=int,default=10)
    p.set_defaults(func=compare)
    p=subs.add_parser('diagnostics',help='Stream raw per-batch CSVs into epoch summaries')
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--clip',type=float,default=1.0);p.set_defaults(func=diagnostics)
    for action in ('train','paired'):
        p=subs.add_parser(action)
        p.add_argument('--run-root',type=Path,required=True)
        p.add_argument('--seeds',nargs='+',type=int,default=[42,43,44,45])
        p.add_argument('--data-dir',type=Path,default=Path('data/train/europarl_bpe/2026-publication-v1'))
        p.add_argument('--teacher-encoder',type=Path,default=Path('results/article_kd/training/teacher-bpe32k-seed42/encoder_38.pth'))
        p.add_argument('--teacher-decoder',type=Path,default=Path('results/article_kd/training/teacher-bpe32k-seed42/decoder_38.pth'))
        if action=='train':
            p.add_argument('--variants',nargs='+',choices=VARIANTS,default=['m1','m2','feature','logit'])
            p.add_argument('--epochs',type=int,default=10);p.add_argument('--execute',action='store_true')
            p.set_defaults(func=train)
        else:
            p.add_argument('--trial-seeds',nargs='+',type=int,default=[1001,1002,1003,1004,1005])
            p.add_argument('--output-dir',type=Path,required=True)
            p.add_argument('--max-batches',type=int);p.add_argument('--preflight-only',action='store_true')
            p.set_defaults(func=paired)
    p=subs.add_parser('tests',help='Run repository tests using the project environment')
    p.set_defaults(func=lambda args:subprocess.run(['uv','run','pytest','-q',
        'tests/test_publication_pipeline.py','tests/test_student_diagnostics.py',
        'tests/test_paired_validation.py'],cwd=args.repo,check=True))
    p=subs.add_parser('channel-check',help='Repeat the scalar Rayleigh Monte Carlo probe')
    p.add_argument('--samples',type=int,default=100000)
    p.set_defaults(func=lambda args:subprocess.run(['uv','run','python',str(HERE/'channel_probe.py'),
        '--samples',str(args.samples)],cwd=args.repo,check=True))
    args=parser.parse_args();args.repo=args.repo.expanduser().resolve()
    for key in ('run_root','output_dir','paired_dir','input'):
        value=getattr(args,key,None)
        if value is not None:setattr(args,key,(args.repo/value).resolve())
    if hasattr(args,'seeds') and (len(set(args.seeds))!=len(args.seeds) or any(x<0 or x>=2**32 for x in args.seeds)):
        parser.error('Training seeds must be unique integers in [0, 2**32)')
    if hasattr(args,'epochs') and args.epochs<1:parser.error('--epochs must be positive')
    if hasattr(args,'variants') and len(set(args.variants))!=len(args.variants):parser.error('Variants must be unique')
    if getattr(args,'max_batches',None) is not None and args.max_batches<1:parser.error('--max-batches must be positive')
    args.func(args)


if __name__=='__main__':main()
