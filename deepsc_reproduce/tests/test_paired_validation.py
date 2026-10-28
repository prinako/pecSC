import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate_paired_validation import identity, paired_summary, preflight, signature, verify


class PairingTests(unittest.TestCase):
    def rows(self):
        return [dict(training_seed=seed, trial_seed=trial, model=model,
                     batch_mean_ce=value, token_mean_ce=value,
                     token_mean_nll=value, tokens=3, batches=1, examples=2)
                for seed in [42, 43] for trial in [1001, 1002]
                for model, value in [('m1', 2.0), ('feature', 1.9 if seed == 42 else 2.2)]]

    def test_average_trials_within_seed_before_seed_sd(self):
        differences, seeds, result = paired_summary(self.rows(), [42, 43], [1001, 1002])
        self.assertEqual(len(differences), 4)
        self.assertEqual(len(seeds), 2)
        metric = result['token_mean_ce']
        self.assertAlmostEqual(metric['mean_paired_difference'], 0.05)
        self.assertAlmostEqual(metric['sample_sd_across_training_seeds'], 0.2121320343559643)
        self.assertEqual(metric['training_seeds_favoring_feature'], 1)

    def test_duplicate_missing_and_count_mismatch_rejected(self):
        rows = self.rows()
        for broken in [rows[:-1], rows + [rows[0]]]:
            with self.assertRaises(ValueError):
                paired_summary(broken, [42, 43], [1001, 1002])
        rows[1]['tokens'] = 4
        with self.assertRaises(ValueError):
            paired_summary(rows, [42, 43], [1001, 1002])

    def test_modified_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'checkpoint.pth'
            path.write_bytes(b'original')
            expected = identity(path)
            path.write_bytes(b'modified')
            with self.assertRaises(ValueError):
                verify(path, expected)


class PreflightTests(unittest.TestCase):
    def fixture(self, root):
        data = root / 'data'
        data.mkdir()
        metadata = {'duplicate_statistics': {'cross_split_overlaps': {
            'train_val': 0, 'train_test': 0, 'val_test': 0}}}
        files = {}
        for key, name in [('val', 'val_en_en.pkl'), ('dataset_manifest', 'dataset_manifest.json'),
                          ('tokenizer_model', 'tokenizer_bpe.model'), ('split_ids', 'split_ids.json')]:
            path = data / name
            path.write_text(json.dumps(metadata) if key == 'dataset_manifest' else key)
            files[key] = identity(path)
        vocab = data / 'vocab_bpe.json'
        vocab.write_text('{}')
        enc, dec = root / 'enc.pth', root / 'dec.pth'
        enc.write_bytes(b'encoder'); dec.write_bytes(b'decoder')
        manifest = {
            'seed': 42, 'architecture': {'student_initialization': 'fresh seeded initialization'},
            'dataset': files, 'tokenizer_vocab': identity(vocab),
            'teacher_checkpoints': {'encoder': identity(enc), 'decoder': identity(dec)},
            'optimizer': {'weight_decay': 0}, 'channel': {'model': 'Rayleigh'},
            'batch_size': 32, 'epoch_budget': 10,
            'loss': {'alpha': 1.0, 'beta': 0.0, 'gamma': 0.0, 'label_smoothing': 0.1},
            'cli_arguments': {'seed': 42, 'dropout': 0.1, 'grad_clip': 1,
                              'init_student_from_teacher': False},
            'checkpoint_selection_rule': 'minimum validation cross-entropy',
        }
        for label in ['m1', 'feature']:
            directory = root / f'{label}-42'
            directory.mkdir()
            checkpoint = directory / 'student_best.pth'
            checkpoint.write_bytes(label.encode())
            m = copy.deepcopy(manifest)
            m['loss']['gamma'] = 0.1 if label == 'feature' else 0.0
            m['selected_checkpoint'] = {**identity(checkpoint), 'epoch': 9,
                'selection_metric': 'validation_cross_entropy',
                'path': '/old/machine/student_best.pth'}
            (directory / 'manifest.json').write_text(json.dumps(m))
        return SimpleNamespace(training_seeds=[42], run_root=root, data_dir=data,
            m1_template='m1-{seed}', feature_template='feature-{seed}',
            teacher_encoder=enc, teacher_decoder=dec)

    def test_relocated_paths_and_no_train_or_test_files_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            entries, *_ = preflight(args)
            self.assertEqual(len(entries), 2)
            self.assertFalse((args.data_dir / 'test_en_en.pkl').exists())

    def test_mismatched_optimizer_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            path = args.run_root / 'feature-42/manifest.json'
            m = json.loads(path.read_text()); m['optimizer']['weight_decay'] = 0.0005
            path.write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError, 'Incompatible'):
                preflight(args)

    def test_incorrect_arm_and_seed_rejected(self):
        for change in ['seed', 'loss']:
            with tempfile.TemporaryDirectory() as tmp:
                args = self.fixture(Path(tmp))
                path = args.run_root / 'feature-42/manifest.json'
                m = json.loads(path.read_text())
                if change == 'seed': m['seed'] = 43
                else: m['loss']['beta'] = 0.3
                path.write_text(json.dumps(m))
                with self.assertRaises(ValueError): preflight(args)


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch is required for numerical tests')
class NumericalPairingTests(unittest.TestCase):
    def test_shared_draws_repeatability_and_unchanged_weights(self):
        import torch
        from evaluate_paired_validation import run_trial

        class TX(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.calls = 0
            def forward(self, src, mask, channel, std):
                self.calls += 1
                return None, None, None, torch.randn(src.shape[0], src.shape[1]-1, 4)

        seen = []
        class RX(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.linear = torch.nn.Linear(4, 5)
            def forward(self, z, prefix, causal, source):
                seen.append(z.data_ptr())
                return self.linear(z), None, None

        a, b, tx = RX(), RX(), TX()
        b.load_state_dict(a.state_dict())
        before = {k: v.clone() for k, v in a.state_dict().items()}
        src = torch.tensor([[1, 4, 2], [1, 2, 0]])
        loader = [(src, src), (src[:1], src[:1])]
        masks = lambda *args: (None, None)
        models = {(42, 'm1'): a, (42, 'feature'): b}
        first = run_trial(torch, tx, models, loader, 1001, 0.2, 'cpu', 0, masks)
        self.assertEqual(tx.calls, 2)
        self.assertEqual(seen[0], seen[1]); self.assertEqual(seen[2], seen[3])
        self.assertEqual(first[(42, 'm1')], first[(42, 'feature')])
        again = run_trial(torch, tx, models, loader, 1001, 0.2, 'cpu', 0, masks)
        self.assertEqual(first, again)
        self.assertEqual(first[(42, 'm1')]['tokens'], 5)
        for key, value in a.state_dict().items(): self.assertTrue(torch.equal(before[key], value))
        self.assertTrue(all(p.grad is None for p in a.parameters()))


if __name__ == '__main__':
    unittest.main()

