import copy
import csv
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reproduce import VARIANTS, control_signature, diagnostics, read_run, training_plan


class ReproductionTests(unittest.TestCase):
    def fixture(self,path):
        manifest={'seed':42,'loss':{'alpha':1,'beta':0,'gamma':0,'temperature':2},
            'selected_checkpoint':{'epoch':2,'selection_value':2.1,
                                   'selection_metric':'validation_cross_entropy'}}
        rows=[dict(epoch=e,ce=ce,val_ce=val,loss=ce,val_loss=val,kd=0,tf=0,time=10,
                   alpha=1,beta=0,gamma=0,temperature=2)
              for e,ce,val in [(1,3.,2.5),(2,2.3,2.1),(3,2.2,2.2)]]
        (path/'manifest.json').write_text(json.dumps(manifest))
        with (path/'training.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        return manifest

    def test_selection_uses_val_ce_not_final_epoch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root)
            _,rows,best=read_run(root)
            self.assertEqual(int(best['epoch']),2);self.assertEqual(len(rows),3)

    def test_manifest_csv_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);m=self.fixture(root)
            m['selected_checkpoint']['epoch']=3
            (root/'manifest.json').write_text(json.dumps(m))
            with self.assertRaises(ValueError):read_run(root)

    def test_missing_epoch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root)
            p=root/'training.csv';lines=p.read_text().splitlines()
            p.write_text('\n'.join([lines[0],lines[1],lines[3]])+'\n')
            with self.assertRaises(ValueError):read_run(root)

    def test_training_plan_all_arms_and_no_process_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'train_multi_vocab_one_student.py').write_text('')
            args=SimpleNamespace(repo=root,run_root=root/'fresh',data_dir=Path('data'),
                teacher_encoder=Path('encoder.pth'),teacher_decoder=Path('decoder.pth'),
                variants=list(VARIANTS),seeds=[42,43],epochs=10)
            jobs=training_plan(args);self.assertEqual(len(jobs),10)
            self.assertFalse((root/'fresh').exists())
            for job in jobs:
                command=job['command'];config=VARIANTS[job['variant']]
                for flag,value in zip(['--alpha','--beta','--gamma','--weight-decay'],config[:4]):
                    self.assertEqual(float(command[command.index(flag)+1]),value)
                self.assertNotIn('test_en_en.pkl',' '.join(command))

    def test_streaming_diagnostics_partial_counts_and_deep_fades(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);file=root/'diagnostics.csv'
            rows=[dict(epoch=1,batch_index=i,ce=ce,gradient_norm_before_clipping=g,
                parameter_norm=param,rayleigh_magnitude=h,has_nan='False',has_inf='False')
                for i,ce,g,param,h in [(1,4,2,10,0.05),(2,2,0.5,11,1)]]
            with file.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            diagnostics(SimpleNamespace(input=file,output_dir=root/'report',clip=1.0))
            with (root/'report/diagnostics_by_epoch.csv').open() as f:r=next(csv.DictReader(f))
            self.assertEqual(float(r['ce']),3);self.assertEqual(float(r['clipping_percent']),50)
            self.assertEqual(float(r['deep_fade_ce']),4);self.assertEqual(int(r['batches']),2)


if __name__=='__main__':unittest.main()
