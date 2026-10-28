"""Scalar Monte Carlo diagnostic only; does not prove training-loss causality."""
import argparse
import json
import torch

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--samples',type=int,default=100000)
args=p.parse_args()
if args.samples<1:p.error('--samples must be positive')
torch.manual_seed(42)
h=(torch.randn(args.samples)+1j*torch.randn(args.samples))/(2**0.5)
magnitude=h.abs(); amplification=1/magnitude.square()
report={'samples':args.samples,'seed':42,'torch':torch.__version__,
    'min_abs_h':magnitude.min().item(),
    'fraction_below_0_1':(magnitude<0.1).float().mean().item(),
    'fraction_below_0_05':(magnitude<0.05).float().mean().item(),
    'fraction_below_0_01':(magnitude<0.01).float().mean().item(),
    'power_amplification_max':amplification.max().item(),
    'power_amplification_p99':torch.quantile(amplification,0.99).item(),
    'noise_power_by_snr_db':{}}
for snr_db in [0,8,20]:
    values=amplification/(10**(snr_db/10))
    report['noise_power_by_snr_db'][snr_db]={'median':values.median().item(),
        'p99':torch.quantile(values,0.99).item(),'max':values.max().item()}
report['limitation']='Unit-power scalar channel simulation; not actual Student logits/losses/gradients.'
print(json.dumps(report,indent=2))
