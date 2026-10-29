"""Run independent experiments on separate Kaggle GPUs, never laptop workers."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import time

from .contracts import require_kaggle_gpu,write_json


def recipes(suite):
    if suite=='main':
        return [('control-p1','P1','L3','M4','C1',12,-1),
                ('pathology-p1','P1','L3','M4','C4',12,-1)]
    if suite in ('ablation','ablation-a','ablation-b'):
        # Paired controls share seed, initialization, split and SIX-epoch schedule.
        # Two six-model jobs bound per-notebook duration without shortening controls.
        result = [('abl-m1','P1','L3','M1','C1',6,-1),('abl-m2','P1','L3','M2','C1',6,-1),
                  ('abl-m3','P1','L3','M3','C1',6,-1),('abl-m4','P1','L3','M4','C1',6,-1)]
        result += [(f'abl-c{k}','P1','L3','M4',f'C{k}',6,-1) for k in (2,3,4)]
        result += [('abl-c5','P1','L2','M4','C5',6,-1)]
        result += [(f'abl-l{k}','P1',f'L{k}','M4','C4',6,-1) for k in (1,4)]
        result += [('abl-p2','P2','L3','M4','C4',6,-1),('abl-p3','P3','L3','M4','C4',6,-1)]
        return result[:6] if suite=='ablation-a' else result[6:] if suite=='ablation-b' else result
    if suite=='teachers':
        return [(f'teacher-fold-{f}','P1','L3','M4','C4',6,f) for f in range(3)]
    if suite=='high-resolution':
        return [('pathology-p3','P3','L3','M4','C4',12,-1)]
    if suite=='warm':
        return [('warm-control-p3','P3','L3','M4','C1',12,-1),
                ('warm-pathology-p3','P3','L3','M4','C4',12,-1)]
    raise ValueError(suite)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite',choices=['main','ablation','ablation-a','ablation-b','teachers','high-resolution','warm'],required=True)
    args = parser.parse_args()
    require_kaggle_gpu()
    import torch
    devices = min(torch.cuda.device_count(),2)
    jobs = recipes(args.suite)
    started = time.time()
    receipt = {'suite':args.suite,'devices':devices,'jobs':[],'gpu_location':'Kaggle',
               'all_training_on_kaggle':True,'complete':False}
    if args.suite=='main':
        wheelhouse = Path('/kaggle/working/offline_wheels')
        wheelhouse.mkdir(exist_ok=True)
        subprocess.run([sys.executable,'-m','pip','download','--no-deps','--only-binary=:all:',
                        '--dest',str(wheelhouse),'pydicom==3.0.1','pylibjpeg==2.1.0',
                        'pylibjpeg-libjpeg','pylibjpeg-openjpeg','python-gdcm','timm==1.0.29'],check=True)
    def worker(device):
        completed = []
        for name,configuration,policy,attention,branches,epochs,fold in jobs[device::devices]:
            command = [sys.executable,'-u','-m','independent_mri.train','--name',name,
                       '--configuration',configuration,'--policy',policy,'--attention',attention,
                       '--branches',branches,'--epochs',str(epochs),'--device',str(device),
                       '--teacher-fold',str(fold)]
            if args.suite=='warm':
                command += ['--initialization','public_s6']
            print(json.dumps({'launch':name,'device':device,'command':command}),flush=True)
            result = subprocess.run(command,check=False)
            record = {'name':name,'device':device,'returncode':result.returncode}
            completed.append(record)
            if result.returncode!=0:
                # Integrity/gradient failures cannot be quietly converted to a finished experiment.
                break
        return completed
    with ThreadPoolExecutor(max_workers=devices) as pool:
        for completed in pool.map(worker,range(devices)):
            receipt['jobs'].extend(completed)
            write_json('/kaggle/working/suite_receipt.json',receipt)
    receipt['complete'] = len(receipt['jobs'])==len(jobs) and all(r['returncode']==0 for r in receipt['jobs'])
    receipt['elapsed_seconds'] = time.time()-started
    write_json('/kaggle/working/suite_receipt.json',receipt)
    if not receipt['complete']:
        raise RuntimeError('Independent training suite did not complete; inspect individual receipts')


if __name__=='__main__': main()
