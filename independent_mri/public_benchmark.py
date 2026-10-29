"""Frozen public SWA comparison on fixed development cohorts, Kaggle GPUs only."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from .contracts import TARGETS,UID,require_kaggle_gpu,sha256,write_json
from .evaluation import metrics
from .prepare import competition_root
from .public_coatnet import WEIGHTS,load_public_model,public_volume,public_windows
from .train import find_unique


def main():
    require_kaggle_gpu()
    torch.set_num_threads(2)
    root = competition_root()
    with np.load(find_unique('supervision.npz'),allow_pickle=False) as source:
        bank = {k:source[k] for k in source.files}
    selected = np.flatnonzero(np.isin(bank['partitions'],['monitor','comparison','gold']))
    descriptors = pd.read_csv(root/'train_series.csv',dtype={UID:str,'SeriesInstanceUID':str})
    devices = {'384':'cuda:0','224':f'cuda:{min(torch.cuda.device_count()-1,1)}'}
    models = {kind:load_public_model(kind,devices[kind]) for kind in ('384','224')}
    predictions = {'384_plain':[],'384_published_mirror':[],'224_plain':[]}
    missing,elapsed = [],[]
    started = time.time()
    def predict(kind,volume,mask):
        device = devices[kind]
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
            p = models[kind](public_windows(volume,mask,kind,device)).sigmoid().float()
            q = p
            if kind=='384':
                q = models[kind](public_windows(volume,mask,kind,device,mirror=True)).sigmoid().float()
            if not torch.isfinite(p).all() or not torch.isfinite(q).all():
                raise FloatingPointError('Nonfinite public benchmark predictions')
        return p.cpu().numpy(),((p+q)/2).cpu().numpy()
    with ThreadPoolExecutor(max_workers=2) as pool:
        for number,index in enumerate(selected):
            begin = time.time()
            volume,mask,absent = public_volume(root,descriptors,str(bank['uids'][index]))
            futures = {kind:pool.submit(predict,kind,volume,mask) for kind in models}
            p384,tta384 = futures['384'].result()
            p224,_ = futures['224'].result()
            predictions['384_plain'].append(p384)
            predictions['384_published_mirror'].append(tta384)
            predictions['224_plain'].append(p224)
            missing.append(len(absent)); elapsed.append(time.time()-begin)
            if (number+1)%25==0:
                print(json.dumps({'completed':number+1,'total':len(selected),
                                  'elapsed_seconds':time.time()-started}),flush=True)
    evaluations = {}
    for name,values in predictions.items():
        values = np.stack(values)
        evaluations[name] = {}
        for cohort in ('monitor','comparison','gold'):
            subset = bank['partitions'][selected]==cohort
            indices = selected[subset]
            y,w = bank['targets'][0,indices],bank['weights'][0,indices]
            if cohort=='gold':
                y = bank['gold_labels'][indices]
                w = np.isfinite(y).astype(np.float32)
            evaluations[name][cohort] = metrics(y,values[subset],w)
            np.savez_compressed(f'/kaggle/working/{name}_{cohort}.npz',uids=bank['uids'][indices],
                                probabilities=values[subset],targets=y,weights=w,
                                groups=bank['groups'][indices],missing_views=np.asarray(missing)[subset])
    receipt = {'format':'independent-licensed-public-swa-benchmark-v1','target_order':TARGETS,
               'weights_sha256':{k:v[1] for k,v in WEIGHTS.items()},'studies':len(selected),
               'elapsed_seconds':time.time()-started,'per_study_seconds':elapsed,
               'source_sha256':sha256(Path(__file__).with_name('public_coatnet.py')),
               'evaluations':evaluations,'gradient_updates':0,'silent_fallback_rows':0,
               'pretraining_caveat':'Public models were trained on the report pool; not clean OOF',
               'gold_caveat':'Gold was reused for upstream selection; duplicate exposure not established',
               'purpose':'External complementarity reference, not an independently trained candidate',
               'license':'Research/competition and non-commercial adaptation/redistribution expressly permitted',
               'source':'https://www.kaggle.com/datasets/nartaa/rsna-knee-publication-swa-weights-20261007',
               'oai_acknowledgement_required':True,'actual_gpu_location':'Kaggle'}
    write_json('/kaggle/working/public_benchmark_receipt.json',receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k!='per_study_seconds'},indent=2),flush=True)


if __name__=='__main__': main()
