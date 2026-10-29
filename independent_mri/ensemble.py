"""Hash-pinned ensembles of independently fitted models; Kaggle GPU inference only.

This module supplies mechanics, not evidence that any particular blend helps.
Weights and TTA choices must be fixed in a manifest from actual experiment results.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from .contracts import TARGETS,UID,percentile_ranks,require_kaggle_gpu,sha256,validate_predictions,write_json
from .data import windows_from_arrays
from .infer import load_model,mirror_inputs,prepare_study,test_root


def validate_manifest(manifest):
    if manifest.get('format')!='independent-evidence-ensemble-v1':
        raise ValueError('Unknown ensemble manifest format')
    if manifest.get('targets')!=TARGETS:
        raise ValueError('Ensemble target order differs from the official schema')
    if manifest.get('transform') not in ('probability','global_rank'):
        raise ValueError('Unknown probability/rank transform')
    members = manifest.get('members',[])
    if not members or len(members)>4:
        raise ValueError('Expected one to four independently trained ensemble members')
    weights = np.asarray([member['weight'] for member in members],float)
    if weights.ndim!=1 or not np.isfinite(weights).all() or (weights<=0).any() or not np.isclose(weights.sum(),1):
        raise ValueError('Weights must be positive, fixed global scalars summing to one')
    identities = []
    for member in members:
        digest = member.get('checkpoint_sha256','')
        if len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Missing or invalid model SHA-256')
        if not isinstance(member.get('mirror',False),bool):
            raise ValueError('TTA choice must be a fixed boolean')
        identities.append((digest,member.get('mirror',False)))
    if len(set(identities))!=len(identities):
        raise ValueError('Duplicate checkpoint/TTA member')
    return weights


def blend_probabilities(banks,weights,transform):
    values = [np.asarray(bank,float) for bank in banks]
    if not values or len(values)!=len(weights) or any(bank.shape!=values[0].shape for bank in values):
        raise ValueError('Ensemble prediction shapes/member counts differ')
    if values[0].ndim!=2 or values[0].shape[1]!=12 or len(values[0])==0:
        raise ValueError('Ensemble requires complete study-by-twelve matrices')
    if any(not np.isfinite(bank).all() or ((bank<0)|(bank>1)).any() for bank in values):
        raise ValueError('Invalid member probabilities; no fallback is permitted')
    if transform=='global_rank':
        # Called exactly once on the complete official test cohort, never per shard.
        values = [percentile_ranks(bank) for bank in values]
    elif transform!='probability':
        raise ValueError('Unknown ensemble transform')
    return sum(float(weight)*bank for weight,bank in zip(weights,values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest',required=True)
    args = parser.parse_args()
    require_kaggle_gpu()
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    weights = validate_manifest(manifest)
    started = time.time()
    members = []
    for i,member in enumerate(manifest['members']):
        device = f'cuda:{i%min(torch.cuda.device_count(),2)}'
        model,training = load_model(member['checkpoint'],member['checkpoint_sha256'],device)
        members.append((model,training['configuration']['configuration'],device))
    root = test_root()
    test = pd.read_csv(root/'test.csv',dtype={UID:str},usecols=[UID])
    if test[UID].duplicated().any() or len(test)==0:
        raise ValueError('Official test UID table is empty or duplicated')
    descriptors = pd.read_csv(root/'test_series.csv',dtype={UID:str,'SeriesInstanceUID':str})
    banks = [[] for _ in members]
    missing = [0]*5
    for uid in test[UID]:
        arrays,audit = prepare_study(root,descriptors,uid)
        for slot in audit['slots']:
            missing[slot['slot']] += int(slot['missing'])
        prepared = {}
        for index,((model,configuration,device),member) in enumerate(zip(members,manifest['members'])):
            if configuration not in prepared:
                prepared[configuration] = windows_from_arrays(arrays,audit,configuration)
            images,metadata,mask = (value.to(device) for value in prepared[configuration])
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
                p = model(images[None],metadata[None],mask[None]).sigmoid().float()[0]
                if member.get('mirror',False):
                    mirrored,md = mirror_inputs(images,metadata)
                    q = model(mirrored[None],md[None],mask[None]).sigmoid().float()[0]
                    p = (p+q)/2
            banks[index].append(p.cpu().numpy())
        if len(banks[0])%25==0:
            print(json.dumps({'progress':len(banks[0]),'total':len(test),
                              'elapsed_seconds':time.time()-started}),flush=True)
    stacked = [np.stack(bank) for bank in banks]
    frame = test.copy()
    frame[TARGETS] = blend_probabilities(stacked,weights,manifest['transform'])
    validate_predictions(frame,test[UID].tolist())
    output = Path('/kaggle/working/submission.csv')
    frame.to_csv(output,index=False)
    # Save raw component probabilities for reproducibility; these stay private.
    np.savez_compressed('/kaggle/working/component_probabilities.npz',uids=test[UID].to_numpy(),
                        probabilities=np.stack(stacked))
    receipt = {'format':'independent-evidence-ensemble-inference-v1','rows':len(frame),
               'manifest_sha256':sha256(manifest_path),'manifest':manifest,
               'submission_sha256':sha256(output),'elapsed_seconds':time.time()-started,
               'missing_slots':missing,'silent_fallback_rows':0,'reports_consumed':False,
               'global_rank_scope':'complete official test cohort',
               'gpu_execution':'Kaggle only','actual_devices':[torch.cuda.get_device_name(i)
                    for i in range(torch.cuda.device_count())]}
    write_json('/kaggle/working/inference_receipt.json',receipt)
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__': main()
