"""Build leakage-audited refinement targets from our OWN held-fold MRI teachers."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import numpy as np

from .contracts import TARGETS,sha256,write_json
from .train import find_unique


def validate_exposure(receipt,supervision,fold,supervision_hash,prediction_hash):
    fitting = (supervision['partitions']=='train')&(supervision['folds']!=fold)
    held = (supervision['partitions']=='train')&(supervision['folds']==fold)
    digest = lambda values:hashlib.sha256('|'.join(values).encode()).hexdigest()
    expected = {'own_teacher_fold':fold,'source_supervision_sha256':supervision_hash,
                'prediction_sha256':prediction_hash,'training_rows':int(fitting.sum()),
                'oof_rows':int(held.sum()),'fitting_uids_sha256':digest(supervision['uids'][fitting]),
                'oof_uids_sha256':digest(supervision['uids'][held]),
                'preserved_initialization_file_sha256':'c4ad2da51831278066da37165374325ce1beb6c9e3ba2950d0edfbab7d4f8563',
                'duplicate_group_overlap':0,'gold_in_fitting':False}
    for key,value in expected.items():
        if receipt.get(key)!=value:
            raise ValueError(f'Unverified/contaminated OOF teacher exposure: {key}')
    if set(supervision['groups'][fitting])&set(supervision['groups'][held]):
        raise ValueError('OOF groups overlap fitting groups')


def main():
    original = find_unique('supervision.npz')
    with np.load(original,allow_pickle=False) as bank:
        supervision = {key:bank[key] for key in bank.files}
    uids = supervision['uids']
    lookup = {uid:i for i,uid in enumerate(uids)}
    oof = np.full((len(uids),12),np.nan,np.float32)
    exposures = []
    for fold in range(3):
        suffix = f'teacher-fold-{fold}/own_oof_predictions.npz'
        files = list(Path('/kaggle/input').glob('*/'+suffix))
        files += list(Path('/kaggle/input').glob('notebooks/*/*/'+suffix))
        if len(files)!=1:
            raise ValueError(f'Expected one independent teacher fold {fold}, found {files}')
        path = files[0]
        receipt = json.loads((path.parent/'oof_exposure.json').read_text())
        validate_exposure(receipt,supervision,fold,sha256(original),sha256(path))
        training_receipt = json.loads((path.parent/'training_receipt.json').read_text())
        model_manifest_path = path.parent/'best_manifest.json'
        model_manifest = json.loads(model_manifest_path.read_text())
        if (training_receipt['checkpoint_sha256']!=receipt['checkpoint_sha256']
                or training_receipt['best_manifest_sha256']!=sha256(model_manifest_path)
                or not training_receipt['strict_reload'] or not training_receipt['encoder_updated']
                or model_manifest['target_order']!=TARGETS
                or model_manifest['configuration']['initialization']!='imagenet'
                or model_manifest['configuration']['teacher_fold']!=fold
                or model_manifest['fitting_uids_sha256']!=receipt['fitting_uids_sha256']):
            raise ValueError('OOF artifact does not match its fitted checkpoint provenance')
        with np.load(path,allow_pickle=False) as predictions:
            ids,p = predictions['uids'],predictions['probabilities']
            if (p.shape!=(len(ids),12) or len(set(ids))!=len(ids) or not np.isfinite(p).all()
                    or ((p<0)|(p>1)).any()):
                raise ValueError('Invalid OOF prediction bank')
            for uid,value in zip(ids,p):
                index = lookup[uid]
                if supervision['partitions'][index]!='train' or supervision['folds'][index]!=fold:
                    raise ValueError('Teacher predicted a study outside its assigned excluded fitting fold')
                if np.isfinite(oof[index]).any():
                    raise ValueError('Duplicate OOF teacher coverage')
                oof[index] = value
        exposures.append({'fold':fold,'prediction_sha256':sha256(path),**receipt})
    train = supervision['partitions']=='train'
    if not np.isfinite(oof[train]).all() or np.isfinite(oof[~train]).any():
        raise ValueError('OOF coverage is incomplete or includes monitor/comparison/Gold studies')
    report = supervision['targets'][2].astype(np.float32)
    weight = supervision['weights'][2].astype(np.float32)
    targets = np.repeat(report[None],4,axis=0)
    weights = np.repeat(weight[None],4,axis=0)
    agreement = np.sign(report-.5)==np.sign(np.nan_to_num(oof,nan=.5)-.5)
    supported = train[:,None] & (weight>0)
    targets[1][supported] = (.75*report+.25*np.nan_to_num(oof,nan=.5))[supported]
    confident = supported & agreement & (2*np.abs(report-.5)>=.7) & (2*np.abs(np.nan_to_num(oof,nan=.5)-.5)>=.6)
    targets[2][confident] = (.8*report+.2*np.nan_to_num(oof,nan=.5))[confident]
    reliability = np.zeros(12,np.float32)
    for j in range(12):
        known = supported[:,j]
        reliability[j] = float(np.mean(agreement[known,j])) if known.any() else 0
    # Fixed target-dependent scaling derives from out-of-training agreement on fitting studies,
    # NEVER from Gold or comparison labels. Agreement still does not prove correctness.
    mixing = np.clip((reliability-.5)/.5,0,1)*.25
    refined = (1-mixing[None])*report+mixing[None]*np.nan_to_num(oof,nan=.5)
    targets[3][confident] = refined[confident]
    np.savez_compressed('/kaggle/working/refined_supervision.npz',uids=uids,
                        targets=targets,weights=weights,oof_probabilities=oof,partitions=supervision['partitions'])
    manifest = {'format':'independent-own-oof-distillation-v1','source_supervision_sha256':sha256(original),
                'refined_sha256':sha256('/kaggle/working/refined_supervision.npz'),'teacher_exposures':exposures,
                'policies':['T1 report only','T2 75/25 report/OOF','T3 supported agreement-aware 80/20',
                            'T4 target-dependent OOF agreement weighting'],
                'refined_cells':[0,int(supported.sum()),int(confident.sum()),int(confident.sum())],
                'target_reliability_from_train_oof_agreement':reliability.tolist(),
                'missing_unmentioned_teacher_only_labels_added':0,'gold_used_to_fit_mixing':False,
                'clinical_correctness_certified':False,'refinement_improves_predictions':'UNTESTED'}
    write_json('/kaggle/working/distillation_manifest.json',manifest)
    print(json.dumps(manifest,indent=2),flush=True)


if __name__=='__main__': main()
