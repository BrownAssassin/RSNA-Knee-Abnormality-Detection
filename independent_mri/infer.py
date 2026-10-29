"""Offline real-DICOM Kaggle inference for independently trained EvidenceNet."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from .contracts import TARGETS,UID,require_kaggle_gpu,sha256,validate_predictions,write_json
from .data import windows_from_arrays
from .dicom import read_series,select_series
from .model import EvidenceNet


def test_root():
    candidates = list(Path('/kaggle/input').glob('*/test.csv'))
    candidates += list(Path('/kaggle/input').glob('competitions/*/test.csv'))
    candidates = [path.parent for path in candidates if (path.parent/'test_series.csv').is_file()]
    if len(candidates)!=1:
        raise RuntimeError(f'Expected unique official test mount; found {candidates}')
    return candidates[0]


def prepare_study(root,descriptors,uid):
    selected = select_series(descriptors[descriptors[UID].eq(uid)])
    arrays,slots = {},[]
    for slot,row in enumerate(selected):
        if row is None:
            slots.append({'slot':slot,'missing':True}); continue
        array,audit = read_series(root/'test_series'/uid/str(row['SeriesInstanceUID']),
                                   expected_plane=row['Anatomical_Plane'])
        arrays[f'slot_{slot}'] = array
        slots.append({'slot':slot,'missing':False,'series_uid':str(row['SeriesInstanceUID']),
                      'fluid':int(row['Fluid_Sensitive']),'fat_suppression':int(row['Fat_Suppression']),**audit})
    if not arrays:
        raise RuntimeError(f'Test study has no usable MRI acquisitions: {uid}')
    return arrays,{'slots':slots}


def mirror_inputs(images,metadata):
    """Physical left/right display reflection; anatomy identities stay unchanged.

    Sagittal depth positions reverse; coronal/axial in-plane left/right flips.
    This does not exchange anatomical menisci or OA target identities.
    """
    result,md = images.clone(),metadata.clone()
    for slot in range(5):
        indices = (md[:,3].long()==slot).nonzero().flatten()
        if not len(indices): continue
        plane = int(md[indices[0],0])
        if plane==0:
            # Images are already ImageNet-normalized. Channel reversal must act
            # on intensities, NOT exchange channel-specific means/stds. Undo
            # normalization, reflect depth, then normalize each destination channel.
            mean = images.new_tensor([.485,.456,.406]).view(1,3,1,1)
            std = images.new_tensor([.229,.224,.225]).view(1,3,1,1)
            raw = result[indices.flip(0)].clone()*std+mean
            result[indices] = (raw.flip(1)-mean)/std
            md[indices,2] = -md[indices.flip(0),2].clone()
            if md.shape[-1]>4:
                md[indices,4] = -md[indices.flip(0),4].clone()
        else:
            result[indices] = result[indices].flip(-1)
    return result,md


def resolve_checkpoint(checkpoint,expected_sha256):
    path = Path(checkpoint)
    if not path.is_file():
        candidates = list(Path('/kaggle/input').glob('*/'+checkpoint))
        candidates += list(Path('/kaggle/input').glob('notebooks/*/*/'+checkpoint))
        candidates = [p for p in candidates if p.is_file() and sha256(p)==expected_sha256]
        if len(candidates)!=1:
            raise FileNotFoundError(f'Expected one hash-matched checkpoint {checkpoint}, got {candidates}')
        path = candidates[0]
    if sha256(path)!=expected_sha256:
        raise ValueError('Independent checkpoint SHA-256 mismatch')
    return path


def validate_runtime_sources(manifest,root=None):
    """A strict tensor load alone does not prove the same computation is used."""
    root = Path(root) if root is not None else Path(__file__).parent
    source_hashes = manifest.get('source_sha256',{})
    for name in ('model.py','data.py','dicom.py','contracts.py'):
        expected = source_hashes.get(name)
        if not expected or not (root/name).is_file() or sha256(root/name)!=expected:
            raise ValueError(f'Checkpoint runtime-source provenance mismatch: {name}')


def load_model(checkpoint,expected_sha256,device='cuda:0'):
    require_kaggle_gpu()
    path = resolve_checkpoint(checkpoint,expected_sha256)
    saved = torch.load(path,map_location='cpu',weights_only=False)
    if saved.get('manifest',{}).get('target_order')!=TARGETS:
        raise ValueError('Checkpoint target order is missing or incompatible')
    validate_runtime_sources(saved['manifest'])
    model = EvidenceNet(pretrained=False,**saved['config']).to(device).eval()
    model.load_state_dict(saved['state_dict'],strict=True)
    return model,saved['manifest']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--sha256',required=True)
    parser.add_argument('--mirror',action='store_true')
    args = parser.parse_args()
    require_kaggle_gpu()
    model,manifest = load_model(args.checkpoint,args.sha256)
    configuration = manifest['configuration']['configuration']
    root = test_root()
    test = pd.read_csv(root/'test.csv',dtype={UID:str},usecols=[UID])
    if test[UID].duplicated().any(): raise ValueError('Duplicate official test IDs')
    descriptors = pd.read_csv(root/'test_series.csv',dtype={UID:str,'SeriesInstanceUID':str})
    predictions,audits,timings = [],[],[]
    started = time.time()
    for uid in test[UID]:
        begin = time.time()
        arrays,audit = prepare_study(root,descriptors,uid)
        image,metadata,mask = windows_from_arrays(arrays,audit,configuration)
        image,metadata,mask = image.cuda(),metadata.cuda(),mask.cuda()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
            p = model(image[None],metadata[None],mask[None]).sigmoid().float()[0]
            if args.mirror:
                mirrored,mirror_metadata = mirror_inputs(image,metadata)
                q = model(mirrored[None],mirror_metadata[None],mask[None]).sigmoid().float()[0]
                p = (p+q)/2
        predictions.append(p.cpu().numpy())
        timings.append(time.time()-begin)
        audits.append({UID:uid,'missing_slots':[s['slot'] for s in audit['slots'] if s['missing']]})
        if len(predictions)%25==0:
            print(json.dumps({'progress':len(predictions),'total':len(test),'elapsed_s':time.time()-started}),flush=True)
    frame = test.copy()
    frame[TARGETS] = np.stack(predictions)
    validate_predictions(frame,test[UID].tolist())
    frame.to_csv('/kaggle/working/submission.csv',index=False)
    receipt = {'format':'independent-pathology-evidence-inference-v1','checkpoint_sha256':args.sha256,
               'configuration':configuration,'mirror':args.mirror,'rows':len(frame),'columns':frame.columns.tolist(),
               'elapsed_seconds':time.time()-started,'per_study_seconds':timings,
               'submission_sha256':sha256('/kaggle/working/submission.csv'),'silent_fallback_rows':0,
               'reports_consumed':False,'anatomy_supervision_claimed':False,'audits':audits,
               'actual_gpu':torch.cuda.get_device_name(),'offline':True}
    write_json('/kaggle/working/inference_receipt.json',receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k not in ['audits','per_study_seconds']},indent=2),flush=True)


if __name__=='__main__': main()
