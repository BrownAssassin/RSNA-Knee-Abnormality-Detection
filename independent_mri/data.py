"""Memory-bounded windows from the declared broad-coverage source-slice grids.

Three channels are adjacent within that representation, not necessarily adjacent
native acquisitions. P1/P2/P3 budgets count distinct selected source frames.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from .contracts import CONFIGS, UID, sha256


def discover_caches(root=Path('/kaggle/input')):
    receipts = list(root.glob('*/preparation_receipt.json'))
    receipts += list(root.glob('notebooks/*/*/preparation_receipt.json'))
    files = {}
    for path in receipts:
        receipt = json.loads(path.read_text())
        if receipt.get('format') != 'independent-physical-mri-cache-v1' or not receipt.get('complete'):
            raise RuntimeError(f'Incomplete or foreign MRI cache receipt: {path}')
        for record in receipt['receipts']:
            uid = record[UID]
            if uid in files:
                raise ValueError(f'Duplicate UID across cache shards: {uid}')
            file = path.parent/'cache'/f'{uid}.npz'
            if not file.is_file() or not file.with_suffix('.json').is_file():
                raise FileNotFoundError(f'Missing cached image/audit: {file}')
            files[uid] = file
    if not files:
        raise FileNotFoundError('No complete independent cache shards attached')
    return files,receipts


def window_plan(length, count, rng=None, keep=None):
    if length<=0:
        raise ValueError('Cannot sample an empty MRI acquisition')
    centers = np.unique(np.linspace(0,length-1,min(length,count)).round().astype(int))
    if rng is not None and keep is not None and len(centers)>keep:
        # Stratify into bins to preserve broad coverage during training, not just a central crop.
        bins = np.array_split(centers,keep)
        centers = np.array([rng.choice(part) for part in bins])
    return centers


def windows_from_arrays(arrays,audit,configuration='P1',training=False,windows=12,seed=0):
    size,allocation = CONFIGS[configuration]
    available = [slot for slot in range(5) if f'slot_{slot}' in arrays]
    if not available:
        raise ValueError('No available MRI sequence')
    rng = np.random.default_rng(seed)
    counts = {slot: windows//len(available)+(i<windows%len(available)) for i,slot in enumerate(available)}
    images,metadata = [],[]
    for slot in available:
        stack = arrays[f'slot_{slot}']
        info = audit['slots'][slot]
        plane = {'Sagittal':0,'Coronal':1,'Axial':2}[info['plane']]
        sequence = int(info['fluid'])*2+int(info['fat_suppression'])
        source_grid = window_plan(len(stack),allocation[slot])
        centers = np.arange(len(source_grid))
        if training and len(centers)>counts[slot]:
            # Neighboring windows enable genuine local transitions. Random block
            # placement covers peripheral anatomy over epochs; inference is broad.
            keep = counts[slot]
            start = int(rng.integers(0,len(centers)-keep+1))
            centers = centers[start:start+keep]
        coordinates = np.array([s['coordinate_mm'] for s in info['sources']])
        lo,hi = float(coordinates.min()),float(coordinates.max())
        for representation_index in centers:
            native_indices = source_grid[np.clip([representation_index-1,representation_index,representation_index+1],
                                                  0,len(source_grid)-1)]
            center = native_indices[1]
            triplet = stack[native_indices]
            images.append(triplet)
            position = 2*(float(coordinates[center])-lo)/max(hi-lo,1e-6)-1
            metadata.append([plane,sequence,position,slot,float(coordinates[center])])
    image = torch.from_numpy(np.stack(images).astype(np.float32)/255)
    image = F.interpolate(image,size=(size,size),mode='bilinear',align_corners=False,antialias=True)
    if training:
        # Intensity only; no generic left/right flip or unrecorded orientation change.
        gamma = float(rng.uniform(.9,1.1))
        image = (image.pow(gamma)*float(rng.uniform(.9,1.1))).clamp(0,1)
    mean = image.new_tensor([.485,.456,.406])[:,None,None]
    std = image.new_tensor([.229,.224,.225])[:,None,None]
    image = (image-mean)/std
    return image,torch.tensor(metadata,dtype=torch.float32),torch.ones(len(image),dtype=torch.bool)


class StudyDataset(Dataset):
    def __init__(self,files,supervision,indices,configuration='P1',policy='L3',training=False,windows=12):
        self.files,self.supervision,self.indices = files,supervision,np.asarray(indices)
        self.configuration,self.policy,self.training,self.windows = configuration,int(policy[1:])-1,training,windows
        self.epoch = 0
        group_ids,group_counts = np.unique(supervision['groups'],return_counts=True)
        self.group_sizes = dict(zip(group_ids,group_counts))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self,item):
        index = self.indices[item]
        uid = str(self.supervision['uids'][index])
        file = self.files[uid]
        audit = json.loads(file.with_suffix('.json').read_text())
        if audit.get(UID)!=uid:
            raise ValueError('MRI cache metadata/filename study ID mismatch')
        with np.load(file,allow_pickle=False) as data:
            arrays = {name:data[name] for name in data.files}
        for slot,info in enumerate(audit['slots']):
            key = f'slot_{slot}'
            if info['missing']:
                if key in arrays: raise ValueError('Missing-series mask contradicts cached pixels')
            elif (key not in arrays or arrays[key].dtype!=np.uint8 or arrays[key].ndim!=3
                  or len(arrays[key])!=info['native_slice_count']
                  or len(info['sources'])!=info['native_slice_count']):
                raise ValueError('MRI pixel shape/dtype/source mapping integrity failure')
        image,metadata,mask = windows_from_arrays(arrays,audit,self.configuration,self.training,self.windows,
                                                   seed=20261022+index+self.epoch*100000)
        return {'images':image,'metadata':metadata,'mask':mask,'uid':uid,
                'study_weight':torch.tensor(1/self.group_sizes[self.supervision['groups'][index]],dtype=torch.float32),
                'targets':torch.from_numpy(self.supervision['targets'][self.policy,index].astype(np.float32)),
                'weights':torch.from_numpy(self.supervision['weights'][self.policy,index].astype(np.float32))}


def collate_studies(rows):
    length = max(len(r['images']) for r in rows)
    output = {}
    for key in ('images','metadata','mask'):
        values = []
        for row in rows:
            value = row[key]
            padded = torch.zeros((length,)+value.shape[1:],dtype=value.dtype)
            padded[:len(value)] = value
            values.append(padded)
        output[key] = torch.stack(values)
    output['uid'] = [r['uid'] for r in rows]
    for key in ('targets','weights','study_weight'):
        output[key] = torch.stack([r[key] for r in rows])
    return output
