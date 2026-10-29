"""Independent adapter for the explicitly competition-permitted Nartaa SWA models.

Architecture/preprocessing lineage: dreaddevelopment Raptor MIL and nartaa's
Apache-2.0 published training code. Authoritative scored recipe inspected at:
https://www.kaggle.com/code/heliosli/blend-mine-in-private-lb-luck-to-all-upvoters
Weights and OAI acknowledgement retain the release's non-commercial terms:
https://www.kaggle.com/datasets/nartaa/rsna-knee-publication-swa-weights-20261007

This is an external benchmark, not our new pathology architecture or clean OOF.
Its native orientation/normalization intentionally differs from our own pipeline
to match its public training representation. Decoding failures remain fatal.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .contracts import TARGETS,require_kaggle_gpu,sha256

SLOTS = [('Sagittal',1,26),('Sagittal',0,22),('Coronal',1,18),
         ('Coronal',0,12),('Axial',-1,18)]
WEIGHTS = {
    '384':('raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt',
           '7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef'),
    '224':('raptor_ft_alldata_t16_blendjev_oai_d96_c08_r224_swa.pt',
           '3394fdd7e668286956e85af2554bef0386df5cd79d1c1af223b095c828b815d6')}


class PublicRaptor(nn.Module):
    def __init__(self,resolution,architecture):
        super().__init__()
        import timm
        # The published 384 reader overrides the backbone grid to native320;
        # the 224 reader leaves timm's construction grid at its architecture default.
        # They must not be made artificially identical at checkpoint loading.
        extra = {'img_size':resolution} if resolution==320 else {}
        self.backbone = timm.create_model(architecture,pretrained=False,num_classes=0,
                                          in_chans=3,global_pool='avg',**extra)
        width = self.backbone.num_features
        self.norm = nn.LayerNorm(width)
        self.att = nn.Sequential(nn.Linear(width,256),nn.Tanh(),nn.Dropout(.2),nn.Linear(256,12))
        self.clsW = nn.Parameter(torch.zeros(12,width))
        self.clsb = nn.Parameter(torch.zeros(12))

    def forward(self,windows):
        features = torch.cat([self.backbone(chunk) for chunk in windows.split(16)])
        features = self.norm(features)
        attention = self.att(features).softmax(0)
        return (torch.einsum('kt,kd->td',attention,features)*self.clsW).sum(-1)+self.clsb


def load_public_model(kind,device):
    require_kaggle_gpu()
    name,digest = WEIGHTS[kind]
    roots = list(Path('/kaggle/input').glob('*/'+name))
    roots += list(Path('/kaggle/input').glob('datasets/*/*/'+name))
    if len(roots)!=1 or sha256(roots[0])!=digest:
        raise ValueError('Expected one exact authorized public SWA checkpoint')
    saved = torch.load(roots[0],map_location='cpu',weights_only=False)
    if int(saved.get('res',int(kind)))!=int(kind):
        raise ValueError('Public checkpoint training resolution differs')
    if 'targets' in saved and list(saved['targets'])!=TARGETS:
        raise ValueError('Public checkpoint target order differs')
    model = PublicRaptor(320 if kind=='384' else 224,
                         saved.get('arch','coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k'))
    model.load_state_dict(saved['model'],strict=True)
    return model.to(device).eval()


def public_volume(root,descriptors,uid):
    import cv2
    import pydicom
    from pydicom.pixels import apply_modality_lut
    rows = descriptors[descriptors['StudyInstanceUID'].eq(uid)].to_dict('records')
    used = set()
    volume = np.zeros((96,384,384),np.uint8)
    offset = 0
    missing = []
    for slot,(plane,fluid,count) in enumerate(SLOTS):
        available = [r for r in rows if r['Anatomical_Plane']==plane and r['SeriesInstanceUID'] not in used]
        preferred = [r for r in available if int(r['Fluid_Sensitive'])==fluid] if fluid in (0,1) else available
        chosen = (preferred or available)
        if not chosen:
            missing.append(slot); offset += count; continue
        selected = chosen[0]
        used.add(selected['SeriesInstanceUID'])
        files = list((root/'train_series'/uid/selected['SeriesInstanceUID']).glob('*.dcm'))
        if not files:
            raise FileNotFoundError('Selected public-model series has no DICOM files')
        def header(file):
            h = pydicom.dcmread(file,stop_before_pixels=True)
            iop = np.asarray(h.ImageOrientationPatient,float)
            position = float(np.dot(np.asarray(h.ImagePositionPatient,float),np.cross(iop[:3],iop[3:])))
            spacing = float(h.PixelSpacing[0])
            if spacing<=0 or not np.isfinite(position):
                raise ValueError('Invalid public reference geometry')
            return position,file,spacing
        with ThreadPoolExecutor(max_workers=16) as pool:
            ordered = sorted(pool.map(header,files),key=lambda x:x[0])
        lo = int(len(ordered)*.02)
        hi = max(lo,int(len(ordered)*.98)-1)
        picks = np.linspace(lo,hi,count).round().astype(int) if len(ordered)>1 else np.zeros(count,int)
        chosen_files = [ordered[min(i,len(ordered)-1)] for i in picks]
        def read(record):
            _,file,spacing = record
            d = pydicom.dcmread(file)
            a = apply_modality_lut(d.pixel_array,d).astype(np.float32)
            if a.ndim!=2 or not np.isfinite(a).all():
                raise ValueError('Nonfinite/non-slice public reference pixels')
            if str(getattr(d,'PhotometricInterpretation',''))=='MONOCHROME1':
                a = a.max()-a
            return a,spacing
        with ThreadPoolExecutor(max_workers=8) as pool:
            decoded = list(pool.map(read,chosen_files))
        lower,upper = np.percentile(np.concatenate([a.ravel() for a,_ in decoded]),[2,98])
        for local,(pixels,spacing) in enumerate(decoded):
            pixels = np.clip((pixels-lower)/(upper-lower+1e-6),0,1)
            height,width = pixels.shape
            crop = min(int(round(140/max(spacing,1e-3))),height,width)
            y,x = (height-crop)//2,(width-crop)//2
            resized = cv2.resize(pixels[y:y+crop,x:x+crop],(384,384),interpolation=cv2.INTER_AREA)
            volume[offset+local] = (resized*255).astype(np.uint8)
        offset += count
    mask = volume.reshape(96,-1).sum(1)>0
    if not mask.any():
        raise ValueError('Public reference has no valid acquisitions; no default prediction')
    return volume,mask,missing


def public_windows(volume,mask,kind,device,mirror=False):
    valid = np.flatnonzero(mask)
    if len(valid)<3:
        raise ValueError('Public reference cannot form valid three-slice windows')
    lo,hi = int(valid.min()),int(valid.max())
    centers = np.arange(lo+1,hi)
    if not len(centers):
        raise ValueError('Empty public reference window grid')
    centers = centers[np.linspace(0,len(centers)-1,94).round().astype(int)]
    tri = torch.from_numpy(np.ascontiguousarray(volume[np.stack((centers-1,centers,centers+1),1)])).to(device)
    if mirror:
        if kind!='384':
            raise ValueError('No unverified TTA policy for the public 224 model')
        sagittal = torch.tensor(centers<48,device=device).view(-1,1,1,1)
        tri = torch.where(sagittal,tri.flip(1),tri.flip(-1))
    image = tri.float()/255
    if kind=='384':
        image = image[:,:,32:352,32:352]  # Published native320 from384-trained images.
    else:
        crop = int(round(384*.8))
        start = (384-crop)//2
        image = F.interpolate(image[:,:,start:start+crop,start:start+crop],size=(224,224),
                              mode='bilinear',align_corners=False)
    mean = image.new_tensor([.485,.456,.406]).view(1,3,1,1)
    std = image.new_tensor([.229,.224,.225]).view(1,3,1,1)
    return (image-mean)/std
