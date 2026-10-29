# SPDX-License-Identifier: GPL-3.0-or-later
# Separate external-model screen. Architecture follows DOSMA's published GPLv3
# IWOAI 2019 Team 6 U-Net. Cite Desai et al., DOSMA and the IWOAI challenge.
# https://github.com/ad12/DOSMA/blob/master/dosma/models/oaiunet2d.py
"""Generate genuine candidate anatomy masks on Kaggle; never assume quality.

This module is not imported by the independent classifier. Its external weights
retain their GPLv3 terms; no anatomical training signal is enabled by this screen.
"""
from __future__ import annotations

import os
os.environ['KERAS_BACKEND'] = 'torch'

import argparse
import json
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd
from PIL import Image
import torch

from .contracts import TARGETS,UID,require_kaggle_gpu,sha256,write_json
from .dicom import read_series
from .prepare import competition_root

WEIGHT_URL = ('https://huggingface.co/arjundd/dosma-models/resolve/main/'
              'iwoai-2019-t6-normalized/iwoai-2019-unet2d-normalized_fc-tc-pc-men_weights.h5')
WEIGHT_SHA256 = 'a175bf63f22c9c2308a0fc7d0fdb8b5ecf6cd2f875dde1de8f385dc4e0e91fd1'


def external_model():
    import keras
    layers = keras.layers
    inputs = layers.Input((384,384,1))
    value,skips = inputs,[]
    widths = [32,64,128,256,512,1024]
    for depth,width in enumerate(widths):
        value = layers.Conv2D(width,3,padding='same',activation='relu',kernel_initializer='he_normal')(value)
        value = layers.Conv2D(width,3,padding='same',activation='relu',kernel_initializer='he_normal')(value)
        value = layers.BatchNormalization(axis=-1,momentum=.95,epsilon=.001)(value)
        value = layers.Dropout(0)(value)
        skips.append(value)
        if depth<5:
            value = layers.MaxPooling2D(pool_size=2)(value)
    for depth in range(4,-1,-1):
        value = layers.Conv2DTranspose(widths[depth],3,padding='same',strides=2)(value)
        value = layers.Concatenate(axis=3)([value,skips[depth]])
        value = layers.Conv2D(widths[depth],3,padding='same',activation='relu',kernel_initializer='he_normal')(value)
        value = layers.Conv2D(widths[depth],3,padding='same',activation='relu',kernel_initializer='he_normal')(value)
        value = layers.BatchNormalization(axis=-1,momentum=.95,epsilon=.001)(value)
        value = layers.Dropout(0)(value)
    outputs = layers.Conv2D(4,1,padding='same',activation='sigmoid')(value)
    return keras.Model(inputs,outputs)


def overlay(image,probabilities):
    base = np.repeat(image[:,:,None],3,axis=2).astype(np.float32)
    colors = np.array([[255,75,75],[75,255,75],[75,125,255],[255,210,50]],np.float32)
    for channel,color in enumerate(colors):
        mask = probabilities[:,:,channel]>.5
        base[mask] = .55*base[mask]+.45*color
    return np.clip(base,0,255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',choices=['iwoai','skmtea'],default='iwoai')
    args = parser.parse_args()
    require_kaggle_gpu()
    torch.set_num_threads(2)
    out = Path('/kaggle/working/anatomy_screen'); out.mkdir(exist_ok=True)
    checkpoint = Path('/kaggle/working/dosma_iwoai_normalized.h5')
    weight_url,expected_sha256 = WEIGHT_URL,WEIGHT_SHA256
    structure_order = ['femoral_cartilage','tibial_cartilage','patellar_cartilage','meniscus']
    if args.model=='skmtea':
        weight_url = 'https://huggingface.co/arjundd/dosma-models/resolve/main/skm-tea/qDESS_2021_v1-rms-unet2d-pc_fc_tc_men_weights.h5'
        expected_sha256 = 'd5459b8d96cfe60e2a22e16428de5ecab781896fc38ffde5c679fdc8d53d3640'
        structure_order = ['patellar_cartilage','femoral_cartilage','tibial_cartilage','meniscus']
    urllib.request.urlretrieve(weight_url,checkpoint)
    if sha256(checkpoint)!=expected_sha256:
        raise ValueError('External anatomy checkpoint SHA-256 mismatch')
    model = external_model()
    model.load_weights(str(checkpoint),skip_mismatch=False)
    variable = model.trainable_variables[0].value
    if variable.device.type!='cuda':
        raise RuntimeError('Anatomy model must run on a Kaggle GPU, not CPU/laptop')
    root = competition_root()
    train = pd.read_csv(root/'train.csv',dtype={UID:str})
    rows = train.loc[train[TARGETS].isna().all(axis=1)].sort_values(UID)
    indices = np.linspace(0,len(rows)-1,6).round().astype(int)
    series = pd.read_csv(root/'train_series.csv',dtype={UID:str,'SeriesInstanceUID':str})
    audits = []
    for number,index in enumerate(indices):
        uid = rows.iloc[index][UID]
        choices = series[series[UID].eq(uid)&series.Anatomical_Plane.eq('Sagittal')].sort_values(
            ['Fluid_Sensitive','Fat_Suppression','SeriesInstanceUID'],ascending=[False,True,True])
        if choices.empty:
            audits.append({UID:uid,'missing_sagittal':True}); continue
        row = choices.iloc[0]
        stack,source = read_series(root/'train_series'/uid/row.SeriesInstanceUID,'Sagittal',fov_mm=140.16)
        import pydicom
        header_path = next((root/'train_series'/uid/row.SeriesInstanceUID).glob('*.dcm'))
        header = pydicom.dcmread(header_path,stop_before_pixels=True)
        description = str(getattr(header,'SeriesDescription',''))
        # DOSMA expects whole-volume whitening and SI/AP in-plane orientation.
        normalized = (stack.astype(np.float32)-float(stack.mean()))/max(float(stack.std()),1e-8)
        select = np.linspace(0,len(stack)-1,min(16,len(stack))).round().astype(int)
        probabilities = []
        with torch.no_grad():
            for start in range(0,len(select),2):
                value = torch.from_numpy(normalized[select[start:start+2],:,:,None]).to(variable.device)
                prediction = model(value,training=False).detach().float().cpu().numpy()
                if not np.isfinite(prediction).all():
                    raise FloatingPointError('Nonfinite anatomical masks')
                probabilities.append(prediction)
        probabilities = np.concatenate(probabilities)
        mask = probabilities>.5
        stats = {'study_number':number,UID:uid,'series_uid':str(row.SeriesInstanceUID),
                 'sequence_description':description,'native_dess':bool('dess' in description.lower()),
                 'structure_order':structure_order,
                 'foreground_voxels_by_structure':mask.sum((0,1,2)).tolist(),
                 'multistructure_overlap_fraction':float((mask.sum(-1)>1).mean()),
                 'empty_all_structures':bool(not mask.any()),'selected_native_indices':select.tolist(),
                 'independent_visual_review':'PENDING; computational counts do not establish mask quality',
                 'fit_to_pathology_labels':False,'source':source}
        np.savez_compressed(out/f'candidate_masks_{number}.npz',source_indices=select,probabilities=probabilities.astype(np.float16))
        for view in np.unique(np.linspace(0,len(select)-1,3).round().astype(int)):
            original = np.repeat(stack[select[view],:,:,None],3,axis=2)
            panel = np.concatenate((original,overlay(stack[select[view]],probabilities[view])),axis=1)
            Image.fromarray(panel).save(out/f'review_{number}_{view}.png')
        audits.append(stats)
    receipt = {'format':'independent-anatomy-feasibility-screen-v1','weights_url':weight_url,
               'weights_sha256':expected_sha256,'external_model':args.model,
               'external_license':'GPL-3.0, retain DOSMA and challenge attribution',
               'backend':'Keras torch on Kaggle CUDA','strict_weight_load':True,'studies':audits,
               'domain_caveat':'OAI DESS segmenter applied to potentially non-DESS clinical scans',
               'geometry_caveat':'In-plane canonical resampling, not a 3D isotropic DESS reformat',
               'quantization_caveat':'uint8 input may differ from checkpoint-native raw-intensity preprocessing',
               'quality_approved':False,'used_for_classifier_training':False,
               'synthetic_anatomy_used':False}
    write_json(out/'screen_receipt.json',receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k!='studies'},indent=2),flush=True)


if __name__=='__main__': main()
