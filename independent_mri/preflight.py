"""Kaggle-only real-MRI encoder update and strict reload test; no test-image fitting."""
import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .contracts import TARGETS, UID, require_kaggle_gpu, sha256, write_json
from .dicom import read_series, select_series
from .model import EvidenceNet, supervised_loss
from .prepare import competition_root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone',default='convnext_small.fb_in22k_ft_in1k')
    parser.add_argument('--benchmark',action='store_true')
    args = parser.parse_args()
    require_kaggle_gpu()
    torch.set_num_threads(2)
    torch.manual_seed(20261022)
    root = competition_root()
    train = pd.read_csv(root/'train.csv',dtype={UID:str})
    series = pd.read_csv(root/'train_series.csv',dtype={UID:str,'SeriesInstanceUID':str})
    # These are official TRAIN images only; never perform backward on test scans.
    uid = train.loc[train[TARGETS].isna().all(axis=1),UID].iloc[0]
    row = next(r for r in select_series(series[series[UID].eq(uid)]) if r is not None)
    stack,audit = read_series(root/'train_series'/uid/row['SeriesInstanceUID'],row['Anatomical_Plane'])
    indices = np.linspace(1,len(stack)-2,12).round().astype(int) if len(stack)>2 else np.zeros(12,int)
    windows = np.stack([stack[np.clip([i-1,i,i+1],0,len(stack)-1)] for i in indices])
    image = torch.from_numpy(windows.astype(np.float32)/255)[None].cuda()
    image = F.interpolate(image.flatten(0,1),size=(224,224),mode='bilinear',align_corners=False).reshape(1,12,3,224,224)
    mean = image.new_tensor([.485,.456,.406])[None,None,:,None,None]
    std = image.new_tensor([.229,.224,.225])[None,None,:,None,None]
    image = (image-mean)/std
    metadata = torch.zeros(1,12,4,device='cuda')
    metadata[0,:,2] = torch.linspace(-1,1,12,device='cuda')
    mask = torch.ones(1,12,dtype=torch.bool,device='cuda')
    model = EvidenceNet(backbone=args.backbone,pretrained=True).cuda()
    original = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    torch.save({'config':model.config,'state_dict':original,'origin':'public ImageNet initialization; no MRI fitting'},
               '/kaggle/working/initialization.pt')
    parameter_name,parameter = next((n,p) for n,p in model.encoder.named_parameters() if p.requires_grad)
    before = parameter.detach().clone()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4)
    scaler = torch.amp.GradScaler('cuda',init_scale=128)
    started = time.time()
    attempts = []
    for attempt in range(3):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):
            logits = model(image,metadata,mask)
            # Numerical update smoke only, not a trained checkpoint or supervision claim.
            loss = supervised_loss(logits,torch.full_like(logits,.2),torch.ones_like(logits))
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = float(parameter.grad.float().norm())
        all_grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(),1))
        scaler.step(optimizer); scaler.update()
        delta = float((parameter.detach()-before).abs().max())
        attempts.append({'attempt':attempt,'parameter':parameter_name,'gradient_norm':grad_norm,
                         'all_grad_norm':all_grad_norm,'update':delta,'amp_scale':scaler.get_scale()})
        print(json.dumps(attempts[-1]),flush=True)
        if np.isfinite(grad_norm) and grad_norm>0 and delta>0:
            break
    if not grad_norm > 0 or not delta > 0:
        raise RuntimeError('Real MRI encoder parameters did not update')
    model.load_state_dict(original,strict=True)  # Discard smoke-step synthetic targets entirely.
    model.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
        prediction = model(image,metadata,mask).sigmoid().float().cpu().numpy()
    manifest = {'format':'independent-real-mri-preflight-v1','kaggle_gpu':torch.cuda.get_device_name(),
                'device_count':torch.cuda.device_count(),
                'backbone':args.backbone,'encoder_parameter':parameter_name,'gradient_norm':grad_norm,
                'max_encoder_update':delta,'strict_reload':True,'training_seconds':time.time()-started,
                'peak_vram_bytes':torch.cuda.max_memory_allocated(),'input_shape':list(image.shape),
                'source_is_training_study':True,'gold_row':False,'test_images_used_for_backward':False,
                'smoke_update_discarded':True,'attempts':attempts,'initialization_sha256':sha256('/kaggle/working/initialization.pt'),
                'finite_predictions':bool(np.isfinite(prediction).all()),'source_audit':audit}
    if args.benchmark:
        benchmarks = []
        for size in (224,336,384):
            resized = F.interpolate(image.flatten(0,1),size=(size,size),mode='bilinear',align_corners=False).reshape(1,12,3,size,size)
            model.train()
            timings = []
            for step in range(3):
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize(); begin = time.time()
                with torch.autocast('cuda',dtype=torch.float16):
                    logits = model(resized,metadata,mask)
                    loss = supervised_loss(logits,torch.full_like(logits,.2),torch.ones_like(logits))
                scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),1)
                scaler.step(optimizer); scaler.update()
                torch.cuda.synchronize(); timings.append(time.time()-begin)
            benchmarks.append({'resolution':size,'windows':12,'steady_study_seconds':float(np.median(timings[1:])),
                               'peak_vram_bytes':torch.cuda.max_memory_allocated(),
                               'measurement':'real training MRI numerical update smoke; weights discarded'})
        model.load_state_dict(original,strict=True)
        manifest['benchmarks'] = benchmarks
        manifest['all_benchmark_updates_discarded'] = True
    write_json('/kaggle/working/preflight.json',manifest)
    print(json.dumps({k:v for k,v in manifest.items() if k!='source_audit'},indent=2),flush=True)


if __name__ == '__main__':
    main()
