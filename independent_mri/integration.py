"""Real cached-study integration checks on Kaggle; no qualified model produced."""
from __future__ import annotations

import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

from .contracts import require_kaggle_gpu,sha256,write_json
from .data import StudyDataset,collate_studies,discover_caches
from .model import EvidenceNet,supervised_loss
from .train import find_unique,move


def main():
    require_kaggle_gpu()
    torch.set_num_threads(2); torch.manual_seed(20261022)
    files,receipts = discover_caches()
    path = find_unique('supervision.npz')
    with np.load(path,allow_pickle=False) as loaded:
        supervision = {k:loaded[k] for k in loaded.files}
    lookup = {uid:i for i,uid in enumerate(supervision['uids'])}
    for uid,file in files.items():
        audit = json.loads(file.with_suffix('.json').read_text())
        if audit['StudyInstanceUID']!=uid or sha256(file)!=audit['cache_sha256']:
            raise RuntimeError('Source-to-cache UID/hash integrity failure')
    training = [lookup[uid] for uid in sorted(files) if supervision['partitions'][lookup[uid]]=='train'][:8]
    data = StudyDataset(files,supervision,training,'P1','L3',True,12)
    loader = DataLoader(data,batch_size=1,num_workers=2,pin_memory=True,collate_fn=collate_studies)
    model = EvidenceNet(pretrained=True).cuda()
    original = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    parameter_name,parameter = next((n,p) for n,p in model.encoder.named_parameters() if p.requires_grad)
    before = parameter.detach().clone()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=2e-5)
    scaler = torch.amp.GradScaler('cuda',init_scale=128)
    model.train(); optimizer.zero_grad(set_to_none=True)
    actual = 0
    for batch in loader:
        batch = move(batch,'cuda')
        if batch['weights'].sum()<=0: continue
        with torch.autocast('cuda',dtype=torch.float16):
            prediction = model(batch['images'],batch['metadata'],batch['mask'])
            loss = supervised_loss(prediction,batch['targets'],batch['weights'])
        scaler.scale(loss/len(data)).backward(); actual += 1
    scaler.unscale_(optimizer)
    gradient = float(parameter.grad.norm())
    norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(),1))
    scaler.step(optimizer); scaler.update()
    delta = float((parameter-before).abs().max())
    if not np.isfinite(norm) or not gradient>0 or not delta>0:
        raise RuntimeError('Real multi-plane cached training MRI did not update the encoder')
    model.load_state_dict(original,strict=True)
    model.eval()
    evaluation = StudyDataset(files,supervision,[lookup[uid] for uid in sorted(files)],'P1','L1',False)
    eval_loader = DataLoader(evaluation,batch_size=1,num_workers=2,pin_memory=True,collate_fn=collate_studies)
    started,finished,windows = time.time(),0,0
    with torch.no_grad():
        for batch in eval_loader:
            batch = move(batch,'cuda')
            with torch.autocast('cuda',dtype=torch.float16):
                p = model(batch['images'],batch['metadata'],batch['mask']).sigmoid().float()
            if p.shape!=(1,12) or not torch.isfinite(p).all():
                raise RuntimeError('Multi-plane integration inference failed')
            finished += 1; windows += int(batch['mask'].sum())
            if finished%25==0:
                print(json.dumps({'progress':finished,'total':len(evaluation),'elapsed_seconds':time.time()-started}),flush=True)
    receipt = {'format':'independent-200-study-multiplane-integration-v1','studies':finished,
               'hash_verified_caches':len(files),'all_finite':True,'silent_substitutions':0,
               'kaggle_gpu':torch.cuda.get_device_name(),'elapsed_inference_seconds':time.time()-started,
               'windows_evaluated':windows,'encoder_parameter':parameter_name,'encoder_gradient_norm':gradient,
               'encoder_max_update':delta,'training_smoke_studies':actual,'gold_in_backward':False,
               'all_smoke_updates_discarded':True,'strict_reload':True,
               'trained_candidate_produced':False,'qualified_accuracy_claim':False,
               'supervision_sha256':sha256(path),'cache_receipts_sha256':[sha256(p) for p in receipts]}
    write_json('/kaggle/working/integration_receipt.json',receipt)
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__': main()
