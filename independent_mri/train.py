"""Actual independent MRI encoder fitting, exclusively on Kaggle GPUs."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .contracts import TARGETS, require_kaggle_gpu, sha256, write_json
from .data import StudyDataset, collate_studies, discover_caches
from .evaluation import metrics
from .model import EvidenceNet, supervised_loss


def find_unique(filename):
    paths = list(Path('/kaggle/input').glob('*/'+filename))
    paths += list(Path('/kaggle/input').glob('notebooks/*/*/'+filename))
    if len(paths)!=1:
        raise RuntimeError(f'Expected one {filename}, found {paths}')
    return paths[0]


def move(batch,device):
    return {key:(value.to(device,non_blocking=True) if isinstance(value,torch.Tensor) else value)
            for key,value in batch.items()}


@torch.no_grad()
def predict(model,loader,device):
    model.eval()
    uids,ps,ys,ws,missing = [],[],[],[],[]
    for batch in loader:
        batch = move(batch,device)
        with torch.autocast('cuda',dtype=torch.float16):
            p = model(batch['images'],batch['metadata'],batch['mask']).sigmoid().float()
        if not torch.isfinite(p).all():
            raise FloatingPointError('Nonfinite held-study probabilities')
        uids.extend(batch['uid']); ps.append(p.cpu().numpy())
        ys.append(batch['targets'].cpu().numpy()); ws.append(batch['weights'].cpu().numpy())
        for md,mask in zip(batch['metadata'],batch['mask']):
            missing.append(5-len(md[mask,3].unique()))
    return np.array(uids),np.concatenate(ps),np.concatenate(ys),np.concatenate(ws),np.array(missing)


def save_checkpoint(path,model,optimizer,scaler,manifest,ema=None,inference_only=False,loader_generator=None):
    temp = path.with_suffix('.tmp')
    checkpoint = {'config':model.config,'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                  'manifest':manifest}
    if not inference_only:
        checkpoint.update({'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
                           'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
                           'numpy_rng':np.random.get_state(),'python_rng':random.getstate()})
        if ema is not None:
            checkpoint['ema_state_dict'] = {k:v.detach().cpu() for k,v in ema.state_dict().items()}
        if loader_generator is not None:
            checkpoint['loader_rng'] = loader_generator.get_state()
    torch.save(checkpoint,temp)
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name',required=True)
    parser.add_argument('--configuration',choices=['P1','P2','P3'],default='P1')
    parser.add_argument('--policy',choices=['L1','L2','L3','L4'],default='L3')
    parser.add_argument('--attention',choices=['M1','M2','M3','M4'],default='M4')
    parser.add_argument('--branches',choices=['C1','C2','C3','C4','C5'],default='C4')
    parser.add_argument('--epochs',type=int,default=12)
    parser.add_argument('--windows',type=int,default=12)
    parser.add_argument('--accumulation',type=int,default=8)
    parser.add_argument('--device',type=int,default=0)
    parser.add_argument('--seed',type=int,default=20261022)
    parser.add_argument('--teacher-fold',type=int,default=-1)
    parser.add_argument('--backbone',default='convnext_small.fb_in22k_ft_in1k')
    parser.add_argument('--distillation',choices=['T1','T2','T3','T4'],default='T1')
    parser.add_argument('--initialization',choices=['imagenet','public_s6'],default='imagenet')
    args = parser.parse_args()
    if args.teacher_fold>=0 and args.initialization!='imagenet':
        raise ValueError('Public task-pretrained weights cannot be called clean OOF teachers')
    require_kaggle_gpu()
    torch.cuda.set_device(args.device)
    device = torch.device('cuda',args.device)
    torch.set_num_threads(2)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    files,receipts = discover_caches()
    supervision_path = find_unique('supervision.npz')
    with np.load(supervision_path,allow_pickle=False) as loaded:
        supervision = {key:loaded[key] for key in loaded.files}
    if set(files)!=set(supervision['uids']):
        raise ValueError('MRI cache and complete supervision UID coverage differ')
    partitions = supervision['partitions']
    distillation_sha256 = None
    if args.distillation!='T1':
        refined_path = find_unique('refined_supervision.npz')
        with np.load(refined_path,allow_pickle=False) as refined:
            if not np.array_equal(refined['uids'],supervision['uids']):
                raise ValueError('Distillation UID order differs from source supervision')
            selected_policy = int(args.distillation[1:])-1
            train_rows = partitions=='train'
            policy_index = int(args.policy[1:])-1
            supervision['targets'][policy_index,train_rows] = refined['targets'][selected_policy,train_rows]
            supervision['weights'][policy_index,train_rows] = refined['weights'][selected_policy,train_rows]
        distillation_sha256 = sha256(refined_path)
    training = partitions=='train'
    if args.teacher_fold>=0:
        training &= supervision['folds']!=args.teacher_fold
    indices = np.flatnonzero(training)
    policy_index = int(args.policy[1:])-1
    fitting_y = supervision['targets'][policy_index,indices]
    fitting_w = supervision['weights'][policy_index,indices]
    positive = (fitting_y*fitting_w).sum(0)
    negative = ((1-fitting_y)*fitting_w).sum(0)
    positive_cells = ((fitting_y>.5)&(fitting_w>0)).sum(0)
    negative_cells = ((fitting_y<.5)&(fitting_w>0)).sum(0)
    if (positive_cells==0).any() or (negative_cells==0).any():
        raise ValueError('A target has no supervised positives or negatives in this fitting cohort')
    positive_weight = torch.tensor(np.clip(np.sqrt(negative/positive.clip(1)),.5,4),
                                  dtype=torch.float32,device=device)
    # Match the macro target objective without cancelling per-cell confidence.
    # This is derived only from the fitting cohort and is shared by all controls.
    target_mass = fitting_w.sum(0)
    target_balance = np.clip(target_mass.mean()/target_mass.clip(1),.5,4).astype(np.float32)
    supervision['weights'][policy_index,indices] *= target_balance[None,:]
    monitor_indices = np.flatnonzero(partitions=='monitor')
    # Every experiment evaluates against the SAME L1 monitor, not its own preferred label strategy.
    train_data = StudyDataset(files,supervision,indices,args.configuration,args.policy,True,args.windows)
    monitor = StudyDataset(files,supervision,monitor_indices,args.configuration,'L1',False)
    # Separate sampler randomness from architecture-dependent dropout draws.
    # Matched controls therefore see exactly the same study order each epoch.
    shuffle_generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(train_data,batch_size=1,shuffle=True,num_workers=2,pin_memory=True,
                        collate_fn=collate_studies,persistent_workers=False,generator=shuffle_generator)
    val_loader = DataLoader(monitor,batch_size=1,num_workers=2,pin_memory=True,collate_fn=collate_studies)
    initialization_path = find_unique('initialization.pt' if args.initialization=='imagenet' else 'initialization_s6.pt')
    expected_initialization = 'c4ad2da51831278066da37165374325ce1beb6c9e3ba2950d0edfbab7d4f8563'
    upstream_exposure = 'ImageNet only; no declared RSNA MRI exposure'
    public_exposure = None
    if args.initialization=='public_s6':
        init_manifest = json.loads(find_unique('initialization_s6_manifest.json').read_text())
        if init_manifest['parent_checkpoint_sha256']!='0c7297d58f061ab4e5d7c55046f51fa3e8934928046f0647a0c7854e98371c12':
            raise ValueError('Unexpected public MRI parent checkpoint')
        expected_initialization = init_manifest['initialization_sha256']
        upstream_exposure = 'Public S6 saw report-pool MRI; monitor/comparison are adaptation diagnostics, NOT clean OOF'
        with np.load(find_unique('s6_exposure.npz'),allow_pickle=False) as exposure:
            if not np.array_equal(exposure['uids'],supervision['uids']):
                raise ValueError('Public exposure audit is not aligned with the fitting UID table')
            public_exposure = exposure['unexposed_eligible'].copy()
    if sha256(initialization_path)!=expected_initialization:
        raise ValueError('Preserved ImageNet initialization hash mismatch')
    model = EvidenceNet(args.backbone,pretrained=False,attention=args.attention,branches=args.branches).to(device)
    public_initialization = torch.load(initialization_path,map_location='cpu',weights_only=False)
    encoder_state = {name[len('encoder.'):]:value for name,value in public_initialization['state_dict'].items()
                     if name.startswith('encoder.')}
    model.encoder.load_state_dict(encoder_state,strict=True)
    initial_digest = hashlib.sha256()
    for name,value in model.state_dict().items():
        initial_digest.update(name.encode()); initial_digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    full_initialization_sha256 = initial_digest.hexdigest()
    optimizer = torch.optim.AdamW([
        {'params':[p for p in model.encoder.parameters() if p.requires_grad],'lr':2e-5 if args.initialization=='imagenet' else 1e-5},
        {'params':[p for name,p in model.named_parameters() if not name.startswith('encoder.')],'lr':2e-4}],weight_decay=.01)
    scaler = torch.amp.GradScaler('cuda',init_scale=128)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters(): p.requires_grad_(False)
    tracked_name,tracked = next((name,p) for name,p in model.encoder.named_parameters() if p.requires_grad)
    initial = tracked.detach().clone()
    out = Path('/kaggle/working')/args.name
    out.mkdir(parents=True,exist_ok=True)
    started,best,history = time.time(),-math.inf,[]
    base_lrs = [g['lr'] for g in optimizer.param_groups]
    proved_update = False
    def optimizer_step(correction=1):
        nonlocal proved_update
        scaler.unscale_(optimizer)
        if correction!=1:
            for parameter in model.parameters():
                if parameter.grad is not None: parameter.grad.mul_(correction)
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(),1))
        if not math.isfinite(gradient_norm):
            raise FloatingPointError('Nonfinite training gradients; checkpoint not promoted')
        scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        if not proved_update:
            delta = float((tracked.detach()-initial).abs().max())
            if delta<=0:
                raise RuntimeError('MRI encoder did not actually update in this training run')
            proved_update = True
            write_json(out/'encoder_update.json',{'parameter':tracked_name,'max_update':delta,
                       'real_training_mri':True,'kaggle_device':str(device),
                       'gold_in_fitting':False,'test_data_in_backward':False})
        with torch.no_grad():
            for ep,p in zip(ema.parameters(),model.parameters()): ep.lerp_(p.detach(),.005)
            for eb,b in zip(ema.buffers(),model.buffers()): eb.copy_(b)
    for epoch in range(args.epochs):
        train_data.epoch = epoch
        model.train(); optimizer.zero_grad(set_to_none=True)
        epoch_started,total_loss,trained,steps,pending = time.time(),0.,0,0,0
        for step,batch in enumerate(loader):
            batch = move(batch,device)
            if batch['weights'].sum()<=0:
                continue
            with torch.autocast('cuda',dtype=torch.float16):
                logits = model(batch['images'],batch['metadata'],batch['mask'])
                loss = supervised_loss(logits,batch['targets'],batch['weights'],positive_weight,batch['study_weight'])
            scaler.scale(loss/args.accumulation).backward()
            total_loss += float(loss.detach()); trained += 1; pending += 1
            if pending==args.accumulation:
                optimizer_step(); steps += 1; pending = 0
            if trained%200==0:
                print(json.dumps({'experiment':args.name,'epoch':epoch+1,'studies':trained,
                                  'total':len(train_data),'loss':total_loss/trained,
                                  'elapsed_s':time.time()-started}),flush=True)
        if pending:
            optimizer_step(args.accumulation/pending); steps += 1
        uids,p,y,w,missing = predict(ema,val_loader,device)
        score = metrics(y,p,w)
        value = score['macro_roc_auc']
        receipt = {'experiment':args.name,'epoch':epoch+1,'train_loss':total_loss/max(trained,1),
                   'target_order':TARGETS,
                   'optimizer_steps':steps,'monitor':score,'train_rows':len(indices),
                   'epoch_seconds':time.time()-epoch_started,'elapsed_seconds':time.time()-started,
                   'peak_vram_bytes':torch.cuda.max_memory_allocated(device),
                   'configuration':vars(args),'supervision_sha256':sha256(supervision_path),
                   'positive_class_weight_train_only':positive_weight.cpu().tolist(),
                   'fitting_known_positive_cells':positive_cells.tolist(),
                   'fitting_known_negative_cells':negative_cells.tolist(),
                   'target_coverage_balance_train_only':target_balance.tolist(),
                   'trainable_encoder_parameters':sum(p.numel() for p in model.encoder.parameters() if p.requires_grad),
                   'distillation_sha256':distillation_sha256,
                   'cache_receipt_sha256':[sha256(path) for path in receipts],
                   'fitting_uids_sha256':__import__('hashlib').sha256('|'.join(supervision['uids'][indices]).encode()).hexdigest(),
                   'gold_used_for_selection':False,'comparison_used_for_selection':False,
                   'environment':{'torch':torch.__version__,'timm':__import__('timm').__version__,
                                  'numpy':np.__version__,'cuda':torch.version.cuda},
                   'source_sha256':{p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')},
                   'preserved_initialization_file_sha256':expected_initialization,
                   'upstream_mri_exposure':upstream_exposure,
                   'new_fitting_studies_absent_from_upstream_mri_pool':int(public_exposure[indices].sum())
                        if public_exposure is not None else len(indices),
                   'report_comparison_independent_of_pretraining':args.initialization=='imagenet',
                   'full_initialization_state_sha256':full_initialization_sha256,
                   'public_pretraining':'ImageNet-22k -> ImageNet-1k; Apache-2.0' if args.initialization=='imagenet'
                                        else 'MRI S6 initializer; CC-BY-NC-4.0 with nartaa/Heliosli credit',
                   'source':'https://huggingface.co/timm/'+args.backbone if args.initialization=='imagenet'
                            else 'https://www.kaggle.com/datasets/nartaa/rsna-knee-s6fall-weights',
                   'kaggle_gpu':torch.cuda.get_device_name(device)}
        history.append(receipt)
        # Preserve matched-duration comparisons, not only a model-selected best epoch.
        # These compact arrays contain no MRI images or report text.
        np.savez_compressed(out/f'monitor_epoch_{epoch+1:02d}.npz',uids=uids,probabilities=p,
                            targets=y,weights=w,missing_views=missing,
                            groups=supervision['groups'][monitor_indices])
        save_checkpoint(out/'latest.pt',model,optimizer,scaler,receipt,ema=ema,
                        loader_generator=shuffle_generator)
        if value is not None and value>best:
            best = value
            save_checkpoint(out/'best.pt',ema,optimizer,scaler,receipt,inference_only=True)
            write_json(out/'best_manifest.json',receipt)
            np.savez_compressed(out/'monitor_predictions.npz',uids=uids,probabilities=p,targets=y,weights=w,
                                missing_views=missing)
        write_json(out/'history.json',history)
        print(json.dumps({'experiment':args.name,'epoch':epoch+1,'monitor_auc':value,
                          'best':best,'epoch_seconds':receipt['epoch_seconds']}),flush=True)
        for group,base in zip(optimizer.param_groups,base_lrs):
            group['lr'] = base*(.05+.95*(1+math.cos(math.pi*(epoch+1)/args.epochs))/2)
    checkpoint = torch.load(out/'best.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(checkpoint['state_dict'],strict=True)
    final_evaluations = {}
    cohorts = [('comparison',np.flatnonzero(partitions=='comparison')),
               ('gold_reused_diagnostic',np.flatnonzero(partitions=='gold'))]
    if public_exposure is not None:
        clean = np.flatnonzero((partitions=='comparison')&public_exposure)
        if len(clean): cohorts.append(('comparison_upstream_unseen',clean))
    if args.teacher_fold>=0:
        cohorts.append(('own_oof',np.flatnonzero((partitions=='train')&(supervision['folds']==args.teacher_fold))))
    for cohort,cohort_indices in cohorts:
        data = StudyDataset(files,supervision,cohort_indices,args.configuration,'L1',False)
        evaluation_loader = DataLoader(data,batch_size=1,num_workers=2,pin_memory=True,collate_fn=collate_studies)
        uids,p,y,w,missing = predict(model,evaluation_loader,device)
        if cohort=='gold_reused_diagnostic':
            y = supervision['gold_labels'][cohort_indices]
            w = np.isfinite(y).astype(np.float32)
        final_evaluations[cohort] = metrics(y,p,w)
        if cohort=='comparison_upstream_unseen':
            final_evaluations[cohort]['scope_caveat'] = (
                'Small source-selected cohort, report targets not clinical truth; '
                'not representative or adequate evidence of full 12-target generalization')
        final_evaluations[cohort]['missing_view_groups'] = {
            str(k):metrics(y[missing==k],p[missing==k],w[missing==k]) for k in np.unique(missing)}
        np.savez_compressed(out/f'{cohort}_predictions.npz',uids=uids,probabilities=p,targets=y,weights=w,
                            missing_views=missing,groups=supervision['groups'][cohort_indices])
        if cohort=='own_oof':
            if set(supervision['groups'][indices])&set(supervision['groups'][cohort_indices]):
                raise RuntimeError('OOF teacher has a duplicate-report-group training overlap')
            write_json(out/'oof_exposure.json',{'own_teacher_fold':args.teacher_fold,
                       'training_rows':len(indices),'oof_rows':len(cohort_indices),
                       'duplicate_group_overlap':0,'gold_in_fitting':False,
                       'source_supervision_sha256':sha256(supervision_path),
                       'preserved_initialization_file_sha256':expected_initialization,
                       'fitting_uids_sha256':hashlib.sha256('|'.join(supervision['uids'][indices]).encode()).hexdigest(),
                       'oof_uids_sha256':hashlib.sha256('|'.join(uids).encode()).hexdigest(),
                       'checkpoint_sha256':sha256(out/'best.pt'),
                       'prediction_sha256':sha256(out/'own_oof_predictions.npz'),
                       'pretraining':'ImageNet only, no external task teacher',
                       'claims':'Predictions on train-pool groups excluded from this teacher fitting'})
    final = {'checkpoint_sha256':sha256(out/'best.pt'),'best_monitor_auc':best,'strict_reload':True,
             'best_manifest_sha256':sha256(out/'best_manifest.json'),
             'epochs_completed':len(history),'encoder_updated':proved_update,'candidate_submitted':False,
             'final_evaluations':final_evaluations}
    write_json(out/'training_receipt.json',final)
    print(json.dumps(final),flush=True)


if __name__ == '__main__':
    main()
