"""Target-specific multiscale spatial MIL with learned pathology evidence branches."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class EvidenceNet(nn.Module):
    def __init__(self, backbone='convnext_small.fb_in22k_ft_in1k', pretrained=False,
                 attention='M4', branches='C4', width=128, freeze_early=True):
        super().__init__()
        import timm
        self.config = dict(backbone=backbone, attention=attention, branches=branches, width=width,
                           freeze_early=freeze_early)
        self.encoder = timm.create_model(backbone, pretrained=pretrained, features_only=True,
                                         out_indices=(2,3))
        channels = self.encoder.feature_info.channels()
        self.projections = nn.ModuleList(nn.Conv2d(c, width, 1) for c in channels)
        self.plane = nn.Embedding(3, width)
        self.sequence = nn.Embedding(4, width)  # Fluid-sensitive and fat-suppressed are separate bits.
        self.position = nn.Sequential(nn.Linear(3,width), nn.GELU(), nn.Linear(width,width))
        self.target_queries = nn.Parameter(torch.randn(12,width) * .02)
        self.spatial_queries = nn.Parameter(torch.randn(12,width) * .02)
        self.scale_gate = nn.Parameter(torch.zeros(12))
        self.baseline_weight = nn.Parameter(torch.randn(12,width) * .02)
        self.baseline_bias = nn.Parameter(torch.zeros(12))
        self.focal = nn.Sequential(nn.Conv1d(width*2,width,3,padding=1), nn.GELU(),
                                    nn.Conv1d(width,width,3,padding=1,groups=width))
        # RF 7 versus focal RF 5. Every kernel offset is exercised by the sampled
        # 2/3-window blocks; untrained wide-dilation offsets would be unsafe at inference.
        self.diffuse = nn.Sequential(nn.Conv1d(width,width,5,padding=2), nn.GELU(),
                                      nn.Conv1d(width,width,3,padding=1))
        self.branch_weight = nn.Parameter(torch.randn(3,12,width) * .02)
        self.target_gates = nn.Parameter(torch.zeros(12,3))
        with torch.no_grad():
            self.target_gates[[0,1,2,3,10,11],0] = 1
            self.target_gates[[4,5,6,7,8,9],1] = 1
        # Small initial residual, but not zero: every branch can receive gradients immediately.
        self.residual_scale = nn.Parameter(torch.full((12,), -2.0))
        self.drop = nn.Dropout(.15)
        self.freeze_early = freeze_early
        if freeze_early:
            for name, parameter in self.encoder.named_parameters():
                if name.startswith(('stem', 'stages_0', 'stages_1')):
                    parameter.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_early:
            for name, module in self.encoder.named_children():
                if name.startswith(('stem','stages_0','stages_1')):
                    module.eval()
        return self

    def encode(self, images, metadata):
        batch, windows = images.shape[:2]
        pools,spatials = [],[]
        flat = images.flatten(0,1)
        for chunk in flat.split(16):
            maps = self.encoder(chunk)
            projected = [projection(feature) for projection,feature in zip(self.projections,maps)]
            pools.append(projected[-1].mean((-2,-1)))
            if self.config['attention'] in ('M3','M4'):
                attended = []
                for feature in projected:
                    tokens = feature.flatten(2).transpose(1,2)
                    scores = torch.einsum('ndc,tc->ntd',tokens,self.spatial_queries)/tokens.shape[-1]**.5
                    attended.append(torch.einsum('ntd,ndc->ntc',scores.float().softmax(-1).to(tokens.dtype),tokens))
                spatial_chunk = attended[-1]
                if self.config['attention']=='M4':
                    gate = self.scale_gate.sigmoid()[None,:,None]
                    spatial_chunk = gate*attended[0]+(1-gate)*attended[1]
                spatials.append(spatial_chunk)
        pooled = torch.cat(pools).reshape(batch,windows,-1)
        plane = metadata[...,0].long()
        sequence = metadata[...,1].long()
        position = metadata[...,2].float()
        embed = self.plane(plane) + self.sequence(sequence) + self.position(
            torch.stack((position, position.sin(), position.cos()),dim=-1))
        pooled = pooled + embed
        if self.config['attention'] in ('M1','M2'):
            spatial = pooled[:,:,None,:].expand(-1,-1,12,-1)
        else:
            spatial = torch.cat(spatials).reshape(batch,windows,12,-1)
            spatial = spatial + embed[:,:,None,:]
        return pooled, spatial

    def aggregate(self, spatial, pooled, mask):
        if self.config['attention'] == 'M1':
            weights = mask[:,None,:].float().expand(-1,12,-1)
            weights = weights / weights.sum(-1,keepdim=True).clamp_min(1)
        else:
            scores = torch.einsum('bwd,td->btw', pooled, self.target_queries) / pooled.shape[-1]**.5
            scores = scores.masked_fill(~mask[:,None,:], -1e4)
            weights = scores.softmax(-1)
        return torch.einsum('btw,bwtd->btd',weights,spatial), weights

    def forward(self, images, metadata, mask, return_attention=False):
        if (~mask.any(1)).any():
            raise ValueError('Study has no valid MRI windows')
        pooled, spatial = self.encode(images,metadata)
        global_evidence, attention = self.aggregate(spatial,pooled,mask)
        base = (self.drop(global_evidence)*self.baseline_weight).sum(-1) + self.baseline_bias
        if self.config['branches'] == 'C1':
            return (base, attention) if return_attention else base
        focal, diffuse = torch.zeros_like(pooled), torch.zeros_like(pooled)
        # Keep temporal operations within a sequence; never bridge unrelated acquisitions.
        for batch in range(len(images)):
            for slot in range(5):
                indices = ((metadata[batch,:,3].long() == slot) & mask[batch]).nonzero().flatten()
                if indices.numel() == 0:
                    continue
                values = pooled[batch,indices]
                cuts = [0,len(indices)]
                if metadata.shape[-1]>4 and len(indices)>1:
                    gaps = (metadata[batch,indices[1:],4]-metadata[batch,indices[:-1],4]).abs()
                    cuts = [0]+(gaps>8).nonzero().flatten().add(1).tolist()+[len(indices)]
                for start,end in zip(cuts[:-1],cuts[1:]):
                    local = values[start:end]
                    differences = torch.cat((torch.zeros_like(local[:1]),local[1:]-local[:-1]),dim=0)
                    focal[batch,indices[start:end]] = self.focal(torch.cat((local,differences),-1).T[None])[0].T.to(focal.dtype)
                diffuse[batch,indices] = self.diffuse(values.T[None])[0].T.to(diffuse.dtype)
        local_evidence,_ = self.aggregate(spatial+focal[:,:,None,:],pooled+focal,mask)
        broad_evidence,_ = self.aggregate(spatial+diffuse[:,:,None,:],pooled+diffuse,mask)
        evidence = torch.stack((local_evidence,broad_evidence,global_evidence),1)
        branch_logits = (self.drop(evidence)*self.branch_weight[None]).sum(-1).transpose(1,2)
        gate = self.target_gates.softmax(-1)
        if self.config['branches'] == 'C2':
            gate = gate*0 + torch.tensor([1,0,0],device=images.device)
        elif self.config['branches'] == 'C3':
            gate = gate*0 + torch.tensor([0,1,0],device=images.device)
        logits = base + self.residual_scale.sigmoid() * (branch_logits*gate[None]).sum(-1)
        return (logits,attention) if return_attention else logits


def supervised_loss(logits, targets, weights, positive_weight=None, study_weight=None):
    if not torch.isfinite(logits).all():
        raise FloatingPointError('Nonfinite MRI model output')
    denominator = weights.sum()
    if denominator <= 0:
        raise ValueError('Batch contains no known supervision')
    cells = F.binary_cross_entropy_with_logits(logits.float(),targets.float(),reduction='none',
                                               pos_weight=positive_weight)*weights
    if study_weight is not None:
        # A per-study scalar INSIDE numerator and denominator cancels for batch
        # size one. Apply duplicate-group discounts OUTSIDE that normalization.
        per_study = cells.sum(-1)/weights.sum(-1).clamp_min(1e-8)
        return (per_study*study_weight).mean()
    return cells.sum()/denominator
