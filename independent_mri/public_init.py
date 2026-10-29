"""Audit an independently accessible, explicitly permitted MRI encoder initializer."""
from __future__ import annotations

import json
from pathlib import Path
import re
import torch

from .contracts import require_kaggle_gpu,sha256,write_json
from .model import EvidenceNet

PARENT_SHA256 = '0c7297d58f061ab4e5d7c55046f51fa3e8934928046f0647a0c7854e98371c12'
SOURCE = 'https://www.kaggle.com/datasets/nartaa/rsna-knee-s6fall-weights'


def canonical_key(key):
    match = re.search(r'(?:stem|stages)(?:\.|_)',key)
    key = key[match.start():] if match else key
    return re.sub(r'^(stem|stages)_(\d+)\.',r'\1.\2.',key)


def main():
    require_kaggle_gpu()
    candidates = list(Path('/kaggle/input').glob('*/E_prior_s42_fall.pt'))
    candidates += list(Path('/kaggle/input').glob('datasets/*/*/E_prior_s42_fall.pt'))
    if len(candidates)!=1: raise ValueError('Expected one published S6 all-data checkpoint')
    path = candidates[0]
    if sha256(path)!=PARENT_SHA256: raise ValueError('Public MRI checkpoint hash mismatch')
    # Safe weights-only unpickling; do not execute arbitrary public pickle code.
    checkpoint = torch.load(path,map_location='cpu',weights_only=True)
    dictionaries = [checkpoint]+[v for v in checkpoint.values() if isinstance(v,dict)]
    def score(value):
        return sum(isinstance(v,torch.Tensor) and bool(re.search('stem|stages',str(k))) for k,v in value.items())
    state = max(dictionaries,key=score)
    if score(state)==0: raise ValueError('No identifiable public encoder tensor state')
    source = {}
    for name,value in state.items():
        if not isinstance(value,torch.Tensor): continue
        key = canonical_key(name)
        if key in source: raise ValueError(f'Ambiguous public parameter mapping {key}')
        source[key] = value
    model = EvidenceNet(pretrained=False).cuda()
    mapped = {}
    for name,value in model.encoder.state_dict().items():
        key = canonical_key(name)
        if key not in source or source[key].shape!=value.shape:
            raise ValueError(f'Public encoder cannot strictly initialize {name}, shape {tuple(value.shape)}')
        if not torch.isfinite(source[key]).all(): raise ValueError('Nonfinite public initializer')
        mapped[name] = source[key]
    model.encoder.load_state_dict(mapped,strict=True)
    exported = {'state_dict':{'encoder.'+k:v.detach().cpu() for k,v in model.encoder.state_dict().items()},
                'origin':'public MRI-task pretraining, NOT clean ImageNet-only initialization',
                'parent_sha256':PARENT_SHA256,'source':SOURCE,'license':'CC-BY-NC-4.0',
                'exposure':'4263 report-supervised studies; Gold excluded from original gradients but reused for selection'}
    target = Path('/kaggle/working/initialization_s6.pt')
    torch.save(exported,target)
    readme = (path.parent/'README.md').read_text()
    if not all(term in readme.lower() for term in ('fine-tune','non-commercial')):
        raise ValueError('Expected explicit adaptation/use terms missing from attached public release')
    (Path('/kaggle/working')/'S6_SOURCE_TERMS.md').write_text(readme,encoding='utf-8')
    manifest = {'format':'independent-public-mri-initializer-audit-v1','source':SOURCE,
                'parent_checkpoint_sha256':PARENT_SHA256,'source_terms_sha256':sha256(path.parent/'README.md'),
                'initialization_sha256':sha256(target),'encoder_tensors_matched':len(mapped),
                'strict_encoder_load':True,'public_pickle_weights_only':True,'new_mri_fitting_performed':False,
                'weights_license':'CC-BY-NC-4.0; credit nartaa and Heliosli architecture',
                'inference_or_training_recipe_copied':False,'private_teammate_resources_used':False,
                'oai_data_used_in_this_public_checkpoint':False,'upstream_report_studies':4263,
                'gold_used_in_upstream_training':False,'gold_used_in_upstream_selection':True,
                'monitor_comparison_mri_upstream_exposure':'LIKELY; adaptation diagnostics, not clean OOF',
                'permitted_for_own_oof_teachers':False}
    write_json('/kaggle/working/initialization_s6_manifest.json',manifest)
    print(json.dumps(manifest,indent=2),flush=True)


if __name__=='__main__': main()
