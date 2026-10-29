"""Compare released public MRI training membership with our own fixed partitions."""
from pathlib import Path
import json
import numpy as np
from .contracts import sha256,write_json
from .supervision import source_file
from .train import find_unique


def main():
    slug = 'nartaa/rsna-knee-s6fall-weights'
    root = source_file(slug,'README.md').parent
    pool_path = root/'training_code/s6/data/pool_ok.json'
    gold_path = root/'training_code/s6/data/gold58.json'
    trainer_path = root/'training_code/s6/train_s6f0_all.py'
    pool = set(json.loads(pool_path.read_text()))
    released_gold = set(json.loads(gold_path.read_text()))
    with np.load(find_unique('supervision.npz'),allow_pickle=False) as loaded:
        data = {k:loaded[k] for k in loaded.files}
    actual_gold = set(data['uids'][data['gold']])
    if released_gold!=actual_gold:
        raise ValueError('Released Gold exclusions disagree with official expert-row membership')
    upstream = pool-released_gold
    if len(upstream)!=4263 or upstream&actual_gold:
        raise ValueError('Released upstream training membership contradicts checkpoint manifest')
    source = trainer_path.read_text()
    if 'set(pool) != done - set(gold)' not in source or 'gold_in_training' not in source:
        raise ValueError('Expected all-data pool / Gold exclusion assertions not found in published trainer')
    seen = np.isin(data['uids'],list(upstream))
    seen_groups = set(data['groups'][seen])
    group_seen = np.isin(data['groups'],list(seen_groups))
    eligible = ~np.isin(data['partitions'],['gold','excluded_gold_duplicate'])
    clean = (~seen)&(~group_seen)&eligible
    counts = {role:int((clean&(data['partitions']==role)).sum()) for role in ['train','monitor','comparison']}
    np.savez_compressed('/kaggle/working/s6_exposure.npz',uids=data['uids'],upstream_seen=seen,
                        upstream_group_seen=group_seen,unexposed_eligible=clean,partitions=data['partitions'])
    manifest = {'format':'independent-public-s6-mri-exposure-audit-v1','source':'https://www.kaggle.com/datasets/'+slug,
                'released_cache_pool_count':len(pool),'released_expert_exclusions_match_official':True,
                'upstream_training_studies':len(upstream),'upstream_gold_overlap':0,
                'published_trainer_excludes_gold':True,'this_is_not_a_retraining_verification':True,
                'eligible_cases_not_in_released_mri_pool':int((eligible&~seen).sum()),
                'eligible_unseen_after_duplicate_group_exclusion':int(clean.sum()),
                'unseen_by_own_partition':counts,'pool_file_sha256':sha256(pool_path),
                'published_trainer_sha256':sha256(trainer_path),'exposure_bank_sha256':sha256('/kaggle/working/s6_exposure.npz'),
                'diagnostic':'Warm report-monitor/comparison scores are generally exposed through public pretraining. '
                             'Only comparison cases not seen by either upstream or our fitting groups are clean MRI holdouts; '
                             'this small, source-selected cohort is not representative clinical validation.',
                'gold_selection_caveat':'Original public model and many label releases reused Gold for development'}
    write_json('/kaggle/working/s6_exposure_audit.json',manifest)
    print(json.dumps(manifest,indent=2),flush=True)


if __name__=='__main__': main()
