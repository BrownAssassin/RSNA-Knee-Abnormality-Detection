"""CPU aggregation of compact supervision arrays; no image/model computation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .contracts import TARGETS,sha256,write_json


def audit(bank,manifest):
    targets,weights = bank['targets'],bank['weights']
    n = len(bank['uids'])
    sources = [source['source'] for source in manifest['sources']]
    if len(set(bank['uids']))!=n or targets.shape!=(4,n,12) or weights.shape!=targets.shape:
        raise ValueError('Supervision UID/target shape integrity failure')
    if not np.isfinite(targets).all() or not np.isfinite(weights).all() or (weights<0).any():
        raise ValueError('Nonfinite/negative supervision')
    if ((targets<0)|(targets>1)).any():
        raise ValueError('Targets outside probability range')
    status,probabilities = bank['source_status'],bank['source_p']
    if status.shape!=(len(sources),n,12) or probabilities.shape!=status.shape:
        raise ValueError('Source-aware target coverage differs')
    supported = np.isin(status,['positive','negative'])&np.isfinite(probabilities)
    mention = supported[0]|supported[1]
    if (weights[:,~mention]>0).any():
        raise ValueError('Unmentioned cells acquired gradient supervision')
    excluded = np.isin(bank['partitions'],['gold','excluded_gold_duplicate'])
    if (weights[:,excluded]>0).any():
        raise ValueError('Gold or its duplicate groups carry fitting weights')
    protected_groups_with_multiple_roles = 0
    for group in np.unique(bank['groups']):
        rows = bank['groups']==group
        roles = bank['partitions'][rows]
        # Gold and its excluded nongold duplicates intentionally have different
        # audit roles, but both belong to the SAME protected/nonfitting partition.
        partitions = np.where(np.isin(roles,['gold','excluded_gold_duplicate']),'gold_protected',roles)
        if len(np.unique(partitions))!=1 or len(np.unique(bank['folds'][rows]))!=1:
            raise ValueError('Duplicate-report group crosses partition/fold boundary')
        protected_groups_with_multiple_roles += int(len(np.unique(roles))>1)
    fitting = bank['partitions']=='train'
    source_stats,pairs = {},{}
    for s,name in enumerate(sources):
        source_stats[name] = {}
        for j,target in enumerate(TARGETS):
            counts = {kind:int(((status[s,:,j]==kind)&fitting).sum())
                      for kind in ('positive','negative','uncertain','unmentioned','missing')}
            counts['available_probability'] = int((np.isfinite(probabilities[s,:,j])&fitting).sum())
            source_stats[name][target] = counts
        for other in range(s):
            per_target = {}
            for j,target in enumerate(TARGETS):
                known = fitting&supported[s,:,j]&supported[other,:,j]
                conflict = (probabilities[s,:,j]>=.5)!=(probabilities[other,:,j]>=.5)
                count = int(known.sum())
                per_target[target] = {'both_supported':count,'opposite_polarity':int((known&conflict).sum()),
                    'polarity_agreement':float((~conflict[known]).mean()) if count else None}
            pairs[f'{sources[other]}__{name}'] = per_target
    policies = {}
    for policy in range(4):
        policy_stats = {}
        for j,target in enumerate(TARGETS):
            known = fitting&(weights[policy,:,j]>0)
            values = bank['agreement'][known,j]
            policy_stats[target] = {'known':int(known.sum()),
                'positive':int((known&(targets[policy,:,j]>.5)).sum()),
                'negative':int((known&(targets[policy,:,j]<.5)).sum()),
                'coverage_fraction_of_fitting':float(known.sum()/max(fitting.sum(),1)),
                'effective_cell_weight':float(weights[policy,known,j].sum()),
                'mean_source_agreement':float(values.mean()) if len(values) else None,
                'source_conflict_cells':int((known&(bank['agreement'][:,j]<1)).sum())}
        policies[f'L{policy+1}'] = policy_stats
    return {'format':'independent-source-conflict-coverage-audit-v1','fitting_studies':int(fitting.sum()),
            'partition_counts':{str(k):int(v) for k,v in zip(*np.unique(bank['partitions'],return_counts=True))},
            'unique_report_groups':len(np.unique(bank['groups'])),
            'sources':source_stats,'pairwise_source_conflicts_fitting_only':pairs,
            'policy_coverage_agreement_fitting_only':policies,'gold_used_to_select_source_weights':False,
            'unmentioned_supervision_cells':0,'gold_or_near_duplicate_gradient_cells':0,
            'duplicate_groups_crossing_partitions':0,
            'protected_gold_groups_with_multiple_audit_roles':protected_groups_with_multiple_roles,
            'caveat':'Shared reports and dependent hybrid/teacher labels are not independent clinical truth',
            'actual_mri_prediction_quality':'Requires matched trained-model results; not inferred from this audit'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bank',required=True)
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding='utf-8'))
    if sha256(args.bank)!=manifest['supervision_sha256']:
        raise ValueError('Supervision bank differs from its source manifest')
    with np.load(args.bank,allow_pickle=False) as loaded:
        bank = {k:loaded[k] for k in loaded.files}
    result = audit(bank,manifest)
    result['supervision_sha256'] = sha256(args.bank)
    write_json(args.output,result)
    print(json.dumps({'output':args.output,'fitting_studies':result['fitting_studies'],
                      'source_pairs':len(result['pairwise_source_conflicts_fitting_only']),
                      'all_integrity_checks_passed':True},indent=2))


if __name__=='__main__': main()
