"""CPU-only missing-view diagnostics for already-exported prediction banks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from independent_mri.contracts import TARGETS
from independent_mri.evaluation import metrics, metrics_by_missing_views, paired_bootstrap


ROOT = Path(__file__).resolve().parent
BANKS = {
    'independent_p1_control': ROOT/'private/training_outputs/main_raw/control-p1/comparison_predictions.npz',
    'independent_p1_pathology': ROOT/'private/training_outputs/main_raw/pathology-p1/comparison_predictions.npz',
    'independent_p3_control': ROOT/'private/training_outputs/warm_raw/warm-control-p3/comparison_predictions.npz',
    'independent_p3_pathology': ROOT/'private/training_outputs/warm_raw/warm-pathology-p3/comparison_predictions.npz',
    'public_224_plain': ROOT/'private/public_benchmark/224_plain_comparison.npz',
    'public_384_plain': ROOT/'private/public_benchmark/384_plain_comparison.npz',
    'public_384_published_mirror': ROOT/'private/public_benchmark/384_published_mirror_comparison.npz',
}


def sha256(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_bank(name: str,path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f'{name} prediction bank missing: {path}')
    with np.load(path,allow_pickle=False) as archive:
        required={'uids','probabilities','targets','weights','missing_views','groups'}
        absent=required-set(archive.files)
        if absent:
            raise ValueError(f'{name} missing arrays: {sorted(absent)}')
        bank={key:archive[key] for key in required}
    n=len(bank['uids'])
    if (bank['probabilities'].shape!=(n,len(TARGETS)) or
            bank['targets'].shape!=(n,len(TARGETS)) or
            bank['weights'].shape!=(n,len(TARGETS)) or
            bank['missing_views'].shape!=(n,) or bank['groups'].shape!=(n,)):
        raise ValueError(f'{name} prediction/label/group coverage mismatch')
    if len(np.unique(bank['uids']))!=n:
        raise ValueError(f'{name} has duplicate UIDs')
    if not np.isfinite(bank['probabilities']).all() or np.any((bank['probabilities']<0)|(bank['probabilities']>1)):
        raise ValueError(f'{name} contains invalid probabilities')
    if not np.issubdtype(bank['missing_views'].dtype,np.integer) or np.any(bank['missing_views']<0):
        raise ValueError(f'{name} has invalid missing-view counts')
    return bank


def paired_representation_check(control: dict,treatment: dict,label: str) -> None:
    for key in ('uids','weights','groups','missing_views'):
        if not np.array_equal(control[key],treatment[key]):
            raise ValueError(f'{label} pair is not aligned on {key}')
    if not np.allclose(control['targets'],treatment['targets'],rtol=1e-7,atol=1e-7,equal_nan=True):
        raise ValueError(f'{label} pair has different labels')


def pair_report(control: dict,treatment: dict,label: str) -> dict:
    paired_representation_check(control,treatment,label)
    masks={'all':np.ones(len(control['uids']),dtype=bool),
           'complete':control['missing_views']==0,
           'any_missing':control['missing_views']>0}
    for count in np.unique(control['missing_views']):
        masks[f'missing_count_{int(count)}']=control['missing_views']==count
    report={}
    for stratum,mask in masks.items():
        indices=np.flatnonzero(mask)
        if not len(indices):
            report[stratum]={'studies':0,'paired_metrics':None,'cluster_bootstrap':None}
            continue
        base=metrics(control['targets'][indices],control['probabilities'][indices],control['weights'][indices])
        new=metrics(treatment['targets'][indices],treatment['probabilities'][indices],treatment['weights'][indices])
        delta=(new['macro_roc_auc']-base['macro_roc_auc']
               if base['macro_roc_auc'] is not None and new['macro_roc_auc'] is not None else None)
        uncertainty=None
        # Overall paired uncertainty is reported here; sparse subgroups remain
        # descriptive to avoid presenting unstable tiny-stratum intervals.
        if stratum=='all' and len(np.unique(control['groups'][indices]))>=2:
            uncertainty=paired_bootstrap(control['targets'][indices],control['probabilities'][indices],
                         treatment['probabilities'][indices],control['weights'][indices],
                         replicates=1000,seed=20261022,groups=control['groups'][indices])
        report[stratum]={'studies':int(len(indices)),
                         'duplicate_report_groups':int(len(np.unique(control['groups'][indices]))),
                         'control_macro_roc_auc':base['macro_roc_auc'],
                         'treatment_macro_roc_auc':new['macro_roc_auc'],
                         'treatment_minus_control_macro_roc_auc':delta,
                         'per_target_delta':{
                             target:(new['per_target'][target]['roc_auc']-base['per_target'][target]['roc_auc']
                                     if new['per_target'][target]['roc_auc'] is not None and
                                        base['per_target'][target]['roc_auc'] is not None else None)
                             for target in TARGETS},
                         'cluster_bootstrap':uncertainty}
    return {'pair':label,'missing_view_semantics':
            'Shared only after exact equality check of the representation-specific missing-view count.',
            'strata':report}


def main() -> None:
    banks={name:load_bank(name,path) for name,path in BANKS.items()}
    analyses={}
    for name,bank in banks.items():
        analyses[name]={'artifact_sha256':sha256(BANKS[name]),'studies':int(len(bank['uids'])),
                        'missing_view_metrics':metrics_by_missing_views(
                            bank['targets'],bank['probabilities'],bank['missing_views'],
                            bank['weights'],bank['groups'])}
    pairs=[
        ('independent_p1_control','independent_p1_pathology','P1 control vs pathology'),
        ('independent_p3_control','independent_p3_pathology','P3 control vs pathology'),
    ]
    paired={label:pair_report(banks[control],banks[treatment],label)
            for control,treatment,label in pairs}
    report={
        'format':'independent-mri-missing-view-diagnostics-v1',
        'target_order':list(TARGETS),
        'cohort':'583-study report-held comparison banks; report-derived labels, not clinical truth',
        'data_handling':'CPU-only metrics over previously exported compact prediction arrays; no images/reports loaded',
        'cross_representation_caveat':(
            'Missing-view counts are representation-specific. Public benchmark and independent model strata '
            'are reported separately; paired subgroup deltas are emitted only for matched control/pathology pairs.'),
        'models':analyses,'matched_pairs':paired,
    }
    destination=ROOT/'private/missing_view_diagnostics_20261022.json'
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(destination),'models':{
        name:{'studies':value['studies'],
              'missing_counts':{k:v['studies'] for k,v in value['missing_view_metrics']['by_missing_view_count'].items()},
              'overall_macro_auc':value['missing_view_metrics']['overall']['metrics']['macro_roc_auc']}
        for name,value in analyses.items()},
        'paired':{name:{stratum:data['treatment_minus_control_macro_roc_auc']
                        for stratum,data in value['strata'].items()}
                  for name,value in paired.items()}},indent=2))


if __name__=='__main__':
    main()
