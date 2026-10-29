"""CPU statistics on compact Kaggle prediction outputs; no MRI/model inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from independent_mri.contracts import TARGETS,percentile_ranks,sha256,write_json
from independent_mri.ensemble import blend_probabilities
from independent_mri.evaluation import metrics,paired_bootstrap


def load_bank(path):
    with np.load(path,allow_pickle=False) as loaded:
        data = {key:loaded[key] for key in loaded.files}
    n = len(data['uids'])
    if not n or len(set(data['uids']))!=n:
        raise ValueError('Prediction bank has empty/duplicate study IDs')
    if data['probabilities'].shape!=(n,12) or data['targets'].shape!=(n,12) or data['weights'].shape!=(n,12):
        raise ValueError('Prediction bank shape/target coverage differs')
    p = data['probabilities']
    if not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('Prediction bank contains nonfinite/out-of-range values')
    order = np.argsort(data['uids'])
    return {key:value[order] for key,value in data.items() if value.shape and value.shape[0]==n}


def correlation(a,b):
    if len(a)<2 or np.std(a)<1e-10 or np.std(b)<1e-10:
        return None
    return float(np.corrcoef(a,b)[0,1])


def compare(control,treatment,replicates=1000):
    for key in ('uids','targets','weights'):
        if key=='uids':
            matches = np.array_equal(control[key],treatment[key])
        else:
            # Prediction banks can serialize the same soft-label table as
            # float32 or float64. Accept only sub-micro rounding noise; do not
            # let an actual label/weight mismatch through.
            matches = np.allclose(control[key],treatment[key],rtol=0,atol=1e-7,equal_nan=True)
        if not matches:
            raise ValueError(f'Control/treatment {key} are not the same cohort and held labels')
    if 'groups' in control and ('groups' not in treatment or not np.array_equal(control['groups'],treatment['groups'])):
        raise ValueError('Prediction banks disagree on duplicate-report groups')
    y,w = control['targets'],control['weights']
    a,b = control['probabilities'],treatment['probabilities']
    per_target = {}
    for j,target in enumerate(TARGETS):
        take = (w[:,j]>0)&np.isfinite(y[:,j])
        binary = (y[take,j]>=.5).astype(float)
        rank = None if np.std(a[:,j])<1e-10 or np.std(b[:,j])<1e-10 else float(spearmanr(a[:,j],b[:,j]).statistic)
        per_target[target] = {'probability_correlation_all_studies':correlation(a[:,j],b[:,j]),
                              'rank_correlation_all_studies':rank,
                              'report_residual_correlation_known_cells':correlation(binary-a[take,j],binary-b[take,j]),
                              'control_std':float(np.std(a[:,j])),'treatment_std':float(np.std(b[:,j]))}
    result = {'studies':len(y),'control':metrics(y,a,w),'treatment':metrics(y,b,w),
              'paired_uncertainty':paired_bootstrap(y,a,b,w,replicates=replicates,groups=control.get('groups')),
              'per_target_complementarity':per_target,'fixed_blends':{},
              'scope':'Development diagnostics only; not clinical validation or a submission score',
              'selection':'No automatic checkpoint/target-weight selection or submission'}
    for transform in ('probability','global_rank'):
        p = blend_probabilities([a,b],[.5,.5],transform)
        result['fixed_blends'][transform] = {'weights':[.5,.5],'metrics':metrics(y,p,w),
            'paired_uncertainty_vs_control':paired_bootstrap(
                y,a,p,w,replicates=replicates,groups=control.get('groups')),
            'caveat':'Percentile ranks are not calibrated probabilities' if transform=='global_rank'
                     else 'Report-derived targets are not clinical ground truth'}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--control',required=True)
    parser.add_argument('--treatment',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--replicates',type=int,default=1000)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if not output.is_relative_to(Path(__file__).resolve().parent/'private'):
        raise ValueError('Development artifacts must remain in ignored private/')
    result = compare(load_bank(args.control),load_bank(args.treatment),args.replicates)
    result['input_sha256'] = {'control':sha256(args.control),'treatment':sha256(args.treatment)}
    write_json(output,result)
    print(json.dumps({'output':str(output),'control_auc':result['control']['macro_roc_auc'],
                      'treatment_auc':result['treatment']['macro_roc_auc'],
                      'paired_uncertainty':result['paired_uncertainty']},indent=2))


if __name__=='__main__': main()
