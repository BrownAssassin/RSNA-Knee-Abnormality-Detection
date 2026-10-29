from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score,precision_score,recall_score,brier_score_loss
from .contracts import TARGETS


def metrics(labels,predictions,weights=None):
    labels,predictions = np.asarray(labels),np.asarray(predictions)
    if labels.shape != predictions.shape or not np.isfinite(predictions).all():
        raise ValueError('Invalid evaluation prediction coverage/values')
    known = np.isfinite(labels) if weights is None else (np.asarray(weights)>0)&np.isfinite(labels)
    binary = labels>=.5
    result,aucs,prs = {},[],[]
    for j,target in enumerate(TARGETS):
        take = known[:,j]
        y,p = binary[take,j],predictions[take,j]
        valid = len(y)>1 and len(np.unique(y))==2
        auc = float(roc_auc_score(y,p)) if valid else None
        ap = float(average_precision_score(y,p)) if valid else None
        if valid:
            aucs.append(auc); prs.append(ap)
        result[target] = {'known':int(take.sum()),'positive':int(y.sum()),'roc_auc':auc,'pr_auc':ap,
                          'precision_at_0_5':float(precision_score(y,p>=.5,zero_division=0)) if len(y) else None,
                          'recall_at_0_5':float(recall_score(y,p>=.5,zero_division=0)) if len(y) else None,
                          'brier_vs_report_binary':float(brier_score_loss(y,p)) if len(y) else None,
                          'prediction_std':float(np.std(p)) if len(p) else None}
    return {'macro_roc_auc':float(np.mean(aucs)) if aucs else None,
            'macro_pr_auc':float(np.mean(prs)) if prs else None,'targets_evaluable':len(aucs),
            'per_target':result,'calibration_caveat':'Report-derived targets are not clinical ground truth'}


def metrics_by_missing_views(labels,predictions,missing_views,weights=None,groups=None):
    """Report metrics overall and by the representation's missing-view count.

    ``missing_views`` is representation-specific. Results from two different
    preprocessing pipelines must not be compared within strata unless their
    view masks have first been proven identical.
    """
    labels,predictions = np.asarray(labels),np.asarray(predictions)
    missing_views = np.asarray(missing_views)
    if labels.ndim!=2 or len(missing_views)!=len(labels):
        raise ValueError('Missing-view coverage differs from study coverage')
    if (not np.issubdtype(missing_views.dtype,np.integer) or
            np.any(missing_views<0)):
        raise ValueError('Missing-view counts must be nonnegative integers')
    if weights is not None and np.asarray(weights).shape!=labels.shape:
        raise ValueError('Evaluation weight shape differs from label coverage')
    if groups is not None:
        groups=np.asarray(groups)
        if groups.shape!=(len(labels),):
            raise ValueError('Duplicate-group coverage differs from study coverage')

    def summarize(indices):
        return {
            'studies':int(len(indices)),
            'duplicate_report_groups':int(len(np.unique(groups[indices]))) if groups is not None else None,
            'metrics':metrics(labels[indices],predictions[indices],
                              None if weights is None else np.asarray(weights)[indices]),
        }

    result={'overall':summarize(np.arange(len(labels))),
            'by_missing_view_count':{},
            'complete_vs_any_missing':{}}
    for count in np.unique(missing_views):
        selected=np.flatnonzero(missing_views==count)
        result['by_missing_view_count'][str(int(count))]=summarize(selected)
    for name,selected in (
        ('complete',np.flatnonzero(missing_views==0)),
        ('any_missing',np.flatnonzero(missing_views>0)),
    ):
        if len(selected):
            result['complete_vs_any_missing'][name]=summarize(selected)
    result['missing_view_definition']='Count supplied by this model representation; not cross-pipeline comparable without parity.'
    return result


def paired_bootstrap(labels,control,treatment,weights=None,replicates=1000,seed=20261022,groups=None):
    labels,control,treatment = map(np.asarray,(labels,control,treatment))
    if labels.shape!=control.shape or labels.shape!=treatment.shape:
        raise ValueError('Paired bootstrap arrays are not aligned')
    known = np.isfinite(labels) if weights is None else (np.asarray(weights)>0)&np.isfinite(labels)
    if not np.isfinite(control).all() or not np.isfinite(treatment).all():
        raise ValueError('Paired bootstrap has invalid predictions')
    clusters = None
    if groups is not None:
        groups = np.asarray(groups)
        if len(groups)!=len(labels): raise ValueError('Bootstrap group coverage differs')
        clusters = [np.flatnonzero(groups==group) for group in np.unique(groups)]
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(replicates):
        index = (rng.integers(0,len(labels),len(labels)) if clusters is None else
                 np.concatenate([clusters[i] for i in rng.integers(0,len(clusters),len(clusters))]))
        differences = []
        for j in range(12):
            take = index[known[index,j]]
            y = labels[take,j]>=.5
            if len(y)>1 and np.unique(y).size==2:
                differences.append(float(roc_auc_score(y,treatment[take,j])-roc_auc_score(y,control[take,j])))
        if len(differences)==12:
            deltas.append(float(np.mean(differences)))
    return {'valid_replicates':len(deltas),'delta_ci95':np.quantile(deltas,[.025,.975]).tolist() if deltas else None,
            'positive_fraction':float(np.mean(np.array(deltas)>0)) if deltas else None,
            'unit':'paired study' if clusters is None else 'paired duplicate-report group cluster',
            'independent_units':len(labels) if clusters is None else len(clusters),
            'caveat':'Repeated-development and upstream-pretraining exposure caveats still apply'}
