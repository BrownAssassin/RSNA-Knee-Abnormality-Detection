import numpy as np
import pytest

from independent_mri.evaluation import metrics_by_missing_views, paired_bootstrap
from compare import compare


def bank():
    labels = np.repeat(np.array([[0],[0],[1],[1]],dtype=float),12,axis=1)
    return {'uids':np.array(['a','b','c','d']),'targets':labels,'weights':np.ones_like(labels),
            'probabilities':labels*.8+.1,'groups':np.array(['g0','g0','g1','g1'])}


def test_paired_cluster_unit_and_identical_predictions():
    b = bank()
    result = paired_bootstrap(b['targets'],b['probabilities'],b['probabilities'],b['weights'],
                              replicates=20,groups=b['groups'])
    assert result['independent_units']==2
    assert result['valid_replicates']>0
    assert result['delta_ci95']==[0.,0.]


def test_cohort_mismatch_is_not_silent():
    a,b = bank(),bank()
    b['uids'][0]='z'
    with pytest.raises(ValueError,match='uids'): compare(a,b,replicates=1)


def test_float32_label_roundtrip_is_accepted_but_real_mismatch_is_not():
    a,b = bank(),bank()
    b['targets'] = b['targets'].astype(np.float32)
    result = compare(a,b,replicates=5)
    assert result['studies']==4
    assert 'paired_uncertainty_vs_control' in result['fixed_blends']['global_rank']

    b['targets'][0,0] = 0.25
    with pytest.raises(ValueError,match='targets'):
        compare(a,b,replicates=1)


def test_missing_view_metrics_report_count_and_duplicate_groups():
    labels=np.repeat(np.array([[0],[1],[0],[1]],dtype=float),12,axis=1)
    predictions=labels*.8+.1
    result=metrics_by_missing_views(labels,predictions,np.array([0,0,1,1]),
                                    np.ones_like(labels),np.array(['a','b','a','c']))
    assert result['overall']['studies']==4
    assert result['overall']['duplicate_report_groups']==3
    assert result['by_missing_view_count']['0']['studies']==2
    assert result['by_missing_view_count']['1']['metrics']['macro_roc_auc']==1.0
    assert result['complete_vs_any_missing']['any_missing']['duplicate_report_groups']==2
    assert 'not cross-pipeline comparable' in result['missing_view_definition']


@pytest.mark.parametrize('missing,weights,groups,error',[
    (np.array([0,1]),None,None,'coverage'),
    (np.array([0.,0.,1.,1.]),None,None,'integers'),
    (np.array([0,0,1,1]),np.ones((4,11)),None,'weight shape'),
    (np.array([0,0,1,1]),None,np.array(['only-one']), 'group coverage'),
])
def test_missing_view_metrics_reject_misaligned_strata(missing,weights,groups,error):
    labels=np.repeat(np.array([[0],[1],[0],[1]],dtype=float),12,axis=1)
    with pytest.raises(ValueError,match=error):
        metrics_by_missing_views(labels,labels,missing,weights,groups)
