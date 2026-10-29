import numpy as np
import pytest

from independent_mri.supervision_audit import audit


def fixture():
    bank = {'uids':np.array(['a','b','c']),'targets':np.full((4,3,12),.5),
            'weights':np.zeros((4,3,12)),'source_status':np.full((7,3,12),'unmentioned'),
            'source_p':np.full((7,3,12),.5),'partitions':np.array(['train','monitor','gold']),
            'groups':np.arange(3),'folds':np.arange(3),'agreement':np.zeros((3,12))}
    manifest = {'sources':[{'source':f's{i}'} for i in range(7)]}
    return bank,manifest


def test_silence_is_not_a_negative():
    bank,manifest = fixture()
    assert audit(bank,manifest)['unmentioned_supervision_cells']==0
    bank['weights'][0,0,0]=1
    with pytest.raises(ValueError,match='Unmentioned'): audit(bank,manifest)


def test_duplicate_report_separation():
    bank,manifest = fixture()
    bank['groups'][1]=bank['groups'][0]
    with pytest.raises(ValueError,match='Duplicate-report'): audit(bank,manifest)


def test_gold_duplicate_roles_share_one_protected_partition():
    bank,manifest = fixture()
    bank['partitions']=np.array(['train','excluded_gold_duplicate','gold'])
    bank['groups'][1]=bank['groups'][2]
    bank['folds'][1]=bank['folds'][2]
    assert audit(bank,manifest)['protected_gold_groups_with_multiple_audit_roles']==1
