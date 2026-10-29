import hashlib

import numpy as np
import pytest

from independent_mri.distill import validate_exposure


def fixture():
    source = {'uids':np.array(['a','b','c','gold']),
              'partitions':np.array(['train','train','train','gold']),
              'folds':np.array([0,1,2,0]),'groups':np.arange(4)}
    digest = lambda values:hashlib.sha256('|'.join(values).encode()).hexdigest()
    receipt = {'own_teacher_fold':0,'source_supervision_sha256':'source','prediction_sha256':'prediction',
               'training_rows':2,'oof_rows':1,'fitting_uids_sha256':digest(['b','c']),
               'oof_uids_sha256':digest(['a']),
               'preserved_initialization_file_sha256':'c4ad2da51831278066da37165374325ce1beb6c9e3ba2950d0edfbab7d4f8563',
               'duplicate_group_overlap':0,'gold_in_fitting':False}
    return source,receipt


def test_actual_teacher_membership_is_audited():
    source,receipt = fixture()
    validate_exposure(receipt,source,0,'source','prediction')
    receipt['fitting_uids_sha256']='wrong cohort'
    with pytest.raises(ValueError,match='fitting_uids'): validate_exposure(receipt,source,0,'source','prediction')


def test_public_mri_initialization_is_not_clean_oof():
    source,receipt = fixture()
    receipt['preserved_initialization_file_sha256']='public MRI exposure'
    with pytest.raises(ValueError,match='initialization'): validate_exposure(receipt,source,0,'source','prediction')
