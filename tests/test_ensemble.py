import numpy as np
import pytest

from independent_mri.contracts import TARGETS
from independent_mri.ensemble import blend_probabilities,validate_manifest


def manifest():
    return {'format':'independent-evidence-ensemble-v1','targets':TARGETS,'transform':'global_rank',
            'members':[{'checkpoint':'a/best.pt','checkpoint_sha256':'a'*64,'weight':.5},
                       {'checkpoint':'b/best.pt','checkpoint_sha256':'b'*64,'weight':.5}]}


def test_fixed_global_manifest():
    np.testing.assert_allclose(validate_manifest(manifest()),[.5,.5])
    bad = manifest(); bad['members'][0]['weight'] = [1]*12
    with pytest.raises(ValueError): validate_manifest(bad)
    bad = manifest(); bad['members'][1]['checkpoint_sha256'] = 'a'*64
    with pytest.raises(ValueError): validate_manifest(bad)


def test_blend_uses_global_ranks():
    a = np.repeat(np.array([[.1],[.9],[.5]]),12,axis=1)
    b = np.repeat(np.array([[.8],[.3],[.2]]),12,axis=1)
    actual = blend_probabilities([a,b],[.5,.5],'global_rank')
    np.testing.assert_allclose(actual[:,0],[2/3,5/6,1/2])
    np.testing.assert_allclose(blend_probabilities([a,b],[.5,.5],'probability'),(a+b)/2)
    with pytest.raises(ValueError): blend_probabilities([a,b[:2]],[.5,.5],'global_rank')
