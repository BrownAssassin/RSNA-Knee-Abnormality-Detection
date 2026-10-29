from types import SimpleNamespace
from unittest.mock import patch
import sys

from independent_mri.public_coatnet import PublicRaptor


def test_published_backbone_grid_conventions():
    calls = []
    def create(name,**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(num_features=8)
    with patch.dict(sys.modules,{'timm':SimpleNamespace(create_model=create)}):
        PublicRaptor(320,'coatnet')
        PublicRaptor(224,'coatnet')
    assert calls[0]['img_size']==320
    assert 'img_size' not in calls[1]
