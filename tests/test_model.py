"""CPU-only dummy-tensor forward contracts; no fitting and no MRI inference."""
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from independent_mri.model import EvidenceNet,supervised_loss
from independent_mri.infer import mirror_inputs


class FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3,8,1)
        self.final = nn.Conv2d(8,16,1)
        self.feature_info = SimpleNamespace(channels=lambda:[8,16])

    def forward(self,x):
        a = self.stem(x)
        return [a,self.final(a)]


class ModelContracts(unittest.TestCase):
    def model(self,attention='M4',branches='C4'):
        with patch.dict(sys.modules,{'timm':SimpleNamespace(create_model=lambda *a,**kw:FakeEncoder())}):
            return EvidenceNet(attention=attention,branches=branches,width=16).eval()

    def test_variants_are_finite(self):
        images = torch.rand(1,5,3,4,4)
        metadata = torch.tensor([[[0,2,-1,0,0],[0,2,0,0,3],[0,2,1,0,6],
                                  [1,3,-1,2,30],[1,3,1,2,50]]],dtype=torch.float32)
        mask = torch.ones(1,5,dtype=torch.bool)
        for attention in ('M1','M2','M3','M4'):
            with torch.no_grad():
                output = self.model(attention)(images,metadata,mask)
            self.assertEqual(output.shape,(1,12))
            self.assertTrue(torch.isfinite(output).all())

    def test_padding_does_not_change_prediction(self):
        torch.manual_seed(1)
        model = self.model()
        image = torch.rand(1,2,3,4,4)
        md = torch.zeros(1,2,5)
        md[0,1,4]=3
        with torch.no_grad():
            p = model(image,md,torch.ones(1,2,dtype=torch.bool))
            q = model(torch.cat((image,torch.rand(1,1,3,4,4)),1),
                      torch.cat((md,torch.zeros(1,1,5)),1),torch.tensor([[True,True,False]]))
        torch.testing.assert_close(p,q)

    def test_mirror_is_involution(self):
        image = torch.arange(5*3*2*2).reshape(5,3,2,2).float()
        md = torch.tensor([[0,2,-1,0,4],[0,2,1,0,8],[1,3,0,2,30],
                           [2,3,-1,4,40],[2,3,1,4,44]],dtype=torch.float32)
        a,b = mirror_inputs(image,md)
        c,d = mirror_inputs(a,b)
        torch.testing.assert_close(image,c,atol=1e-5,rtol=1e-5)
        torch.testing.assert_close(md,d)
        # Anatomical target identities are not exchanged by a display reflection.

    def test_sagittal_mirror_renormalizes_destination_channels(self):
        raw = torch.tensor([.1,.4,.8]).view(1,3,1,1).expand(2,3,2,2).clone()
        mean = torch.tensor([.485,.456,.406]).view(1,3,1,1)
        std = torch.tensor([.229,.224,.225]).view(1,3,1,1)
        images = (raw-mean)/std
        md = torch.tensor([[0,2,-1,0,4],[0,2,1,0,8]],dtype=torch.float32)
        mirrored,_ = mirror_inputs(images,md)
        expected = (raw.flip(0).flip(1)-mean)/std
        torch.testing.assert_close(mirrored,expected)

    def test_early_encoder_frozen(self):
        model = self.model()
        self.assertTrue(all(not p.requires_grad for p in model.encoder.stem.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.encoder.final.parameters()))

    def test_duplicate_discount_does_not_cancel(self):
        logits = torch.zeros(1,12)
        targets = torch.ones(1,12)
        weights = torch.ones(1,12)
        base = supervised_loss(logits,targets,weights,study_weight=torch.ones(1))
        discounted = supervised_loss(logits,targets,weights*.5,study_weight=torch.full((1,),.5))
        torch.testing.assert_close(discounted,base*.5)


if __name__=='__main__': unittest.main()
