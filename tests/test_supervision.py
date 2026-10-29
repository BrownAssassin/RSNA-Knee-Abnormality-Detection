import unittest
import numpy as np
import pandas as pd
from independent_mri.contracts import TARGETS,UID
from independent_mri.supervision import parse_source,fuse,grouped_partitions
from independent_mri.data import window_plan,windows_from_arrays,collate_studies


class Supervision(unittest.TestCase):
    def test_explicit_unk_is_not_negative(self):
        frame = pd.DataFrame({UID:['a','b'],'ACL':[.01,.01],'ACL__verdict':['UNK','NO']})
        p,s,c = parse_source(frame,['a','b'],'pilkwang')
        self.assertEqual(s[0,0],'unmentioned')
        self.assertEqual(c[0,0],0)
        self.assertEqual(s[1,0],'negative')
        self.assertTrue(np.isnan(p[:,1:]).all())

    def test_missing_and_unmentioned_weight_zero(self):
        p = np.full((7,4,12),np.nan,dtype=np.float32)
        s = np.full(p.shape,'missing',dtype='<U12')
        c = np.zeros_like(p)
        p[:4,0,0] = .5; s[:4,0,0] = 'unmentioned'
        p[:4,1,0] = .9; s[:4,1,0] = 'positive'; c[:4,1,0] = .8
        targets,weights,agreement,reliability = fuse(p,s,c,np.array([True,True,False,False]))
        self.assertTrue((weights[:,0]==0).all())
        self.assertTrue((targets[:,0]==.5).all())
        self.assertTrue((weights[:,1,0]>0).all())
        self.assertTrue((weights[:,:,1:]==0).all())

    def test_numeric_prior_cannot_override_both_unmentioned(self):
        p = np.full((7,2,12),np.nan,dtype=np.float32)
        s = np.full(p.shape,'missing',dtype='<U12')
        c = np.zeros_like(p)
        p[:2,:,0]=.5; s[:2,:,0]='unmentioned'
        p[2:4,:,0]=.01; s[2:4,:,0]='negative'; c[2:4,:,0]=.98
        targets,weights,agreement,reliability = fuse(p,s,c,np.array([True,False]))
        self.assertTrue((weights[:,:,0]==0).all())

    def test_gold_duplicates_excluded(self):
        text = 'The anterior cruciate ligament is intact with a moderate joint effusion and no acute fracture.'
        frame = pd.DataFrame({UID:['a','b','c'],'Report':[text,text,'Different acquisition report without a duplicate.']})
        for target in TARGETS: frame[target] = [1.,np.nan,np.nan]
        groups,partition,folds,gold = grouped_partitions(frame)
        self.assertEqual(groups[0],groups[1])
        self.assertEqual(partition[0],'gold')
        self.assertEqual(partition[1],'excluded_gold_duplicate')
        self.assertNotEqual(groups[1],groups[2])

    def test_broad_coverage(self):
        self.assertEqual(window_plan(100,26)[0],0)
        self.assertEqual(window_plan(100,26)[-1],99)
        self.assertEqual(len(window_plan(100,26)),26)

    def test_declared_source_grid(self):
        # Every channel must come from the declared 8-frame P1 sagittal-fluid grid.
        stack = np.broadcast_to(np.arange(100,dtype=np.uint8)[:,None,None],(100,8,8)).copy()
        audit = {'slots':[{'plane':'Sagittal','fluid':1,'fat_suppression':0,
                          'sources':[{'coordinate_mm':float(i)} for i in range(100)]}] + [None]*4}
        images,metadata,mask = windows_from_arrays({'slot_0':stack},audit,'P1')
        self.assertEqual(images.shape,(8,3,224,224))
        self.assertTrue(mask.all())
        np.testing.assert_allclose(metadata[:,2].numpy()[[0,-1]],[-1,1])


if __name__=='__main__': unittest.main()
