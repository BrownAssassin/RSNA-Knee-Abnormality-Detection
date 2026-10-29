import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from independent_mri.contracts import CONFIGS, TARGETS, UID, require_kaggle_gpu, validate_predictions, percentile_ranks
from independent_mri.dicom import orientation, canonical_basis, render_slice, select_series


class Contracts(unittest.TestCase):
    def test_allocations(self):
        self.assertEqual([sum(CONFIGS[k][1]) for k in ('P1','P2','P3')], [32,64,96])

    def test_no_local_training(self):
        with self.assertRaisesRegex(RuntimeError, 'Kaggle only'):
            require_kaggle_gpu()

    def test_submission_integrity(self):
        frame = pd.DataFrame([['a'] + [.1]*12], columns=[UID]+TARGETS)
        validate_predictions(frame, ['a'])
        with self.assertRaises(ValueError):
            validate_predictions(frame, ['b'])
        frame.loc[0,'ACL'] = np.nan
        with self.assertRaises(ValueError):
            validate_predictions(frame, ['a'])

    def test_global_ranks(self):
        values = np.array([[.1,.2],[.3,.1],[.2,.3]])
        np.testing.assert_allclose(percentile_ranks(values)[:,0], [1/3,1,2/3])

    def test_physical_orientation(self):
        for plane, iop in [('Axial',[1,0,0,0,1,0]), ('Coronal',[1,0,0,0,0,-1]), ('Sagittal',[0,1,0,0,0,-1])]:
            self.assertEqual(orientation(iop)[3], plane)
            *_, out_col,out_row,depth,actual = canonical_basis(iop)
            self.assertAlmostEqual(out_col @ out_row,0)
            self.assertAlmostEqual(np.linalg.norm(out_col),1)
            image = np.arange(49).reshape(7,7).astype(np.float32)
            np.testing.assert_allclose(render_slice(image,iop,[1,1],7,7), image, atol=1e-4)

    def test_fluid_is_not_fat(self):
        rows = pd.DataFrame([['s','a','Sagittal',1,0], ['s','b','Sagittal',0,1]],
            columns=[UID,'SeriesInstanceUID','Anatomical_Plane','Fluid_Sensitive','Fat_Suppression'])
        chosen = select_series(rows)
        self.assertEqual(chosen[0]['SeriesInstanceUID'],'a')
        self.assertEqual(chosen[1]['SeriesInstanceUID'],'b')
        self.assertIsNone(chosen[4])


if __name__ == '__main__':
    unittest.main()
