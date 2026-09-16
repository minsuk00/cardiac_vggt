import os
from pathlib import Path
import tempfile
import unittest

import nibabel as nib
import numpy as np
import torch

from dangi.align import align_nifti, load_model, translate_slices
from dangi.augment import GRID, augment
from dangi.config import Config
from dangi.data import centers_from_labels, make_split, normalize, preprocess, read_case
from dangi.model import DangiCenterNet
from dangi.train import save_checkpoint, ssd_loss
from dataclasses import asdict


class GeometryTests(unittest.TestCase):
    def test_all_augmentations_move_image_and_target_together(self):
        y, x = np.mgrid[:192, :192]
        center = np.array([82., 106.])
        image = np.exp(-((x-center[0])**2+(y-center[1])**2)/18).astype(np.float32)
        self.assertEqual(len(GRID), 189)
        for index in range(len(GRID)):
            warped, target = augment(image, center, index)
            measured = np.array([(warped*x).sum(), (warped*y).sum()]) / warped.sum()
            np.testing.assert_allclose(measured, target, atol=.04)

    def test_translation_sign_axes_and_zero_fill(self):
        source = np.zeros((1, 192, 192), np.float32)
        source[0, 40, 60] = 1
        moved = translate_slices(source, [[7, -4]], order=0)
        self.assertEqual(moved[0, 36, 67], 1)
        restored = translate_slices(moved, [[-7, 4]], order=0)
        np.testing.assert_array_equal(restored, source)
        self.assertEqual(translate_slices(source, [[200, 0]]).sum(), 0)

    def test_propagation_and_invalid_stack(self):
        labels = np.zeros((5, 20, 20), np.uint8)
        labels[1, 3:5, 7:9] = 3
        labels[3, 8:10, 12:14] = 3
        centers, valid = centers_from_labels(labels)
        np.testing.assert_allclose(centers[0], [7.5, 3.5])
        np.testing.assert_allclose(centers[4], [12.5, 8.5])
        np.testing.assert_array_equal(valid, [False, True, False, True, False])
        with self.assertRaises(ValueError):
            centers_from_labels(np.zeros_like(labels))

    def test_resample_crop_pad_preserves_physical_landmark(self):
        data = np.zeros((220, 180, 2), np.float32)
        data[100, 60, 1] = 3
        affine = np.diag([1.5625, 1.5625, 10., 1.])
        volume = nib.Nifti1Image(data, affine)
        slices, out_affine = preprocess(volume, order=0)
        z, y, x = np.argwhere(slices == 3)[0]
        np.testing.assert_allclose(out_affine @ [x, y, z, 1], affine @ [100, 60, 1, 1])
        self.assertEqual(slices.shape, (2, 192, 192))
        np.testing.assert_array_equal(normalize(np.ones_like(slices)), np.zeros_like(slices))

    def test_anisotropic_resampling_and_orientation(self):
        data = np.zeros((96, 192, 1), np.float32)
        data[40, 100, 0] = 3
        affine = np.array([[0, -1.5625, 0, 20], [3.125, 0, 0, -30],
                           [0, 0, 8, 5], [0, 0, 0, 1.]])
        slices, out_affine = preprocess(nib.Nifti1Image(data, affine))
        self.assertEqual(slices[0, 100, 80], 3)
        np.testing.assert_allclose(out_affine @ [80, 100, 0, 1], affine @ [40, 100, 0, 1])
        np.testing.assert_allclose(np.linalg.norm(out_affine[:3, :3], axis=0), [1.5625, 1.5625, 8])


class ModelTests(unittest.TestCase):
    def test_forward_backward_checkpoint_and_nifti(self):
        device = os.environ.get('DANGI_TEST_DEVICE', 'cpu')
        torch.manual_seed(7)
        model = DangiCenterNet().to(device)
        inputs = torch.rand(2, 1, 192, 192, device=device)
        expected = torch.tensor([[90., 100.], [80., 95.]], device=device)
        output = model(inputs)
        self.assertEqual(tuple(output.shape), (2, 2))
        self.assertEqual(model.fc[0].in_features, 51840)
        loss = ssd_loss(output, expected)
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.parameters()))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer.step()
        reference = model(inputs).detach().cpu()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'test.pt'
            save_checkpoint(path, dict(model=model.state_dict(), config=asdict(Config())))
            loaded, config = load_model(path, device)
            torch.testing.assert_close(loaded(inputs).cpu(), reference)
            # Known constant output makes the end-to-end image shift predictable.
            with torch.no_grad():
                loaded.fc[-1].weight.zero_()
                loaded.fc[-1].bias.copy_(torch.tensor([80., 100.], device=device))
            raw = np.zeros((192, 192, 2, 2), np.float32)
            raw[80, 100, :, :] = 17
            affine = np.diag([1.5625, 1.5625, 8., 1.])
            source = nib.Nifti1Image(raw, affine)
            corrected, metadata = align_nifti(loaded, source, config)
            np.testing.assert_allclose(corrected.get_fdata()[96, 96], 17)
            self.assertEqual(corrected.shape, raw.shape)
            np.testing.assert_allclose(metadata['translations_xy'][0], [[16, -4], [16, -4]])
            nib.save(corrected, Path(temporary) / 'aligned.nii.gz')
            reread = nib.load(Path(temporary) / 'aligned.nii.gz')
            np.testing.assert_allclose(reread.affine, affine)


class ACDCIntegrationTests(unittest.TestCase):
    root = Path(__file__).resolve().parents[1] / 'ACDC/database'

    @unittest.skipUnless((root / 'training/patient001').exists(), 'ACDC not installed')
    def test_real_acdc_and_patient_split(self):
        split = make_split(self.root)
        self.assertEqual([len(split[k]) for k in ('train', 'val', 'test')], [80, 10, 10])
        self.assertEqual(len(set(sum(split.values(), []))), 100)
        self.assertEqual(split, make_split(self.root))
        image = self.root / 'training/patient001/patient001_frame01.nii.gz'
        slices, centers, valid, affine = read_case(image)
        self.assertEqual(slices.shape, (10, 192, 192))
        self.assertTrue(np.isfinite(centers).all())
        self.assertTrue(valid.any())
        self.assertGreaterEqual(slices.min(), 0)
        self.assertLessEqual(slices.max(), 1)
        self.assertEqual(affine.shape, (4, 4))


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
