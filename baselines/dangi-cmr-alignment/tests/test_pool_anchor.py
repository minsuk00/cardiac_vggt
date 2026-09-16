import unittest

import numpy as np

from dangi.align import anchor_point
from dangi.config import Config


class AnchorTests(unittest.TestCase):
    centers = np.array([[80., 100.], [84., 98.], [90., 96.]], np.float32)

    def test_image_center_is_paper_default(self):
        np.testing.assert_array_equal(anchor_point(self.centers, None), [96, 96])

    def test_reference_slice_anchor_zeroes_that_slice_and_keeps_relative_shifts(self):
        shifts_ref = anchor_point(self.centers, 1) - self.centers
        shifts_img = anchor_point(self.centers, None) - self.centers
        np.testing.assert_array_equal(shifts_ref[1], [0, 0])
        # Same per-slice shifts up to one whole-stack constant.
        np.testing.assert_allclose(shifts_ref - shifts_ref[0], shifts_img - shifts_img[0])

    def test_explicit_anchor(self):
        np.testing.assert_array_equal(anchor_point(self.centers, (10, 20)), [10, 20])


if __name__ == '__main__':
    unittest.main()
