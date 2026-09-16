from itertools import product

import numpy as np
from scipy.ndimage import affine_transform


# Full Cartesian grid, generated on demand rather than saved as augmented files.
GRID = tuple(product((-0.1, 0.0, 0.1), (-0.1, 0.0, 0.1),
                     (-30, -20, -10, 0, 10, 20, 30), (0.9, 1.0, 1.1)))


def augment(image, center_xy, index):
    """Scale, rotate about (96,96), then translate; image is (row=y,col=x).

    Positive rotation is from +x toward +y (clockwise on a displayed image).
    scipy samples using the inverse affine; centre targets use the forward map.
    """
    tx, ty, degrees, scale = GRID[index]
    angle = np.deg2rad(degrees)
    forward = scale * np.array([[np.cos(angle), -np.sin(angle)],
                               [np.sin(angle), np.cos(angle)]])
    origin = np.array([image.shape[1] / 2, image.shape[0] / 2])
    offset = origin - forward @ origin + np.array([tx, ty]) * image.shape[::-1]
    inverse = np.linalg.inv(forward)
    swap = np.array([[0, 1], [1, 0]])
    warped = affine_transform(image, swap @ inverse @ swap,
                              offset=swap @ (-inverse @ offset), order=1,
                              mode="constant", cval=0, prefilter=False)
    target = forward @ center_xy + offset
    return warped.astype(np.float32), target.astype(np.float32)
