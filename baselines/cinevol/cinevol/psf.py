import math
import torch

FWHM_TO_SIGMA = 1 / math.sqrt(8 * math.log(2))


def training_samples(batch, noise):
    fwhm = batch["spacing"].clone()
    fwhm[:, :2] *= 1.2
    local = noise * (fwhm * FWHM_TO_SIGMA)[:, None]
    return batch["xyz"][:, None] + torch.einsum("bij,buj->bui", batch["rotation"], local)
