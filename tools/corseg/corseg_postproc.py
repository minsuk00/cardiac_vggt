"""Anatomical post-processing pipeline, EXTRACTED VERBATIM from CorSeg-CineSAX_en.py
(section 6, lines 317-491 of the upstream file).

WHY a copy: the upstream script imports PyQt6 at module top level, so it cannot be imported
headlessly. This module is a byte-for-byte lift of the three post-processing steps + the
violation detector, with only the imports re-added. Do NOT edit the logic.

Labels: 0=background, 1=LV myocardium, 2=LV cavity, 3=RV cavity.
"""
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from collections import defaultdict

from scipy import ndimage as sp_ndimage
HAS_SCIPY = True

# 6.  Anatomical Post-Processing Pipeline
# ══════════════════════════════════════════════════════════════
#
# Label definition:  0=Background  1=LV Myocardium  2=LV Cavity  3=RV
#
# Anatomical priors:
#   · Each structure should form a single connected region per slice
#   · LV Cavity is fully enclosed by LV Myocardium
#   · No background holes enclosed by cardiac structures

def detect_violations(mask: np.ndarray) -> Dict[str, bool]:
    stats: Dict[str, bool] = {
        "has_fragment": False,
        "has_containment_violation": False,
        "has_gap": False,
    }
    if not HAS_SCIPY:
        return stats

    struct = sp_ndimage.generate_binary_structure(2, 1)

    # -- Fragments: any label with >1 connected component --
    for lv in (1, 2, 3):
        binary = (mask == lv)
        if not binary.any():
            continue
        _, n_cc = sp_ndimage.label(binary)
        if n_cc > 1:
            stats["has_fragment"] = True
            break

    # -- Containment violation: LV Cavity touches Background or RV --
    lv_cav = (mask == 2)
    if lv_cav.any():
        non_lv = (mask == 0) | (mask == 3)
        non_lv_dil = sp_ndimage.binary_dilation(non_lv, structure=struct)
        if (lv_cav & non_lv_dil).any():
            stats["has_containment_violation"] = True

    # -- Gap: background holes enclosed by cardiac structures --
    cardiac = (mask > 0)
    if cardiac.any():
        filled = sp_ndimage.binary_fill_holes(cardiac)
        if (filled & ~cardiac).any():
            stats["has_gap"] = True
        else:
            lvm = (mask == 1)
            rv  = (mask == 3)
            if lvm.any() and rv.any():
                lvm_dil = sp_ndimage.binary_dilation(lvm, structure=struct)
                rv_dil  = sp_ndimage.binary_dilation(rv, structure=struct)
                if (lvm_dil & rv_dil & (mask == 0)).any():
                    stats["has_gap"] = True

    return stats


def pp_step1_largest_component(mask: np.ndarray) -> np.ndarray:
    """Step 1: Keep only the largest connected component per label."""
    if not HAS_SCIPY:
        return mask
    result = np.zeros_like(mask)
    for lv in (1, 2, 3):
        binary = (mask == lv)
        if not binary.any():
            continue
        labeled, n_cc = sp_ndimage.label(binary)
        if n_cc <= 1:
            result[binary] = lv
            continue
        sizes = sp_ndimage.sum(binary, labeled, range(1, n_cc + 1))
        largest_id = int(np.argmax(sizes)) + 1
        result[labeled == largest_id] = lv
    return result


def pp_step2_containment(mask: np.ndarray) -> np.ndarray:
    """Step 2: Ensure LV Cavity (2) is fully enclosed by LV Myocardium (1)."""
    if not HAS_SCIPY:
        return mask
    result = mask.copy()
    struct = sp_ndimage.generate_binary_structure(2, 1)

    original_cav = int(np.sum(result == 2))
    if original_cav == 0:
        return result

    max_iter = 50
    min_remaining_frac = 0.5

    for _ in range(max_iter):
        lv_cav = (result == 2)
        non_lv = (result == 0) | (result == 3)
        exposed = lv_cav & sp_ndimage.binary_dilation(non_lv, structure=struct)
        if not exposed.any():
            break
        result[exposed] = 1
        remaining = int(np.sum(result == 2))
        if remaining < original_cav * min_remaining_frac:
            break

    return result


def pp_step3_fill_gaps(mask: np.ndarray) -> np.ndarray:
    """Step 3: Fill background holes enclosed by cardiac structures."""
    if not HAS_SCIPY:
        return mask
    result = mask.copy()
    struct = sp_ndimage.generate_binary_structure(2, 1)

    # -- Part A: Fill enclosed holes --
    cardiac = (result > 0)
    if cardiac.any():
        filled = sp_ndimage.binary_fill_holes(cardiac)
        holes = filled & ~cardiac
        if holes.any():
            hole_labeled, n_holes = sp_ndimage.label(holes)
            for h_id in range(1, n_holes + 1):
                h_mask = (hole_labeled == h_id)
                border = (sp_ndimage.binary_dilation(h_mask, structure=struct,
                                                      iterations=2)
                          & ~h_mask & (result > 0))
                if border.any():
                    counts = np.bincount(result[border], minlength=4)
                    best = int(np.argmax(counts[1:])) + 1 if counts[1:].sum() > 0 else 1
                    result[h_mask] = best
                else:
                    result[h_mask] = 1

    # -- Part B: Fill narrow gaps between LV Myo and RV --
    bg  = (result == 0)
    lvm = (result == 1)
    rv  = (result == 3)
    if bg.any() and lvm.any() and rv.any():
        lvm_adj = sp_ndimage.binary_dilation(lvm, structure=struct) & bg
        rv_adj  = sp_ndimage.binary_dilation(rv,  structure=struct) & bg
        septum_gap = lvm_adj & rv_adj
        if septum_gap.any():
            result[septum_gap] = 1

    return result


def apply_postprocessing(
    mask: np.ndarray,
    steps: Dict[str, bool],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    pre_stats  = detect_violations(mask)
    result     = mask.copy()
    pixels_changed = defaultdict(int)

    if steps.get("step1", False):
        before = result.copy()
        result = pp_step1_largest_component(result)
        pixels_changed["step1"] = int(np.sum(before != result))

    if steps.get("step2", False):
        before = result.copy()
        result = pp_step2_containment(result)
        pixels_changed["step2"] = int(np.sum(before != result))

    if steps.get("step3", False):
        before = result.copy()
        result = pp_step3_fill_gaps(result)
        pixels_changed["step3"] = int(np.sum(before != result))

    post_stats = detect_violations(result)

    return result, {
        "pre": pre_stats,
        "post": post_stats,
        "pixels_changed": dict(pixels_changed),
    }
