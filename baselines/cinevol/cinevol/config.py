"""Published settings plus explicitly documented implementation choices."""
import copy
import json
from pathlib import Path


def configuration(profile="invivo", smoke=False):
    paper = json.loads(Path(__file__).with_name("paper_defaults.json").read_text())
    if profile not in paper["profiles"]:
        raise ValueError("Profile must be phantom or invivo")
    c = copy.deepcopy(paper["common"])
    c["loss_weights"].update(paper["profiles"][profile]["loss_weights"])
    c.update(profile=profile, smoke=smoke, backend="torch")
    c["implementation_choices"] = {
        "adam_betas": [0.9, 0.99], "adam_eps": 1e-15,
        "other_weight_decay": 0.0, "scheduler": None,
        "intensity_activation": "softplus", "motion_last_layer_uniform_bound": 1e-4,
        "bias_correction_reduction": "mean_absolute_over_psf_samples",
        "jacobian": "full_branch_map_at_pixel_centres_in_mm",
        "distance_epsilon_mm": 1e-6, "hash_interpolation": "smoothstep",
        "inference_fwhm": "output_voxel_spacing", "phase_endpoint": False,
        "coordinate_padding_mm": 20.0,
    }
    if smoke:
        c["optimization"].update(steps_per_subject=30, batch_observed_pixels=128)
        for key in ("spatial_grid", "spatiotemporal_grids"):
            c[key].update(features_per_level=2, log2_hashmap_size=10,
                          base_resolution=[4, 4, 4], finest_resolution=[12, 12, 8])
        c["spatial_grid"]["levels"] = 2
        c["spatiotemporal_grids"]["levels_each"] = 2
        c["mlps"]["hidden_width"] = 32
        c["mlps"]["cardiac_input_output"] = [16, 3]
        c["mlps"]["respiratory_input_output"] = [16, 3]
        c["mlps"]["intensity_input_output"] = [4, 17]
    return c


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
