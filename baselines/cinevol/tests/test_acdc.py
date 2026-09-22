import json

import nibabel as nib
import numpy as np
import pytest

from cinevol.acdc import corrupt_cine, generate
from cinevol.data import Observations
from cinevol.prepare import prepare


def test_displacement_sign_units_and_supplied_state():
    clean = np.zeros((9, 9, 2, 3), np.float32)
    clean[4, 4, :, :] = 1
    states = np.broadcast_to([0., .5, 1.], (2, 3))
    observed, shifts = corrupt_cine(clean, states, [2., 3., 8.], [4., -6.])
    for t, coordinate in enumerate([(4, 4), (5, 3), (6, 2)]):
        assert observed[coordinate[0], coordinate[1], 0, t] == 1
        assert observed[:, :, 0, t].sum() == 1
    np.testing.assert_array_equal(shifts[0], [[0, 0], [2, -3], [4, -6]])
    np.testing.assert_array_equal(observed[..., 0], clean[..., 0])


def test_acdc_preparation_provenance_and_target_independence(tmp_path, monkeypatch):
    patient = tmp_path / 'patient001'
    patient.mkdir()
    (patient / 'Info.cfg').write_text('ED: 2\nES: 3\n')
    rng = np.random.default_rng(4)
    clean = rng.uniform(1, 10, (9, 10, 4, 5)).astype(np.float32)
    affine = np.array([[0., -2., 0., 12.], [2., 0., 0., -8.], [0., 0., 6., -9.], [0., 0., 0., 1.]])
    nib.save(nib.Nifti1Image(clean, affine), patient / 'patient001_4d.nii.gz')
    out = tmp_path / 'adapted'
    manifest = generate(patient, out, displacement_mm=(2., 0.), voxel_mm=2.)
    data = Observations(manifest)
    assert data.n == clean.size
    assert isinstance(data.arrays['xyz'], np.memmap)
    assert data.n_slices == 4
    assert data.meta['minimum_stack_coverage'] == 1
    assert data.meta['reference_cardiac_state'] == .2
    assert data.meta['protocol'] == 'acdc_single_sax_simulated'
    states = np.load(out / 'respiratory_states.npy')
    for z in range(4):
        for t in range(5):
            select = (data.arrays['slice_index'] == z) & (data.arrays['frame_index'] == t)
            np.testing.assert_allclose(data.arrays['respiratory'][select], states[z, t])
    gt = np.load(out / 'simulation_gt.npz')
    expected_world = np.zeros((4, 5, 3))
    expected_world[..., 1] = 2 * states
    np.testing.assert_allclose(gt['shifts_world_mm'], expected_world)
    reference = nib.load(out / 'reference.nii.gz')
    np.testing.assert_allclose(np.linalg.norm(reference.affine[:3, :3], axis=0), 2.)
    # Exported sparse input is exactly a subset of the full corrupted cine.
    observed = nib.load(out / 'sax_observed.nii.gz').get_fdata()
    before = nib.load(out / 'before_native.nii.gz').get_fdata()
    for z, t in enumerate(gt['selected_frame_indices']):
        np.testing.assert_array_equal(before[:, :, z], observed[:, :, z, t])
    (out / 'reference.nii.gz').unlink()
    (out / 'simulation_gt.npz').unlink()
    assert Observations(manifest).n == clean.size
    # Complete fit/export with the adaptation and both motion branches, even
    # after the reference and ground-truth shifts are removed.
    import importlib
    from cinevol.config import configuration
    from cinevol.reconstruct import reconstruct
    from cinevol.timing import summarize
    fitting = importlib.import_module('cinevol.fit')
    def tiny_config(profile, smoke):
        cfg = configuration(profile, True)
        cfg['optimization'].update(steps_per_subject=2, batch_observed_pixels=16)
        return cfg
    monkeypatch.setattr(fitting, 'configuration', tiny_config)
    run = tmp_path / 'run'
    checkpoint = fitting.fit(manifest, run, profile='invivo', backend='torch', device='cpu', smoke=True, microbatch=8)
    reconstruct(checkpoint, device='cpu', chunk=256)
    state = fitting.load_checkpoint(checkpoint)
    assert any(k.startswith('cardiac_net.') for k in state['model'])
    assert any(k.startswith('respiratory_net.') for k in state['model'])
    assert nib.load(run / 'reconstruction/cine_end_expiration.nii.gz').shape[-1] == 20
    timing = summarize(run)
    assert timing['protocol'] == 'acdc_single_sax_simulated'
    assert timing['simulation_generation_seconds_excluded'] > 0
    # Fixed seed produces exactly the same observations/states.
    second = tmp_path / 'second'
    generate(patient, second, displacement_mm=(2., 0.), voxel_mm=2.)
    np.testing.assert_array_equal(observed, nib.load(second / 'sax_observed.nii.gz').get_fdata())
    with pytest.raises(FileExistsError):
        generate(patient, out)
    # A protocol change cannot silently make a single stack native.
    meta = json.loads(manifest.read_text())
    meta['protocol'] = 'native_full_cine'
    manifest.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='coverage'):
        Observations(manifest)
    spec = json.loads((out / 'acquisitions.json').read_text())
    spec['protocol'] = 'native_full_cine'
    (out / 'acquisitions.json').write_text(json.dumps(spec))
    with pytest.raises(ValueError, match='multiplanar'):
        prepare(out / 'acquisitions.json', out / 'native')


def test_invalid_states_are_not_accepted():
    with pytest.raises(ValueError, match='Respiratory states'):
        corrupt_cine(np.ones((5, 5, 2, 3)), np.ones((2, 3)) * 1.1, [2, 2, 8], [5, 5])
