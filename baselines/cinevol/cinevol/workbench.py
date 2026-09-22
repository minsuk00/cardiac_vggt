"""CPU-friendly notebook helpers; viewing exports does not load a CUDA model."""
import json
from pathlib import Path
import nibabel as nib
import numpy as np
from .data import resolve


def plot_losses(run):
    import matplotlib.pyplot as plt
    p = Path(run) / "losses.jsonl"
    rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
    steps = [r["step"] for r in rows]
    axes[0].plot(steps, [r["total"] for r in rows], label="Total fitting objective")
    axes[0].plot(steps, [r["mae"] for r in rows], label="Observed-pixel MAE")
    for k in ("tv", "bias", "correction", "jacobian_cardiac", "jacobian_respiratory"):
        axes[1].plot(steps, [r[k] for r in rows], label=k)
    for ax in axes:
        ax.set_xlabel("Optimizer step")
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    axes[0].set_title("Per-subject fitting losses")
    axes[1].set_title("Unweighted regularization terms")
    fig.tight_layout()
    return fig


def load_results(run):
    run = Path(run).resolve()
    meta = json.loads((run / "subject.json").read_text())
    manifest = meta["source_manifest"]
    result = {"run": run, "metadata": meta,
              "states": json.loads((run / "reconstruction/states.json").read_text())}
    result["output_image"] = nib.load(run / "reconstruction/reference_state.nii.gz")
    result["output"] = result["output_image"].get_fdata(dtype=np.float32)
    result["cine"] = nib.load(run / "reconstruction/cine_end_expiration.nii.gz").get_fdata(dtype=np.float32)
    for key in ("before", "reference", "reference_cine"):
        if not meta.get(key):
            continue
        image = nib.load(resolve(manifest, meta[key]))
        target = result["output_image"]
        if key == "before" and (image.shape != target.shape or not np.allclose(image.affine, target.affine, atol=1e-4)):
            from nibabel.processing import resample_from_to
            image = resample_from_to(image, (target.shape, target.affine), order=1)
        if image.shape[:3] != target.shape or not np.allclose(image.affine, target.affine, atol=1e-4):
            raise ValueError(f"{key} must share the output grid for a meaningful comparison")
        result[key] = image.get_fdata(dtype=np.float32)
    return result


def plot_comparison(results, plane="XZ", index=None, frame=None):
    import matplotlib.pyplot as plt
    axes = {"XY": 2, "XZ": 1, "YZ": 0}
    axis = axes[plane]
    output = results["output"] if frame is None else results["cine"][..., frame]
    index = output.shape[axis] // 2 if index is None else index
    panels = []
    if "before" in results:
        panels.append(("Asynchronous input (fixed)", results["before"]))
    panels.append(("CiNeVol output", output))
    reference = results.get("reference") if frame is None else None
    if frame is not None and "reference_cine" in results:
        phases = results["metadata"].get("reference_cine_cardiac_states")
        phase = results["states"]["cardiac_states"][frame]
        if phases is not None:
            matches = np.flatnonzero(np.isclose(phases, phase, atol=1e-6))
            if len(matches):
                reference = results["reference_cine"][..., matches[0]]
    if reference is not None:
        panels.append(("Ground truth / supplied reference", reference))
    reference_window = reference if reference is not None else output
    low, high = np.percentile(reference_window, [1, 99])
    fig, axs = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4), squeeze=False)
    spacing = np.linalg.norm(results["output_image"].affine[:3, :3], axis=0)
    remaining = [i for i in range(3) if i != axis]
    for ax, (title, volume) in zip(axs[0], panels):
        ax.imshow(np.take(volume, index, axis=axis).T, origin="lower", cmap="gray", vmin=low, vmax=high,
                  aspect=spacing[remaining[1]] / spacing[remaining[0]])
        ax.set_title(title)
        ax.axis("off")
    tag = "SMOKE TEST — not a trained benchmark result" if results["states"]["smoke"] else results["states"]["protocol"]
    phase = results["states"]["reference_cardiac_state"] if frame is None else results["states"]["cardiac_states"][frame]
    fig.suptitle(f"{tag}\n{plane} reformat, cardiac state={phase:.3f}, respiration={results['states']['respiratory_state']:.3f}")
    fig.tight_layout()
    return fig


def result_browser(results):
    import ipywidgets as w
    import matplotlib.pyplot as plt
    from IPython.display import display
    plane = w.Dropdown(options=["XY", "XZ", "YZ"], value="XZ", description="Plane")
    index = w.IntSlider(value=results["output"].shape[1] // 2, min=0, max=results["output"].shape[1] - 1, description="Slice")
    frame = w.IntSlider(value=0, min=0, max=results["cine"].shape[3] - 1, description="Cine frame")
    exact = w.Checkbox(value=True, description="Exact reference state")
    output = w.Output()
    def draw(change=None):
        axis = {"XY": 2, "XZ": 1, "YZ": 0}[plane.value]
        index.max = results["output"].shape[axis] - 1
        frame.disabled = exact.value
        with output:
            output.clear_output(wait=True)
            fig = plot_comparison(results, plane.value, index.value, None if exact.value else frame.value)
            display(fig)
            plt.close(fig)
    for widget in (plane, index, frame, exact):
        widget.observe(draw, names="value")
    draw()
    return w.VBox([w.HBox([plane, index]), w.HBox([exact, frame]), output])


def cine_animation(results, plane="XZ", index=None):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from IPython.display import HTML
    axis = {"XY": 2, "XZ": 1, "YZ": 0}[plane]
    volume = results["cine"]
    index = volume.shape[axis] // 2 if index is None else index
    low, high = np.percentile(volume, [1, 99])
    fig, ax = plt.subplots(figsize=(4, 4))
    artist = ax.imshow(np.take(volume[..., 0], index, axis=axis).T, origin="lower", cmap="gray", vmin=low, vmax=high)
    ax.axis("off")
    def update(t):
        artist.set_data(np.take(volume[..., t], index, axis=axis).T)
        ax.set_title(f"{'SMOKE TEST | ' if results['states']['smoke'] else ''}CiNeVol cardiac state {results['states']['cardiac_states'][t]:.2f}")
        return [artist]
    animation = FuncAnimation(fig, update, frames=volume.shape[3], interval=120)
    html = animation.to_jshtml()
    plt.close(fig)
    return HTML(html)
