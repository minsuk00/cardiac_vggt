#!/usr/bin/env python
"""Probe candidate EXTERNAL datasets for validating the VGGT-MRI slice-to-volume model.

What this does (idempotent): download a few representative examples from publicly
accessible cardiac-cine datasets onto a local scratch dir, reconstruct/inspect them to
confirm they match the model's input contract (short-axis cine, multi-slice, cardiac
phases), render preview PNGs, and emit a manifest.json recording exactly what was
verified-downloadable and its parsed dimensions.

Headline targets (the real-time / free-breathing transfer domain we lack):
  - OCMR (ocmr.info)        — direct public S3, ISMRMRD k-space. Has BOTH gated breath-hold
                              SAX cine AND real-time free-breathing (pse) SAX *stacks*.
  - Zenodo 10639392         — paired breath-hold cine vs real-time free-breathing, BART .cfl,
                              but SINGLE SAX slice per subject (2D, not a stack).
  - Zenodo 7621356          — free-running whole-heart k-space (GRE, 4-6 GB/vol). Accessibility
                              checked only (large + contrast/orientation mismatch).

Reuses the vendored OCMR reader at _archive/ocmr/read_ocmr.py.

Run:  micromamba run -n svr python tools/probe_candidate_datasets.py
Outputs: PNGs + manifest.json under result/dataset_probe/ ; downloads cached in
         /tmp/vggt_dataset_probe/ (skip-if-exists).
"""
import json
import os
import subprocess
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "_archive", "ocmr"))
DL = "/tmp/vggt_dataset_probe"
OUT = os.path.join(REPO, "result", "dataset_probe")
os.makedirs(DL, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

OCMR_S3 = "https://ocmr.s3.us-east-2.amazonaws.com"
# Selected OCMR files (from ocmr_data_attributes.csv):
#   fs_0005 — gated breath-hold SAX, single slice, 18 phases  (clean cine demo)
#   fs_0056 — gated breath-hold SAX STACK, 12 slices          (clean 3D-reference-like volume)
#   us_0084 — real-time FREE-BREATHING SAX STACK, 12 sl/128 fr, R~9 (target-domain input)
OCMR_FILES = ["fs_0005_1_5T.h5", "fs_0056_1_5T.h5", "us_0084_1_5T.h5"]


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)


def curl_download(url, path, max_time=2400):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return os.path.getsize(path)
    r = sh(f'curl -sL -C - --max-time {max_time} -o "{path}" "{url}"')
    return os.path.getsize(path) if os.path.exists(path) else -1


def head_size(url):
    r = sh(f'curl -sIL --max-time 30 "{url}"')
    for ln in r.stdout.splitlines():
        if ln.lower().startswith("content-length:"):
            return int(ln.split(":", 1)[1].strip())
    return -1


def cifft(x, axes):
    for a in axes:
        x = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(x, axes=a), axis=a), axes=a)
    return x


def ocmr_dims(path):
    """Header-only dims, cheap (no acquisition loop)."""
    import ismrmrd
    import ismrmrd.xsd

    dset = ismrmrd.Dataset(path, "dataset", create_if_needed=False)
    hdr = ismrmrd.xsd.CreateFromDocument(dset.read_xml_header())
    enc = hdr.encoding[0]
    return dict(
        eNx=enc.encodedSpace.matrixSize.x,
        eNy=enc.encodingLimits.kspace_encoding_step_1.maximum + 1,
        coils=hdr.acquisitionSystemInformation.receiverChannels,
        slices=(enc.encodingLimits.slice.maximum or 0) + 1,
        frames=(enc.encodingLimits.phase.maximum or 0) + 1,
        fov=[enc.encodedSpace.fieldOfView_mm.x, enc.encodedSpace.fieldOfView_mm.y,
             enc.encodedSpace.fieldOfView_mm.z],
        seq=str(hdr.sequenceParameters.sequence_type),
    )


def ocmr_recon_subset(path, fixed="phase", value=0):
    """Memory-safe recon: fill k-space for one fixed phase (-> slice stack) or one fixed
    slice (-> frames), IFFT + SoS coil-combine + crop 2x readout oversampling.
    Returns (image[x,y,N], dims)."""
    import ismrmrd
    import ismrmrd.xsd

    dset = ismrmrd.Dataset(path, "dataset", create_if_needed=False)
    hdr = ismrmrd.xsd.CreateFromDocument(dset.read_xml_header())
    enc = hdr.encoding[0]
    eNx = enc.encodedSpace.matrixSize.x
    eNy = enc.encodingLimits.kspace_encoding_step_1.maximum + 1
    nC = hdr.acquisitionSystemInformation.receiverChannels
    nSl = (enc.encodingLimits.slice.maximum or 0) + 1
    nPh = (enc.encodingLimits.phase.maximum or 0) + 1
    nacq = dset.number_of_acquisitions()
    a0 = None
    for i in range(nacq):
        a = dset.read_acquisition(i)
        if not a.isFlagSet(ismrmrd.ACQ_IS_NOISE_MEASUREMENT):
            a0 = a
            break
    kx_pre = eNx - a0.number_of_samples if a0.center_sample * 2 < eNx else 0
    N = nSl if fixed == "phase" else nPh
    buf = np.zeros((eNx, eNy, nC, N), dtype=np.complex64)
    cnt = 0
    for i in range(nacq):
        a = dset.read_acquisition(i)
        if a.isFlagSet(ismrmrd.ACQ_IS_NOISE_MEASUREMENT):
            continue
        if fixed == "phase":
            if a.idx.phase != value:
                continue
            j = a.idx.slice
        else:
            if a.idx.slice != value:
                continue
            j = a.idx.phase
        buf[kx_pre:, a.idx.kspace_encode_step_1, :, j] = a.data.T
        cnt += 1
    im = cifft(buf, (0, 1))
    sos = np.sqrt((np.abs(im) ** 2).sum(axis=2))
    ro = sos.shape[0]
    return sos[ro // 4: ro // 4 * 3], dict(slices=nSl, frames=nPh, coils=nC, lines=cnt)


def read_cfl(base):
    dims = [int(x) for x in open(base + ".hdr").read().split("\n")[1].split()]
    n = int(np.prod(dims))
    d = np.fromfile(base + ".cfl", dtype=np.float32, count=2 * n)
    return np.squeeze((d[0::2] + 1j * d[1::2]).reshape(dims, order="F"))


def strip(frames, titles, suptitle, path, vmax_q=0.85):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(frames)
    fig, ax = plt.subplots(1, n, figsize=(2.5 * n, 2.9))
    if n == 1:
        ax = [ax]
    for a, fr, t in zip(ax, frames, titles):
        vmax = np.quantile(fr, vmax_q) if fr.max() > 0 else 1
        a.imshow(fr.T, cmap="gray", vmin=0, vmax=vmax)
        a.set_title(t, fontsize=9)
        a.axis("off")
    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def probe_ocmr(manifest):
    csv = os.path.join(DL, "ocmr_data_attributes.csv")
    curl_download(f"{OCMR_S3}/ocmr_data_attributes.csv", csv, max_time=60)
    entry = {"name": "OCMR", "landing": "https://ocmr.info", "access": "public S3 (no login)",
             "license": "CC-BY-NC-4.0", "format": "ISMRMRD .h5 (multi-coil k-space)",
             "attributes_csv_bytes": os.path.getsize(csv) if os.path.exists(csv) else -1,
             "files": []}
    for f in OCMR_FILES:
        path = os.path.join(DL, f)
        nbytes = curl_download(f"{OCMR_S3}/data/{f}", path)
        dims = ocmr_dims(path) if nbytes > 0 else {}
        entry["files"].append({"file": f, "url": f"{OCMR_S3}/data/{f}", "bytes": nbytes,
                               "dims": dims, "verified": nbytes > 0})
    # renders
    cine, _ = ocmr_recon_subset(os.path.join(DL, "fs_0005_1_5T.h5"), "slice", 0)  # single slice -> frames
    idx = np.linspace(0, cine.shape[2] - 1, 6).round().astype(int)
    strip([cine[:, :, p] for p in idx], [f"t={p}" for p in idx],
          "OCMR fs_0005 - gated breath-hold SAX cine (18 phases, SSFP 1.5T)",
          os.path.join(OUT, "ocmr_fs0005_gated_cine.png"))
    stack, sinfo = ocmr_recon_subset(os.path.join(DL, "fs_0056_1_5T.h5"), "phase", 0)
    idx = np.linspace(0, stack.shape[2] - 1, min(8, stack.shape[2])).round().astype(int)
    strip([stack[:, :, z] for z in idx], [f"z={z}" for z in idx],
          f"OCMR fs_0056 - gated SAX STACK (clean), {sinfo['slices']} slices @ phase 0",
          os.path.join(OUT, "ocmr_fs0056_gated_stack.png"))
    rt, rinfo = ocmr_recon_subset(os.path.join(DL, "us_0084_1_5T.h5"), "slice", 6)  # mid slice -> frames
    R = round(rinfo["frames"] * stack.shape[1] / max(rinfo["lines"], 1), 1)
    idx = np.linspace(0, rt.shape[2] - 1, 4).round().astype(int)
    strip([rt[:, :, p] for p in idx] + [rt.mean(2)],
          [f"frame {p}" for p in idx] + ["temporal mean"],
          f"OCMR us_0084 - REAL-TIME FREE-BREATHING SAX (slice 6/12, R~{R} zero-filled; needs CS recon)",
          os.path.join(OUT, "ocmr_us0084_realtime_freebreath.png"))
    manifest["datasets"].append(entry)


def probe_zenodo_rtcine(manifest):
    base = "https://zenodo.org/api/records/10639392/files"
    vol = os.path.join(DL, "zen_images", "images", "vol14")
    entry = {"name": "Zenodo 10639392 (RT free-breathing cine)",
             "landing": "https://zenodo.org/records/10639392",
             "access": "public Zenodo", "license": "see record", "format": "BART .cfl/.hdr (recon images)",
             "note": "SINGLE SAX slice per subject (2D, not a stack)", "files": []}
    if not os.path.exists(os.path.join(vol, "cine.cfl")):
        os.makedirs(os.path.join(DL, "zen_images"), exist_ok=True)
        # stream-extract only vol14 (first dir in the 7.6 GB tar), then stop the stream
        sh(f'cd {DL} && (curl -sL --max-time 2400 "{base}/images.tgz/content" | '
           f'tar -xzv -C zen_images images/vol14 2>/dev/null & '
           f'CPID=$!; while [ ! -f zen_images/images/vol14/rt_maxstress.cfl ]; do sleep 5; '
           f'done; sleep 5; pkill -f "images.tgz/content"; wait $CPID 2>/dev/null) ; true')
    for name in ["cine", "rt"]:
        cfl = os.path.join(vol, f"{name}.cfl")
        if os.path.exists(cfl):
            dims = [int(x) for x in open(os.path.join(vol, f"{name}.hdr")).read().split("\n")[1].split()]
            mag = np.sqrt((np.abs(read_cfl(os.path.join(vol, name))) ** 2).sum(axis=-1))
            np.save(os.path.join(DL, f"zen_vol14_{name}.npy"), mag.astype(np.float32))
            entry["files"].append({"file": f"vol14/{name}.cfl", "bytes": os.path.getsize(cfl),
                                   "bart_dims": dims, "verified": True})
    # render pairing
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cine = np.load(os.path.join(DL, "zen_vol14_cine.npy"))
    rt = np.load(os.path.join(DL, "zen_vol14_rt.npy"))
    fig, axes = plt.subplots(2, 3, figsize=(8, 5.6))
    for j, p in enumerate(np.linspace(0, cine.shape[2] - 1, 3).round().astype(int)):
        axes[0, j].imshow(cine[:, :, p].T, cmap="gray", vmin=0, vmax=np.quantile(cine[:, :, p], 0.9))
        axes[0, j].set_title(f"cine (BH) f{j}", fontsize=8); axes[0, j].axis("off")
    for j, p in enumerate(np.linspace(0, rt.shape[2] - 1, 3).round().astype(int)):
        axes[1, j].imshow(rt[:, :, p].T, cmap="gray", vmin=0, vmax=np.quantile(rt[:, :, p], 0.9))
        axes[1, j].set_title(f"real-time FB f{j}", fontsize=8); axes[1, j].axis("off")
    fig.suptitle("Zenodo 10639392 vol14 - breath-hold cine (top) vs real-time free-breathing (bottom), SINGLE SAX slice", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "zenodo_vol14_cine_vs_rt.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("saved", os.path.join(OUT, "zenodo_vol14_cine_vs_rt.png"))
    manifest["datasets"].append(entry)


def probe_freerunning(manifest):
    """Accessibility check only (4-6 GB/vol, GRE whole-heart -> contrast/orientation mismatch)."""
    url = "https://zenodo.org/api/records/7621356/files/V1.7z/content"
    sz = head_size(url)
    manifest["datasets"].append({
        "name": "Zenodo 7621356 (free-running whole-heart)",
        "landing": "https://zenodo.org/records/7621356", "access": "public Zenodo",
        "format": "raw multi-echo GRE k-space (.7z per volume)",
        "note": "GRE whole-heart (NOT bSSFP SAX); 4-6 GB/vol. Accessibility verified, not downloaded.",
        "files": [{"file": "V1.7z", "url": url, "bytes": sz, "verified": sz > 0, "downloaded": False}],
    })


def probe_gottingen(manifest):
    """Goettingen radial real-time free-breathing bSSFP SAX (Blumenthal/Uecker, NLINV-Net).
    Verify accessibility + BART dims via the tiny .hdr; the .cfl is 3.45 GB/vol radial k-space
    (full recon needs NUFFT), so we record dims rather than download a volume."""
    base = "https://zenodo.org/api/records/10492333/files"
    hdr = os.path.join(DL, "got_vol0001.hdr")
    curl_download(f"{base}/vol0001_vis1.hdr/content", hdr, max_time=60)
    dims = None
    if os.path.exists(hdr):
        for ln in open(hdr).read().split("\n"):
            if ln and not ln.startswith("#"):
                dims = [int(x) for x in ln.split()]
                break
    cfl_bytes = head_size(f"{base}/vol0001_vis1.cfl/content")
    manifest["datasets"].append({
        "name": "Zenodo 10492333 (Goettingen radial real-time bSSFP, 1/5)",
        "landing": "https://zenodo.org/records/10492333", "access": "public Zenodo",
        "license": "CC-BY-4.0", "format": "BART .cfl/.hdr RADIAL raw k-space",
        "note": "40 volunteers (5 parts), 22-27 SAX slices each, free-breathing/ungated, ~33 ms/frame. "
                "Radial -> recon needs NUFFT; no shipped clean reference.",
        "files": [{"file": "vol0001_vis1.cfl", "url": f"{base}/vol0001_vis1.cfl/content",
                   "bytes": cfl_bytes, "bart_dims": dims, "verified": cfl_bytes > 0, "downloaded": False,
                   "dims_meaning": "READ x SPOKES x COIL(34) x ... x FRAMES(127) x SLICES(24)"}],
    })


def probe_cardiopulm(manifest):
    """Free-running 5D cardiopulmonary demo (Zenodo 15033956) - download the small reconstructed
    image .mat (not the 6.5 GB 5D), render a 3D motion-resolved-reference example."""
    import h5py
    url = "https://zenodo.org/api/records/15033956/files/imgNufft1.mat/content"
    path = os.path.join(DL, "cardiopulm_imgNufft1.mat")
    nbytes = curl_download(url, path, max_time=400)
    shape = None
    if nbytes > 0:
        v = h5py.File(path, "r")["imgNufft1"]
        v = np.abs(v["real"][:] + 1j * v["imag"][:]).squeeze()
        shape = list(v.shape)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sl = [v[v.shape[0] // 2], v[:, v.shape[1] // 2], v[:, :, v.shape[2] // 2]]
        fig, ax = plt.subplots(1, 3, figsize=(9, 3.2))
        for a, s, t in zip(ax, sl, ["mid ax0", "mid ax1", "mid ax2"]):
            a.imshow(s.T, cmap="gray", vmin=0, vmax=np.quantile(s, 0.985), origin="lower")
            a.set_title(t, fontsize=9); a.axis("off")
        fig.suptitle("Zenodo 15033956 - free-running 5D cardiopulmonary recon (NUFFT), isotropic whole-heart (N=1 demo)", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "cardiopulm_5d_recon.png"), dpi=115, bbox_inches="tight")
        plt.close(fig)
        print("saved", os.path.join(OUT, "cardiopulm_5d_recon.png"))
    manifest["datasets"].append({
        "name": "Zenodo 15033956 (free-running 5D cardiopulmonary demo)",
        "landing": "https://zenodo.org/records/15033956", "access": "public Zenodo",
        "license": "CC-BY-4.0", "format": "MATLAB .mat (raw k-space + 5D recon outputs)",
        "note": "N=1 demo; ships a motion-resolved 5D recon (cardiac x respiratory). Isotropic whole-heart, NOT SAX cine.",
        "files": [{"file": "imgNufft1.mat", "url": url, "bytes": nbytes, "img_shape": shape, "verified": nbytes > 0}],
    })


def probe_fiss(manifest):
    """FISS 5D whole-heart free-running (Zenodo 13868462) - accessibility only (one 34 GB zip)."""
    url = "https://zenodo.org/api/records/13868462/files/inVivo.zip/content"
    sz = head_size(url)
    manifest["datasets"].append({
        "name": "Zenodo 13868462 (FISS 5D whole-heart free-running)",
        "landing": "https://zenodo.org/records/13868462", "access": "public Zenodo", "license": "CC-BY-4.0",
        "format": "raw k-space (one inVivo.zip)",
        "note": "5 volunteers, 25 cardiac x 4 respiratory phases (5D ref). GRE-family whole-heart, NOT SAX. 34 GB single zip.",
        "files": [{"file": "inVivo.zip", "url": url, "bytes": sz, "verified": sz > 0, "downloaded": False}],
    })


def main():
    manifest = {"datasets": []}
    probe_ocmr(manifest)
    try:
        probe_zenodo_rtcine(manifest)
    except Exception as e:
        print("zenodo probe failed:", e)
    probe_gottingen(manifest)
    probe_cardiopulm(manifest)
    probe_freerunning(manifest)
    probe_fiss(manifest)
    with open(os.path.join(OUT, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print("\nwrote", os.path.join(OUT, "manifest.json"))


if __name__ == "__main__":
    main()
