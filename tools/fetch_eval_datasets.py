#!/usr/bin/env python
"""Download the selected external evaluation datasets to GPFS (resumable, idempotent).

Sets (decided with the user):
  - OCMR        : SAX *stacks* only (multi-slice; both real-time us_* and gated fs_*) -> the
                  volume-relevant subset. Direct public S3.
  - Goettingen  : Part 1/5 of the radial real-time free-breathing bSSFP set (record 10492333),
                  all per-volume .cfl/.hdr. Public Zenodo.
  - Cardiopulm  : 5D cardiopulmonary demo (record 15033956) recon/reference files only
                  (5D recon + smaller recons + params); skips the ~24 GB raw echo k-space.

Each file is fetched with `curl -C -` (resume) and skipped if the local size already matches the
remote Content-Length. Writes a download_manifest.json under DEST_ROOT.

Run (background):  micromamba run -n svr python tools/fetch_eval_datasets.py
"""
import csv
import json
import os
import subprocess

DEST_ROOT = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/vggt/eval_datasets"
OCMR_S3 = "https://ocmr.s3.us-east-2.amazonaws.com"


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)


def remote_size(url):
    r = sh(f'curl -sIL --max-time 40 "{url}"')
    for ln in r.stdout.splitlines():
        if ln.lower().startswith("content-length:"):
            try:
                return int(ln.split(":", 1)[1].strip())
            except ValueError:
                pass
    return -1


def fetch(url, dest, log):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    rsz = remote_size(url)
    if os.path.exists(dest) and rsz > 0 and os.path.getsize(dest) == rsz:
        log(f"SKIP (complete) {os.path.basename(dest)} [{rsz/1e9:.2f} GB]")
        return rsz, True
    log(f"GET  {os.path.basename(dest)} [{rsz/1e9:.2f} GB] ...")
    r = sh(f'curl -fSL -C - --retry 5 --retry-delay 10 --max-time 36000 -o "{dest}" "{url}"')
    ok = os.path.exists(dest) and (rsz <= 0 or os.path.getsize(dest) == rsz)
    got = os.path.getsize(dest) if os.path.exists(dest) else 0
    log(("DONE " if ok else "FAIL ") + f"{os.path.basename(dest)} [{got/1e9:.2f} GB]"
        + ("" if ok else f"  (stderr: {r.stderr[-200:]})"))
    return got, ok


def zenodo_files(record_id):
    # curl (env urllib hits SSL cert-verify issues on this node)
    r = sh(f'curl -sL --max-time 60 "https://zenodo.org/api/records/{record_id}"')
    d = json.loads(r.stdout)
    return {f["key"]: f["links"]["self"] for f in d["files"]}


def main():
    os.makedirs(DEST_ROOT, exist_ok=True)
    logf = open(os.path.join(DEST_ROOT, "fetch.log"), "a", buffering=1)

    def log(m):
        print(m, flush=True)
        logf.write(m + "\n")

    manifest = {"datasets": []}

    # ── OCMR: SAX stacks (sli=stk, viw=sax), both fs and pse ──
    log("\n==== OCMR (SAX stacks) ====")
    ocmr_dir = os.path.join(DEST_ROOT, "ocmr")
    os.makedirs(ocmr_dir, exist_ok=True)
    csv_path = os.path.join(ocmr_dir, "ocmr_data_attributes.csv")
    fetch(f"{OCMR_S3}/ocmr_data_attributes.csv", csv_path, log)
    rows = list(csv.DictReader(open(csv_path)))
    g = lambda r, k: (r.get(k) or "").strip()
    sel = [g(r, "file name") for r in rows if g(r, "viw") == "sax" and g(r, "sli") == "stk"]
    log(f"OCMR SAX-stack files selected: {len(sel)}")
    ofiles, otot = [], 0
    for fn in sel:
        got, ok = fetch(f"{OCMR_S3}/data/{fn}", os.path.join(ocmr_dir, fn), log)
        otot += got
        ofiles.append({"file": fn, "bytes": got, "ok": ok})
    manifest["datasets"].append({"name": "OCMR SAX stacks", "dir": ocmr_dir,
                                 "n_files": len(ofiles), "bytes": otot, "files": ofiles})
    log(f"OCMR total: {otot/1e9:.1f} GB across {len(ofiles)} files")

    # ── Goettingen radial real-time bSSFP, Part 1/5 (record 10492333) ──
    log("\n==== Goettingen radial bSSFP (part 1/5) ====")
    got_dir = os.path.join(DEST_ROOT, "gottingen_radial", "part1")
    files = zenodo_files("10492333")
    gfiles, gtot = [], 0
    for key, url in sorted(files.items()):
        got, ok = fetch(url, os.path.join(got_dir, key), log)
        gtot += got
        gfiles.append({"file": key, "bytes": got, "ok": ok})
    manifest["datasets"].append({"name": "Goettingen radial bSSFP part1", "dir": got_dir,
                                 "n_files": len(gfiles), "bytes": gtot, "files": gfiles})
    log(f"Goettingen part1 total: {gtot/1e9:.1f} GB across {len(gfiles)} files")

    # ── 5D cardiopulmonary demo (record 15033956): recon/reference files only ──
    log("\n==== 5D cardiopulmonary (recon files) ====")
    cp_dir = os.path.join(DEST_ROOT, "cardiopulmonary_5d")
    files = zenodo_files("15033956")
    keep = ("seqParam.mat", "ZGradientTraj.mat", "imgNufft1.mat", "recon_nufft1.mat",
            "MostMoCo5frame1.mat", "recon_RLR_2Ref1.mat", "recon_nufft_5D1.mat")
    cfiles, ctot = [], 0
    for key in keep:
        if key in files:
            got, ok = fetch(files[key], os.path.join(cp_dir, key), log)
            ctot += got
            cfiles.append({"file": key, "bytes": got, "ok": ok})
        else:
            log(f"  (not found in record: {key})")
    manifest["datasets"].append({"name": "Cardiopulmonary 5D (recon)", "dir": cp_dir,
                                 "n_files": len(cfiles), "bytes": ctot, "files": cfiles})
    log(f"Cardiopulmonary total: {ctot/1e9:.1f} GB across {len(cfiles)} files")

    grand = otot + gtot + ctot
    log(f"\n==== GRAND TOTAL: {grand/1e9:.1f} GB ====")
    json.dump(manifest, open(os.path.join(DEST_ROOT, "download_manifest.json"), "w"), indent=2)
    log("wrote download_manifest.json")
    logf.close()


if __name__ == "__main__":
    main()
