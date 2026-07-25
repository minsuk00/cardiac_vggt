#!/usr/bin/env python
"""Download CMRxRecon2025 V2 (post-challenge, full k-space) from Google Drive via rclone.

WHY rclone (not gdown): these files' ANONYMOUS download quota is exhausted, so gdown fails
("cannot access ... after 24 hours"). rclone with the user's OAuth remote `gdrive` (drive.readonly)
is authenticated and works.

Method: `rclone copy` of each series' Drive SUBFOLDER (parallel, resumable, live progress).
- TrainingData (58 parts), TaskR1 (51 parts): clean.
- TaskR2 (59 unique parts): the Drive folder has DUPLICATE names for parts 020-048; `rclone copy`
  auto-dedupes them ("Duplicate object found in source - ignoring", keeps one). There are NO md5s,
  so VALIDATE R2 by a successful `unzip` after merging.
Files land WITH the Drive "Copy of " prefix -> strip before the `cat` merge.

Run in a tmux on a login node (outbound internet + no walltime kill; download is network/IO-bound):
  python tools/download_cmrxrecon2025.py --series TaskR1 TaskR2
  python tools/download_cmrxrecon2025.py --series TrainingData      # (already done)
"""
import argparse, os, subprocess

DEST_DEFAULT = "/home/minsukc/vggt/scratch/data/CMRxRecon2025"
RCLONE = os.path.expanduser("~/.local/bin/rclone")
REMOTE = "gdrive"
ROOT_FOLDER_ID = "1qgT-97N_As0WOP2PiVJsSFDzTIxIBdXU"   # "ValidationTestDataset_TrainingData-updated"
SUBFOLDER = {
    "TrainingData": "TrainingData-updated",
    "TaskR1": "ValidationTestDataTaskR1",
    "TaskR2": "ValidationTestDataTaskR2",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", nargs="+", default=["TaskR1", "TaskR2"], choices=list(SUBFOLDER))
    ap.add_argument("--dest", default=DEST_DEFAULT)
    ap.add_argument("--transfers", type=int, default=4)
    a = ap.parse_args()

    for s in a.series:
        dst = os.path.join(a.dest, s)
        os.makedirs(dst, exist_ok=True)
        src = f"{REMOTE},root_folder_id={ROOT_FOLDER_ID}:{SUBFOLDER[s]}"
        log = os.path.join(a.dest, f"rclone_{s.lower()}.log")
        print(f"\n=== {s}: {src} -> {dst} (log {log}) ===", flush=True)
        cmd = [RCLONE, "copy", src, dst,
               "--transfers", str(a.transfers), "--drive-chunk-size", "64M",
               "--drive-acknowledge-abuse", "--progress", "--stats", "30s",
               "--log-file", log, "--log-level", "INFO"]
        rc = subprocess.run(cmd).returncode
        print(f"=== {s} rclone exit {rc} ===", flush=True)


if __name__ == "__main__":
    main()
