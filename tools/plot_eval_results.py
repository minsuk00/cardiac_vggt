#!/usr/bin/env python
"""Generate summary plot for CMRxRecon 2023 & 2025 evaluation results.

Saves result/cmrxrecon_2023_2025_summary.png
"""

import json
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

VGGT_ROOT = "/home/minsukc/vggt"
SUMMARY_JSON = os.path.join(VGGT_ROOT, "result/cmrxrecon_2023_2025/summary.json")
OUT_PNG = os.path.join(VGGT_ROOT, "result/cmrxrecon_2023_2025_summary.png")


def main():
    if not os.path.exists(SUMMARY_JSON):
        raise FileNotFoundError(f"Missing summary JSON: {SUMMARY_JSON}")
        
    with open(SUMMARY_JSON) as f:
        data = json.load(f)
        
    labels = []
    vendors = []
    psnrs = []
    ssims = []
    
    # 2023 subjects
    for subj, res in data.get("cmrxrecon2023", {}).items():
        labels.append(f"2023: {subj}")
        vendors.append("Siemens")
        psnrs.append(res["breath"]["psnr"])
        ssims.append(res["breath"]["ssim"])
        
    # 2025 subjects
    for subj, res in data.get("cmrxrecon2025", {}).items():
        if "Siemens" in subj:
            if "Aera" in subj:
                lbl = "2025: Siemens 1.5T Aera"
            elif "Prisma" in subj:
                lbl = "2025: Siemens 3.0T Prisma"
            else:
                lbl = "2025: Siemens 3.0T Vida"
            v = "Siemens"
        elif "Philips" in subj:
            lbl = "2025: Philips 3.0T IngeniaCX"
            v = "Philips"
        elif "UIH" in subj:
            if "670" in subj:
                lbl = "2025: UIH 1.5T uMR670"
            else:
                lbl = "2025: UIH 3.0T uMR780"
            v = "UIH"
        else:
            lbl = f"2025: {subj}"
            v = "Other"
            
        labels.append(lbl)
        vendors.append(v)
        psnrs.append(res["breath"]["psnr"])
        ssims.append(res["breath"]["ssim"])
        
    color_map = {
        "Siemens": "#1f77b4",  # Blue
        "Philips": "#2ca02c",  # Green
        "UIH": "#9467bd"      # Purple
    }
    colors = [color_map.get(v, "#7f7f7f") for v in vendors]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    x = np.arange(len(labels))
    width = 0.55
    
    # Panel 1: PSNR
    bars1 = ax1.bar(x, psnrs, width=width, color=colors, edgecolor="black", linewidth=0.8)
    ax1.set_ylabel("PSNR (dB)", fontsize=12, fontweight="bold")
    ax1.set_title("Reconstruction Fidelity under Breathing Motion (PSNR)", fontsize=13, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax1.set_ylim(0, 38)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.axhline(30.0, color="red", linestyle=":", linewidth=1.5, label="30 dB Benchmark")
    
    for bar, val in zip(bars1, psnrs):
        ax1.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.5,
                 f"{val:.1f} dB", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
                 
    # Panel 2: SSIM
    bars2 = ax2.bar(x, ssims, width=width, color=colors, edgecolor="black", linewidth=0.8)
    ax2.set_ylabel("SSIM", fontsize=12, fontweight="bold")
    ax2.set_title("Structural Similarity under Breathing Motion (SSIM)", fontsize=13, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax2.set_ylim(0, 1.05)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.axhline(0.80, color="darkgreen", linestyle=":", linewidth=1.5, label="0.80 Structural Floor")
    
    for bar, val in zip(bars2, ssims):
        ax2.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.015,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
                 
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#1f77b4", edgecolor="black", label="Siemens (In-Domain Vendor)"),
        Patch(facecolor="#2ca02c", edgecolor="black", label="Philips (OOD Vendor)"),
        Patch(facecolor="#9467bd", edgecolor="black", label="UIH (OOD Vendor)")
    ]
    fig.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=3, fontsize=10.5)
    
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    print(f"Summary plot successfully saved to: {OUT_PNG}")


if __name__ == "__main__":
    main()
