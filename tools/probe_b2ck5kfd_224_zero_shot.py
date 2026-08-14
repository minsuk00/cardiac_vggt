import json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/home/minsukc/vggt"
sys.path[:0] = [os.path.join(REPO, "training"), REPO]
from data.datasets.mri_dataset import MRIDataset
from vggt.models.vggt import VGGT
from vggt.utils.checkpoint_stage import stage_checkpoint_to_local
from loss import compute_volume_intensity_loss

CKPT = os.path.join(REPO, "scratch/logs/213520194_mri_volume_heartl1_w050_dynamic_axial_cmrx24only/ckpts/checkpoint_best.pt")
ROOT = os.path.join(REPO, "scratch/data")
SPLIT = os.path.join(REPO, "training/splits/cmrx24only.txt")
OUT_DIR = os.path.join(REPO, "result", "resolution_experiments")

def stack(data, key, device, dtype=np.float32):
    return torch.from_numpy(np.stack(data[key]).astype(dtype)).unsqueeze(0).to(device)

def batch_for(data, device):
    return {
        "images": stack(data, "images", device).permute(0,1,4,2,3).contiguous() / 255.0,
        "scanner_coords": stack(data, "scanner_coords", device),
        "z_indices": stack(data, "z_indices", device),
        "t_indices": stack(data, "t_indices", device),
        "target_t_indices": stack(data, "target_t_indices", device),
        "gt_target_volume": torch.from_numpy(data["gt_target_volume"].astype(np.float32)).unsqueeze(0).to(device),
        "anatomy_bbox": torch.from_numpy(np.asarray(data["anatomy_bbox"], dtype=np.int64)).unsqueeze(0).to(device),
        "z_scale": torch.from_numpy(np.asarray(data["z_scale"], dtype=np.float32)).unsqueeze(0).to(device),
    }

def resize_batch(batch, r):
    out = dict(batch)
    b,s,c,h,w = batch["images"].shape
    out["images"] = F.interpolate(batch["images"].reshape(b*s,c,h,w), (r,r), mode="bilinear", align_corners=True).reshape(b,s,c,r,r)
    sc = batch["scanner_coords"].permute(0,1,4,2,3).reshape(b*s,3,h,w)
    out["scanner_coords"] = F.interpolate(sc, (r,r), mode="bilinear", align_corners=True).reshape(b,s,3,r,r).permute(0,1,3,4,2).contiguous()
    return out

def metric(pred, batch):
    out = compute_volume_intensity_loss(pred, batch, tv_weight=0.0, diffusion_weight=0.0, gather_weight=0.0, heart_weight=0.0)
    return ({k: float(out[k]) for k in ("metric_psnr_3d_bbox", "metric_psnr_3d_full", "metric_psnr_3d_motion", "metric_coverage_frac") if k in out}, out["V_canon"])

def main():
    dev = "cuda"
    os.makedirs(OUT_DIR, exist_ok=True)
    cfg = OmegaConf.create({"img_size":518,"patch_size":14,"rescale":True,"rescale_aug":False,"landscape_check":False,"augs":{"scales":[1.,1.]}})
    ds = MRIDataset(cfg, ROOT, split="val", split_file=SPLIT, mode="dynamic", mri_mode="axial", num_slices=20, target_size=518, reference_slot=True, continuous_z=False, one_frame_per_slice=True)
    model = VGGT(img_size=518, patch_size=14, embed_dim=1024, enable_point=True, use_z_pose_embedding=True, use_reference_token=True, train_on_residual_dvf=True).to(dev)
    ck = torch.load(stage_checkpoint_to_local(CKPT), map_location=dev, weights_only=False)
    result = model.load_state_dict(ck["model"], strict=False)
    print("load", len(result.missing_keys), len(result.unexpected_keys), flush=True)
    model.eval()
    records=[]; panels=[]
    for i in range(10):
        data=ds.get_data(seq_index=i, img_per_seq=20)
        b518=batch_for(data,dev); b224=resize_batch(b518,224)
        row={"seq":i,"D":int(b518["gt_target_volume"].shape[1]),"t":int(np.asarray(data["t_target"]).reshape(-1)[0])}
        preds_by_r={}
        for r,b in ((518,b518),(224,b224)):
            torch.cuda.synchronize(); t0=time.time()
            with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16): pred=model(b["images"],batch=b)
            torch.cuda.synchronize(); met,vol=metric(pred,b); row[str(r)]={**met,"seconds":time.time()-t0}
            preds_by_r[r]=vol
        z0,z1,y0,y1,x0,x1=[int(v) for v in b518["anatomy_bbox"][0].tolist()]
        z=(z0+z1)//2
        panels.append((i,row["t"],b518["gt_target_volume"][0,z,y0:y1,x0:x1].cpu().numpy(),
                       preds_by_r[518][0,z,y0:y1,x0:x1].float().cpu().numpy(),
                       preds_by_r[224][0,z,y0:y1,x0:x1].float().cpu().numpy()))
        records.append(row)
        print(json.dumps(row),flush=True)
    keys=("metric_psnr_3d_bbox","metric_psnr_3d_full","metric_psnr_3d_motion","metric_coverage_frac","seconds")
    summary={}
    for r in ("518","224"):
        summary[r]={k:float(np.mean([x[r][k] for x in records])) for k in keys if all(k in x[r] for x in records)}
    summary["delta_224_minus_518"]={k:summary["224"][k]-summary["518"][k] for k in summary["518"]}
    print("SUMMARY",json.dumps(summary,indent=2),flush=True)
    with open(os.path.join(OUT_DIR, "b2ck5kfd_224_zero_shot.json"), "w") as f:
        json.dump({"checkpoint": CKPT, "records": records, "summary": summary}, f, indent=2)
    err_max=float(np.percentile(np.concatenate([np.abs(a-b).ravel() for _,_,_,a,b in panels]),99.5))
    fig,axs=plt.subplots(10,4,figsize=(12,28),squeeze=False)
    last=None
    for row,(seq,t,gt,a,b) in enumerate(panels):
        vmax=float(np.percentile(gt,99.5)) or 1.0
        for col,(im,title,cmap,vm) in enumerate(((gt,"GT","gray",vmax),(a,"518 output","gray",vmax),(b,"224 output","gray",vmax),(np.abs(b-a),"|224−518|","magma",err_max))):
            last=axs[row,col].imshow(im,cmap=cmap,vmin=0,vmax=vm); axs[row,col].set_xticks([]); axs[row,col].set_yticks([])
            if row==0: axs[row,col].set_title(title)
        axs[row,0].set_ylabel(f"seq{seq}, t={t}\nΔbbox {records[row]['224']['metric_psnr_3d_bbox']-records[row]['518']['metric_psnr_3d_bbox']:+.2f} dB")
    cbar=fig.colorbar(last,ax=axs[:,3],fraction=.025,pad=.02); cbar.set_label("absolute normalized-intensity difference")
    fig.suptitle(f"b2ck5kfd zero-shot input resolution: 518 vs 224\nshared reconstruction window per row; shared difference scale (vmax=p99.5={err_max:.4f})")
    fig.subplots_adjust(top=.965,right=.91,hspace=.12,wspace=.03)
    out_path=os.path.join(REPO,"figs","b2ck5kfd_224_zero_shot.png")
    fig.savefig(out_path,dpi=140,bbox_inches="tight",facecolor="white"); print("SAVED",out_path,flush=True)

if __name__ == "__main__": main()
