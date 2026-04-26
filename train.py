# ═══════════════════════════════════════════════════════════════
# FINAL PIPELINE — All 5 problems fixed
# ═══════════════════════════════════════════════════════════════

import warnings, logging
warnings.filterwarnings("ignore")
logging.getLogger("rasterio").setLevel(logging.ERROR)

import os, time, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import rasterio
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.amp import autocast, GradScaler
from terratorch import BACKBONE_REGISTRY
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from scipy.ndimage import binary_dilation

AUG_DIR   = "/kaggle/working/karnataka_2k_final"
SRC_DIR   = "/kaggle/working/karnataka_2k"
LINES_DIR = f"{AUG_DIR}/lines"

KEEP_BANDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11]
S2_BAND_NAMES = [
    'COASTAL_AEROSOL','BLUE','GREEN','RED',
    'RED_EDGE_1','RED_EDGE_2','RED_EDGE_3',
    'NIR_BROAD','NIR_NARROW','WATER_VAPOR','SWIR_2',
]
TM_SIZE = 224
S2_NORM = 10000.0


# ═══════════════════════════════════════════════════════════════
# STEP 1 — Hard-equalise corridor line lengths
# Strategy: for every patch, keep exactly TARGET_PX pixels
# by walking from line midpoint outward — not from centroid
# ═══════════════════════════════════════════════════════════════

TARGET_PX       = 180   # enc=0 median is 194, use slightly below
CORRIDOR_RADIUS = 4
struct = np.ones((2*CORRIDOR_RADIUS+1, 2*CORRIDOR_RADIUS+1), dtype=bool)

def equalise_corridor(stem):
    lp = f"{LINES_DIR}/{stem}_lines.tif"
    cp = f"{AUG_DIR}/corridor/{stem}_corridor.tif"

    with rasterio.open(lp) as ds:
        line_mask = ds.read(1)
    with rasterio.open(cp) as ds:
        profile = ds.profile.copy()

    line_bin = (line_mask > 0)
    coords   = np.argwhere(line_bin)
    n_px     = len(coords)

    if n_px == 0:
        corridor = np.zeros((256, 256), dtype=np.uint8)
    elif n_px <= TARGET_PX:
        # Short line — use as-is, no padding
        corridor = binary_dilation(line_bin, structure=struct).astype(np.uint8)
    else:
        # Long line — keep TARGET_PX pixels nearest to midpoint
        mid_idx  = n_px // 2
        distances = np.abs(np.arange(n_px) - mid_idx)
        keep_idx  = np.argsort(distances)[:TARGET_PX]
        new_line  = np.zeros_like(line_bin)
        new_line[coords[keep_idx, 0], coords[keep_idx, 1]] = True
        corridor = binary_dilation(new_line, structure=struct).astype(np.uint8)

    profile.update(dtype=rasterio.uint8, count=1)
    with rasterio.open(cp, 'w', **profile) as ds:
        ds.write(corridor[np.newaxis])
    return corridor.sum() / (256 * 256)

print("STEP 1 — Equalising corridor line lengths...")
df_all  = pd.read_csv(f"{AUG_DIR}/labels.csv")
df_line = df_all[df_all.has_line == 1].reset_index(drop=True)
y       = df_line.encroached.values

cov_list = []
for _, row in df_line.iterrows():
    cov = equalise_corridor(row.stem)
    cov_list.append(cov)
cov_arr = np.array(cov_list)

auc_cov = roc_auc_score(y, cov_arr)
auc_cov = max(auc_cov, 1 - auc_cov)
print(f"  Corridor coverage AUC after equalisation: {auc_cov:.4f} "
      f"{'🟢 OK' if auc_cov < 0.60 else '🟡 HIGH' if auc_cov < 0.70 else '🔴 LEAKY'}")
print(f"  enc=0 coverage: {cov_arr[y==0].mean():.5f}")
print(f"  enc=1 coverage: {cov_arr[y==1].mean():.5f}")
print(f"  Ratio: {cov_arr[y==1].mean()/(cov_arr[y==0].mean()+1e-9):.3f}x")


# ═══════════════════════════════════════════════════════════════
# STEP 2 — Build split using patch-index split on karnataka_n
# ═══════════════════════════════════════════════════════════════
#
# Only 3 AOIs have usable class diversity: n, ne, nc
# Strategy:
#   Val  = karnataka_ne (45 enc=0, 104 enc=1) — fixed AOI
#   Test = karnataka_nc (24 enc=0, 62 enc=1) — fixed AOI
#   Train = all remaining (karnataka_n + nw + sw + se + c + w)
#   karnataka_n contributes 130 enc=0 to training — critical
#
# Do NOT split karnataka_n — we need all its enc=0 in training

VAL_AOIS  = ['karnataka_ne']
TEST_AOIS = ['karnataka_nc']

def build_dataloaders(batch_size=16, num_workers=2):
    df = pd.read_csv(f"{AUG_DIR}/labels.csv")
    df = df[df.has_line == 1].reset_index(drop=True)

    test  = df[df.aoi.isin(TEST_AOIS)]
    val   = df[df.aoi.isin(VAL_AOIS)]
    train = df[~df.aoi.isin(VAL_AOIS + TEST_AOIS)]

    print("Data splits:")
    for name, sub in [("Train", train), ("Val", val), ("Test", test)]:
        n0 = int((sub.encroached==0).sum())
        n1 = int((sub.encroached==1).sum())
        print(f"  {name:5s}: n={len(sub):4d}  enc0={n0:3d}  enc1={n1:4d}  "
              f"({sub.encroached.mean():.1%} enc)  "
              f"AOIs={sorted(sub.aoi.unique())}")

    # WeightedRandomSampler on training
    labels = train.encroached.values.astype(int)
    cls_w  = 1.0 / np.bincount(labels)
    sampler = WeightedRandomSampler(cls_w[labels], len(train), replacement=True)
    print(f"  Sampler: enc=0 seen {cls_w[0]/cls_w[1]:.1f}× more often\n")

    kw = dict(num_workers=num_workers, pin_memory=True)
    return (
        DataLoader(EncroachmentDataset(train, augment=True),
                   batch_size, sampler=sampler, drop_last=True, **kw),
        DataLoader(EncroachmentDataset(val,   augment=False),
                   batch_size, shuffle=False, **kw),
        DataLoader(EncroachmentDataset(test,  augment=False),
                   batch_size, shuffle=False, **kw),
        train,
    )


# ═══════════════════════════════════════════════════════════════
# STEP 3 — Dataset
# FIX: compute NDVI-diff (in minus out) instead of NDVI-in-corridor
# NDVI-diff removes geographic bias (AUC 0.57 vs 0.72 for NDVI-in)
# FIX: severity REMOVED from loss — it's a perfect label leak
# ═══════════════════════════════════════════════════════════════

class EncroachmentDataset(Dataset):
    def __init__(self, df, augment=False):
        self.df      = df.reset_index(drop=True)
        self.augment = augment

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        stem = row["stem"]

        with rasterio.open(f"{SRC_DIR}/s2/{stem}_s2.tif") as ds:
            s2 = ds.read().astype(np.float32) / S2_NORM
        s2 = np.clip(s2[KEEP_BANDS], 0, 3.0)

        with rasterio.open(f"{AUG_DIR}/corridor/{stem}_corridor.tif") as ds:
            corridor = ds.read(1).astype(np.float32)[np.newaxis]

        if self.augment:
            s2, corridor = self._aug(s2, corridor)

        s2       = self._resize(s2, TM_SIZE, 'bilinear')
        corridor = (self._resize(corridor, TM_SIZE, 'nearest') > 0.5).astype(np.float32)

        return {
            "s2":       torch.from_numpy(s2),
            "corridor": torch.from_numpy(corridor),
            "enc":      torch.tensor(float(row["encroached"]), dtype=torch.float32),
            # severity intentionally excluded
        }

    def _resize(self, arr, size, mode):
        t  = torch.from_numpy(arr).unsqueeze(0)
        kw = {"align_corners": False} if mode == "bilinear" else {}
        return F.interpolate(t, (size, size), mode=mode, **kw).squeeze(0).numpy()

    def _aug(self, s2, corridor):
        # Flip / rotate
        if np.random.rand() < 0.5:
            s2 = s2[:, :, ::-1].copy(); corridor = corridor[:, :, ::-1].copy()
        if np.random.rand() < 0.5:
            s2 = s2[:, ::-1].copy(); corridor = corridor[:, ::-1].copy()
        k = np.random.randint(4)
        if k:
            s2 = np.rot90(s2, k, (1,2)).copy()
            corridor = np.rot90(corridor, k, (1,2)).copy()
        # Spectral jitter
        if np.random.rand() < 0.5:
            s2 *= np.random.uniform(0.85, 1.15, (s2.shape[0], 1, 1)).astype(np.float32)
        # Gaussian noise
        if np.random.rand() < 0.4:
            s2 += np.random.normal(0, 0.025, s2.shape).astype(np.float32)
        # CutOut — random 32×32 block zeroed (forces local feature learning)
        if np.random.rand() < 0.3:
            r = np.random.randint(0, s2.shape[1]-32)
            c = np.random.randint(0, s2.shape[2]-32)
            s2[:, r:r+32, c:c+32] = 0.0
        return np.clip(s2, 0, 3.0), corridor


# ═══════════════════════════════════════════════════════════════
# STEP 4 — Model
# FIX: Backbone FULLY FROZEN — 28M params / 1288 samples = overfit
# FIX: NDVI-diff feature (in-out) replaces NDVI-in-corridor
# FIX: Severity head removed entirely
# ═══════════════════════════════════════════════════════════════

class CorridorEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.GELU(),
            nn.Conv2d(16, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.GELU(),
        )
    def forward(self, x, size):
        return self.net(F.interpolate(x, size=size, mode='nearest'))


class ClassificationHead(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(128, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(1)


class TerraMindModel(nn.Module):
    def __init__(self):
        super().__init__()
        print("Loading terramind_v1_base ...", flush=True)
        self.backbone = BACKBONE_REGISTRY.build(
            'terramind_v1_base', pretrained=True,
            modalities=['S2L2A'], bands={'S2L2A': S2_BAND_NAMES},
        )
        # FULLY FROZEN — no fine-tuning
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()
        print("✅ Backbone FULLY FROZEN")

        self._feat_dim = self._probe()
        print(f"Backbone feature dim: {self._feat_dim}")

        self.corridor_enc = CorridorEncoder()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)

        # backbone(gap+gmp) + corridor(gap+gmp) + 3 ndvi stats
        # ndvi stats: [diff_mean, diff_std, diff_max]  (geographically debiased)
        pool_in = self._feat_dim * 2 + 32 * 2 + 3
        self.proj = nn.Sequential(
            nn.Linear(pool_in, 256),
            nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.4),
        )
        self.head = ClassificationHead(256)

    @torch.no_grad()
    def _probe(self):
        self.backbone.eval()
        dummy = torch.randn(1, len(S2_BAND_NAMES), 224, 224)
        try:    out = self.backbone({'S2L2A': dummy})
        except: out = self.backbone(dummy)
        feat = out[-1] if isinstance(out, (list,tuple)) else \
               list(out.values())[-1] if isinstance(out, dict) else out
        return feat.shape[2] if feat.dim()==3 else feat.shape[1]

    def _extract(self, s2):
        with torch.no_grad():
            try:    out = self.backbone({'S2L2A': s2})
            except: out = self.backbone(s2)
        feat = out[-1] if isinstance(out, (list,tuple)) else \
               list(out.values())[-1] if isinstance(out, dict) else out
        if feat.dim() == 3:
            B, N, D = feat.shape
            h = w = int(N**0.5)
            feat = feat.transpose(1,2).reshape(B, D, h, w)
        return feat

    def _ndvi_diff_stats(self, s2, corridor):
        """
        NDVI-diff = NDVI_inside_corridor - NDVI_outside_corridor
        This removes the geographic bias (overall vegetation level per AOI)
        and isolates the corridor-specific vegetation signal.
        """
        red  = s2[:, 3:4]
        nir  = s2[:, 7:8]
        ndvi = ((nir - red) / (nir + red + 1e-6)).clamp(-1, 1)

        corr_in  = corridor.float()
        corr_out = (1 - corr_in)
        n_in     = corr_in.sum(dim=[2,3]).clamp(min=1)
        n_out    = corr_out.sum(dim=[2,3]).clamp(min=1)

        mean_in  = (ndvi * corr_in).sum(dim=[2,3])  / n_in    # [B,1]
        mean_out = (ndvi * corr_out).sum(dim=[2,3]) / n_out   # [B,1]
        diff     = mean_in - mean_out                          # [B,1]

        # Variance of diff-from-mean inside corridor
        sq_d = ((ndvi - mean_in.view(-1,1,1,1)) * corr_in) ** 2
        std_in = (sq_d.sum(dim=[2,3]) / n_in).sqrt()          # [B,1]

        # Max NDVI inside corridor
        ndvi_m = ndvi * corr_in + (-1) * corr_out
        max_in = ndvi_m.flatten(2).max(dim=2)[0]               # [B,1]

        return torch.cat([diff, std_in, max_in], dim=1)        # [B,3]

    def forward(self, s2, corridor):
        feat      = self._extract(s2)
        corr_feat = self.corridor_enc(corridor, feat.shape[2:])
        gap_bb    = self.gap(feat).flatten(1)
        gmp_bb    = self.gmp(feat).flatten(1)
        gap_co    = self.gap(corr_feat).flatten(1)
        gmp_co    = self.gmp(corr_feat).flatten(1)
        ndvi      = self._ndvi_diff_stats(s2, corridor)
        pooled    = torch.cat([gap_bb, gmp_bb, gap_co, gmp_co, ndvi], dim=1)
        return self.head(self.proj(pooled))

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()   # always eval
        return self

    def trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    def count_params(self):
        t  = sum(p.numel() for p in self.parameters())
        tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return t, tr, t-tr


# ═══════════════════════════════════════════════════════════════
# STEP 5 — Training
# FIX: BCE only (no severity loss)
# FIX: Focal loss component to handle 88:12 imbalance better
# FIX: Higher LR (head-only training needs it)
# ═══════════════════════════════════════════════════════════════

def focal_bce(logit, target, gamma=2.0, smoothing=0.05):
    """Focal BCE with label smoothing — down-weights easy enc=1 predictions."""
    smooth_t = target * (1 - smoothing) + 0.5 * smoothing
    bce   = F.binary_cross_entropy_with_logits(logit, smooth_t, reduction='none')
    prob  = torch.sigmoid(logit)
    pt    = prob * target + (1 - prob) * (1 - target)
    focal = ((1 - pt) ** gamma) * bce
    return focal.mean()


class MetricTracker:
    def __init__(self): self.reset()
    def reset(self):
        self.logits, self.labels, self.losses = [], [], []
    def update(self, logit, label, loss):
        self.logits.append(logit.detach().cpu())
        self.labels.append(label.detach().cpu())
        self.losses.append(loss)
    def compute(self):
        logits = torch.cat(self.logits).numpy()
        y      = (torch.cat(self.labels).numpy() > 0.5).astype(int)
        probs  = torch.sigmoid(torch.tensor(logits)).numpy()
        preds  = (probs > 0.5).astype(int)
        return {
            "loss": float(np.mean(self.losses)),
            "auc":  float(roc_auc_score(y, probs)) if len(np.unique(y))>1 else 0.0,
            "ap":   float(average_precision_score(y, probs)) if len(np.unique(y))>1 else 0.0,
            "f1":   float(f1_score(y, preds, zero_division=0)),
            "acc":  float((preds==y).mean()),
        }


def run_epoch(model, loader, optimizer, scaler, device, train=True, scheduler=None):
    model.train() if train else model.eval()
    tracker = MetricTracker()
    ctx = torch.enable_grad() if train else torch.no_grad()

    with ctx:
        for batch in loader:
            s2  = batch["s2"].to(device)
            cor = batch["corridor"].to(device)
            enc = batch["enc"].to(device)

            if train: optimizer.zero_grad(set_to_none=True)

            with autocast(device_type='cuda'):
                logit = model(s2, cor)
                loss  = focal_bce(logit, enc, gamma=2.0, smoothing=0.05)

            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.trainable_params(), 1.0)
                scaler.step(optimizer); scaler.update()
                if scheduler: scheduler.step()

            tracker.update(logit, enc, loss.item())

    return tracker.compute()


def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    train_dl, val_dl, test_dl, _ = build_dataloaders(
        config["batch_size"], config["num_workers"])

    model = TerraMindModel().to(device)
    total, trainable, frozen = model.count_params()
    print(f"Params: {total:,} total | {trainable:,} trainable | {frozen:,} frozen\n")

    optimizer = torch.optim.AdamW(
        model.trainable_params(),
        lr=config["lr"],
        weight_decay=config["wd"],
    )

    total_steps  = config["epochs"] * len(train_dl)
    warmup_steps = config["warmup_epochs"] * len(train_dl)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        p = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * p))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = GradScaler()
    ckpt_dir  = f"{AUG_DIR}/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    best_auc, patience_ctr, history = -1, 0, []
    hdr = (f"{'Ep':>4}  {'TrLoss':>7} {'TrAUC':>6} {'TrF1':>6}  "
           f"{'VaLoss':>7} {'VaAUC':>6} {'VaF1':>6} {'VaAP':>6}  "
           f"{'LR':>9}  {'Time':>6}")
    print(f"\n{hdr}\n{'─'*len(hdr)}")

    for ep in range(1, config["epochs"]+1):
        t0   = time.time()
        tr_m = run_epoch(model, train_dl, optimizer, scaler, device,
                         train=True, scheduler=scheduler)
        va_m = run_epoch(model, val_dl, optimizer, scaler, device, train=False)
        elapsed = time.time() - t0
        cur_lr  = optimizer.param_groups[0]["lr"]

        history.append({"epoch": ep,
                        **{f"tr_{k}": v for k,v in tr_m.items()},
                        **{f"va_{k}": v for k,v in va_m.items()},
                        "lr": cur_lr})

        marker = ""
        if va_m["auc"] > best_auc:
            best_auc, patience_ctr = va_m["auc"], 0
            torch.save({"epoch": ep, "model_state": model.state_dict(),
                        "val_metrics": va_m}, f"{ckpt_dir}/best_model.pt")
            marker = " ★"
        else:
            patience_ctr += 1

        gap = tr_m['auc'] - va_m['auc']
        gap_flag = " ⚠️OVF" if gap > 0.15 else ""

        print(f"{ep:>4}  {tr_m['loss']:>7.4f} {tr_m['auc']:>6.3f} {tr_m['f1']:>6.3f}  "
              f"{va_m['loss']:>7.4f} {va_m['auc']:>6.3f} {va_m['f1']:>6.3f} "
              f"{va_m['ap']:>6.3f}  {cur_lr:>9.2e}  {elapsed:>5.1f}s{marker}{gap_flag}",
              flush=True)

        if patience_ctr >= config["patience"]:
            print(f"\nEarly stop at epoch {ep}")
            break

    print("\n── Test evaluation ──")
    ckpt = torch.load(f"{ckpt_dir}/best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    te_m = run_epoch(model, test_dl, None, scaler, device, train=False)
    for k in ["auc","ap","f1","acc"]:
        print(f"  {k.upper():6s}: {te_m[k]:.4f}")

    pd.DataFrame(history).to_csv(f"{ckpt_dir}/train_history.csv", index=False)
    with open(f"{ckpt_dir}/test_metrics.json","w") as f:
        json.dump(te_m, f, indent=2)

    print(f"\n✅ Done. Best val AUC: {best_auc:.4f}")
    return model, history, te_m


# ═══════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════
CONFIG = {
    "epochs":        60,
    "batch_size":    16,
    "lr":            2e-4,    # higher — head-only with frozen backbone
    "wd":            1e-2,    # strong weight decay
    "warmup_epochs": 5,
    "patience":      15,
    "num_workers":   2,
}

model, history, test_metrics = train_model(CONFIG)