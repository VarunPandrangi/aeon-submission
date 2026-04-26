#!/usr/bin/env python3
"""
infer.py — Single-patch inference for TerraMind Encroachment Detector

Usage:
    python infer.py --s2 path/to/patch_s2.tif --corridor path/to/patch_corridor.tif

Output:
    JSON to stdout: {"encroached": 1, "probability": 0.82, "threshold_used": 0.30}

Requirements:
    pip install -r requirements.txt
    Model weights: download best_model.pt from [LINK] and place in ./checkpoints/
"""

import argparse, json, sys, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import rasterio

KEEP_BANDS    = [0,1,2,3,4,5,6,7,8,9,11]
S2_BAND_NAMES = [
    "COASTAL_AEROSOL","BLUE","GREEN","RED",
    "RED_EDGE_1","RED_EDGE_2","RED_EDGE_3",
    "NIR_BROAD","NIR_NARROW","WATER_VAPOR","SWIR_2",
]
S2_NORM = 10000.0
TM_SIZE = 224
THRESHOLD = 0.30   # recall-optimised operating point

# ── Model definition (must match training) ──────────────────

class CorridorEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1,bias=False), nn.BatchNorm2d(16), nn.GELU(),
            nn.Conv2d(16,32,3,padding=1,bias=False), nn.BatchNorm2d(32), nn.GELU(),
        )
    def forward(self, x, size):
        return self.net(F.interpolate(x, size=size, mode="nearest"))

class ClassificationHead(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim,256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.5),
            nn.Linear(256,128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(128,1),
        )
    def forward(self, x): return self.net(x).squeeze(1)

class TerraMindModel(nn.Module):
    def __init__(self):
        super().__init__()
        from terratorch import BACKBONE_REGISTRY
        self.backbone = BACKBONE_REGISTRY.build(
            "terramind_v1_base", pretrained=True,
            modalities=["S2L2A"], bands={"S2L2A": S2_BAND_NAMES},
        )
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()
        self._feat_dim = self._probe()
        self.corridor_enc = CorridorEncoder()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        pool_in = self._feat_dim*2 + 32*2 + 3
        self.proj = nn.Sequential(
            nn.Linear(pool_in,256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.4))
        self.head = ClassificationHead(256)

    @torch.no_grad()
    def _probe(self):
        self.backbone.eval()
        dummy = torch.randn(1, len(S2_BAND_NAMES), 224, 224)
        try:    out = self.backbone({"S2L2A": dummy})
        except: out = self.backbone(dummy)
        feat = out[-1] if isinstance(out,(list,tuple)) else list(out.values())[-1] if isinstance(out,dict) else out
        return feat.shape[2] if feat.dim()==3 else feat.shape[1]

    def _extract(self, s2):
        with torch.no_grad():
            try:    out = self.backbone({"S2L2A": s2})
            except: out = self.backbone(s2)
        feat = out[-1] if isinstance(out,(list,tuple)) else list(out.values())[-1] if isinstance(out,dict) else out
        if feat.dim()==3:
            B,N,D=feat.shape; h=w=int(N**0.5)
            feat=feat.transpose(1,2).reshape(B,D,h,w)
        return feat

    def _ndvi_diff_stats(self, s2, corridor):
        red=s2[:,3:4]; nir=s2[:,7:8]
        ndvi=((nir-red)/(nir+red+1e-6)).clamp(-1,1)
        ci=corridor.float(); co=1-ci
        n_in=ci.sum(dim=[2,3]).clamp(min=1); n_out=co.sum(dim=[2,3]).clamp(min=1)
        mi=(ndvi*ci).sum(dim=[2,3])/n_in; mo=(ndvi*co).sum(dim=[2,3])/n_out
        diff=mi-mo
        sq=((ndvi-mi.view(-1,1,1,1))*ci)**2
        std=(sq.sum(dim=[2,3])/n_in).sqrt()
        mx=(ndvi*ci+(-1)*co).flatten(2).max(dim=2)[0]
        return torch.cat([diff,std,mx],dim=1)

    def forward(self, s2, corridor):
        feat=self._extract(s2); cf=self.corridor_enc(corridor,feat.shape[2:])
        pooled=torch.cat([self.gap(feat).flatten(1),self.gmp(feat).flatten(1),
                          self.gap(cf).flatten(1),self.gmp(cf).flatten(1),
                          self._ndvi_diff_stats(s2,corridor)],dim=1)
        return self.head(self.proj(pooled))

# ── Main ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s2",       required=True, help="Path to Sentinel-2 GeoTIFF (12+ bands)")
    parser.add_argument("--corridor", required=True, help="Path to corridor mask GeoTIFF (1 band, uint8)")
    parser.add_argument("--weights",  default="checkpoints/best_model.pt")
    parser.add_argument("--threshold",type=float, default=THRESHOLD)
    parser.add_argument("--device",   default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load model
    model = TerraMindModel().to(device)
    ckpt  = torch.load(args.weights, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Load inputs
    with rasterio.open(args.s2) as ds:
        s2 = ds.read().astype(np.float32) / S2_NORM
    s2 = np.clip(s2[KEEP_BANDS], 0, 3.0)

    with rasterio.open(args.corridor) as ds:
        corridor = ds.read(1).astype(np.float32)[np.newaxis]

    def resize(arr, mode):
        t = torch.from_numpy(arr).unsqueeze(0)
        kw = {"align_corners": False} if mode=="bilinear" else {}
        return F.interpolate(t,(TM_SIZE,TM_SIZE),mode=mode,**kw).squeeze(0).numpy()

    s2t  = torch.from_numpy(resize(s2, "bilinear")).unsqueeze(0).to(device)
    ct   = torch.from_numpy(
        (resize(corridor,"nearest") > 0.5).astype(np.float32)
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        logit = model(s2t, ct)
    prob = torch.sigmoid(logit).item()

    result = {
        "encroached":    int(prob >= args.threshold),
        "probability":   round(prob, 4),
        "threshold_used": args.threshold,
    }
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
