"""
TerraMind encroachment detection model.

The packaged checkpoint includes the full backbone weights, so inference can be
run offline by constructing the architecture with ``pretrained=False`` and then
loading the local state dict.
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

S2_BAND_NAMES = [
    "COASTAL_AEROSOL",
    "BLUE",
    "GREEN",
    "RED",
    "RED_EDGE_1",
    "RED_EDGE_2",
    "RED_EDGE_3",
    "NIR_BROAD",
    "NIR_NARROW",
    "WATER_VAPOR",
    "SWIR_2",
]


class CorridorEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )

    def forward(self, x, size):
        return self.net(F.interpolate(x, size=size, mode="nearest"))


class ClassificationHead(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


class TerraMindEncroachmentModel(nn.Module):
    """
    TerraMind-1.0-base backbone (frozen) + corridor encoder + NDVI-diff head.

    Input : s2 [B,11,224,224] normalized to [0,3]
            corridor [B,1,224,224] binary
    Output: encroachment logit [B]
    """

    def __init__(self, pretrained_backbone=False):
        super().__init__()
        from terratorch import BACKBONE_REGISTRY

        self.backbone = BACKBONE_REGISTRY.build(
            "terramind_v1_base",
            pretrained=pretrained_backbone,
            modalities=["S2L2A"],
            bands={"S2L2A": S2_BAND_NAMES},
        )
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()
        self._feat_dim = self._probe()
        self.corridor_enc = CorridorEncoder()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        pool_in = self._feat_dim * 2 + 32 * 2 + 3
        self.proj = nn.Sequential(
            nn.Linear(pool_in, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.4),
        )
        self.head = ClassificationHead(256)

    @torch.no_grad()
    def _probe(self):
        self.backbone.eval()
        dummy = torch.randn(1, len(S2_BAND_NAMES), 224, 224)
        try:
            out = self.backbone({"S2L2A": dummy})
        except Exception:
            out = self.backbone(dummy)
        feat = (
            out[-1]
            if isinstance(out, (list, tuple))
            else list(out.values())[-1]
            if isinstance(out, dict)
            else out
        )
        return feat.shape[2] if feat.dim() == 3 else feat.shape[1]

    def _extract(self, s2):
        with torch.no_grad():
            try:
                out = self.backbone({"S2L2A": s2})
            except Exception:
                out = self.backbone(s2)
        feat = (
            out[-1]
            if isinstance(out, (list, tuple))
            else list(out.values())[-1]
            if isinstance(out, dict)
            else out
        )
        if feat.dim() == 3:
            batch, tokens, channels = feat.shape
            height = width = int(tokens**0.5)
            feat = feat.transpose(1, 2).reshape(batch, channels, height, width)
        return feat

    def _ndvi_diff_stats(self, s2, corridor):
        red = s2[:, 3:4]
        nir = s2[:, 7:8]
        ndvi = ((nir - red) / (nir + red + 1e-6)).clamp(-1, 1)
        in_corridor = corridor.float()
        out_corridor = 1 - in_corridor
        n_in = in_corridor.sum(dim=[2, 3]).clamp(min=1)
        n_out = out_corridor.sum(dim=[2, 3]).clamp(min=1)
        mean_in = (ndvi * in_corridor).sum(dim=[2, 3]) / n_in
        mean_out = (ndvi * out_corridor).sum(dim=[2, 3]) / n_out
        diff = mean_in - mean_out
        sq = ((ndvi - mean_in.view(-1, 1, 1, 1)) * in_corridor) ** 2
        std = (sq.sum(dim=[2, 3]) / n_in).sqrt()
        mx = (ndvi * in_corridor + (-1) * out_corridor).flatten(2).max(dim=2)[0]
        return torch.cat([diff, std, mx], dim=1)

    def forward(self, s2, corridor):
        feat = self._extract(s2)
        corridor_feat = self.corridor_enc(corridor, feat.shape[2:])
        pooled = torch.cat(
            [
                self.gap(feat).flatten(1),
                self.gmp(feat).flatten(1),
                self.gap(corridor_feat).flatten(1),
                self.gmp(corridor_feat).flatten(1),
                self._ndvi_diff_stats(s2, corridor),
            ],
            dim=1,
        )
        return self.head(self.proj(pooled))

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self


def load_checkpoint_model(weights_path, device, pretrained_backbone=False):
    """Load the packaged checkpoint onto the requested device."""
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {weights_path}")

    model = TerraMindEncroachmentModel(
        pretrained_backbone=pretrained_backbone
    ).to(device)
    ckpt = torch.load(weights_path, map_location=device)
    state_dict = ckpt["model_state"] if "model_state" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    return model
