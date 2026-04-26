"""
Reusable inference helpers for local demo execution.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.model import load_checkpoint_model

KEEP_BANDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11]
S2_NORM = 10000.0
TM_SIZE = 224
DEFAULT_THRESHOLD = 0.30


def resolve_input_paths(s2_path=None, corridor_path=None, sample=None, sample_dir="sample_input"):
    sample_dir = Path(sample_dir)

    if sample:
        s2_path = sample_dir / f"{sample}_s2.tif"
        corridor_path = sample_dir / f"{sample}_corridor.tif"
    elif s2_path and corridor_path is None:
        s2_path = Path(s2_path)
        if s2_path.name.endswith("_s2.tif"):
            corridor_path = s2_path.with_name(
                s2_path.name.replace("_s2.tif", "_corridor.tif")
            )
        else:
            raise ValueError(
                "--corridor is required when --s2 does not use the '*_s2.tif' naming convention."
            )
    elif s2_path and corridor_path:
        s2_path = Path(s2_path)
        corridor_path = Path(corridor_path)
    else:
        raise ValueError("Provide either --sample or --s2.")

    if not s2_path.exists():
        raise FileNotFoundError(f"Sentinel-2 file not found: {s2_path}")
    if not corridor_path.exists():
        raise FileNotFoundError(f"Corridor mask not found: {corridor_path}")
    return s2_path, corridor_path


def _resize_array(arr, mode):
    tensor = torch.from_numpy(arr).unsqueeze(0)
    kwargs = {"align_corners": False} if mode == "bilinear" else {}
    return (
        F.interpolate(tensor, (TM_SIZE, TM_SIZE), mode=mode, **kwargs)
        .squeeze(0)
        .numpy()
    )


def _read_tiff(path, band=0):
    try:
        import rasterio

        with rasterio.open(path) as ds:
            if band == 0:
                data = ds.read()
                return data if data.ndim == 3 else data[np.newaxis, :, :]
            return ds.read(band)
    except ModuleNotFoundError:
        try:
            import tifffile
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Install either rasterio or tifffile+imagecodecs before running inference."
            ) from exc

        data = tifffile.imread(path)
        if band == 0:
            if data.ndim == 2:
                return data[np.newaxis, :, :]
            if data.shape[0] <= 16:
                return data
            if data.shape[-1] <= 16:
                return data.transpose(2, 0, 1)
            return data
        if data.ndim == 2:
            return data
        if data.shape[0] <= 16:
            return data[band - 1]
        if data.shape[-1] <= 16:
            return data[..., band - 1]
        return data


def load_input_tensors(s2_path, corridor_path, device):
    s2 = _read_tiff(s2_path).astype(np.float32) / S2_NORM
    s2 = np.clip(s2[KEEP_BANDS], 0, 3.0)

    corridor = _read_tiff(corridor_path, band=1).astype(np.float32)[np.newaxis]

    s2_tensor = torch.from_numpy(_resize_array(s2, "bilinear")).unsqueeze(0).to(device)
    corridor_tensor = torch.from_numpy(
        (_resize_array(corridor, "nearest") > 0.5).astype(np.float32)
    ).unsqueeze(0).to(device)
    return s2_tensor, corridor_tensor


def predict(model, s2_tensor, corridor_tensor, threshold=DEFAULT_THRESHOLD):
    with torch.no_grad():
        logit = model(s2_tensor, corridor_tensor)
    probability = torch.sigmoid(logit).item()
    return {
        "encroached": int(probability >= threshold),
        "probability": round(probability, 4),
        "threshold_used": threshold,
    }


def load_model(weights_path, device, allow_backbone_download=False):
    return load_checkpoint_model(
        weights_path,
        device=device,
        pretrained_backbone=allow_backbone_download,
    )
