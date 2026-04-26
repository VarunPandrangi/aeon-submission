#!/usr/bin/env python3
"""
Minimal Streamlit demo for the packaged encroachment model.

The model needs a Sentinel-2 patch and a matching corridor mask, so the app
supports both bundled demo samples and user uploads for that pair.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
import torch

from src.inference import DEFAULT_THRESHOLD, load_input_tensors, load_model, predict, resolve_input_paths


BASE_DIR = Path(__file__).resolve().parent
SAMPLE_DIR = BASE_DIR / "sample_input"
WEIGHTS_PATH = BASE_DIR / "checkpoints" / "best_model.pt"


st.set_page_config(page_title="Encroachment Demo", layout="centered")
st.markdown(
    """
    <style>
      #MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
      header {visibility: hidden;}
      .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 900px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def list_demo_samples(sample_dir: Path) -> list[str]:
    return sorted(
        path.name.removesuffix("_s2.tif")
        for path in sample_dir.glob("*_s2.tif")
    )


def save_upload(uploaded_file, session_key: str) -> Path | None:
    if uploaded_file is None:
        st.session_state.pop(session_key, None)
        return None

    data = uploaded_file.getvalue()
    signature = (uploaded_file.name, len(data))
    cached = st.session_state.get(session_key)
    if cached and cached["signature"] == signature:
        return Path(cached["path"])

    suffix = Path(uploaded_file.name).suffix or ".tif"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)

    st.session_state[session_key] = {"signature": signature, "path": str(tmp_path)}
    return tmp_path


def _read_tiff(path: Path) -> np.ndarray:
    try:
        import rasterio

        with rasterio.open(path) as ds:
            return ds.read() if ds.count > 1 else ds.read(1)
    except ModuleNotFoundError:
        try:
            import tifffile
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Install either rasterio or tifffile+imagecodecs to read the "
                "GeoTIFF demo files in Kaggle or Colab."
            ) from exc

        data = tifffile.imread(path)
        if data.ndim == 2:
            return data
        if data.shape[0] <= 16:
            return data
        if data.shape[-1] <= 16:
            return data.transpose(2, 0, 1)
        return data


def normalize_rgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.nan_to_num(rgb.astype(np.float32))
    if rgb.max() > 1.5:
        rgb = rgb / 10000.0

    out = np.zeros_like(rgb, dtype=np.float32)
    for idx in range(3):
        channel = rgb[..., idx]
        lo, hi = np.percentile(channel, (2, 98))
        if hi <= lo:
            out[..., idx] = 0.0
        else:
            out[..., idx] = np.clip((channel - lo) / (hi - lo), 0.0, 1.0)
    return out


def preview_s2(path: Path) -> np.ndarray:
    data = _read_tiff(path)
    if data.ndim == 2:
        rgb = np.repeat(data[:, :, None], 3, axis=2)
    elif data.shape[0] >= 4:
        rgb = data[[3, 2, 1]].transpose(1, 2, 0)
    else:
        rgb = np.repeat(data[:1].transpose(1, 2, 0), 3, axis=2)
    return normalize_rgb(rgb)


def preview_corridor(path: Path) -> np.ndarray:
    data = _read_tiff(path)
    if data.ndim == 3:
        data = data[0]
    return data.astype(np.float32)


@st.cache_resource
def get_model(weights_path: str, device_name: str):
    device = torch.device(device_name)
    model = load_model(weights_path, device=device, allow_backbone_download=False)
    return model, device


st.title("Encroachment Demo")
st.caption("Minimal Streamlit front end for the packaged checkpoint.")

device_name = "cuda" if torch.cuda.is_available() else "cpu"

sample_stems = list_demo_samples(SAMPLE_DIR)
if not sample_stems:
    st.error(f"No bundled demo samples found in {SAMPLE_DIR}")
    st.stop()

mode = st.radio("Input source", ["Bundled demo", "Upload files"], horizontal=True)

s2_path = None
corridor_path = None

if mode == "Bundled demo":
    default_sample = "karnataka_nc_g1_r0010" if "karnataka_nc_g1_r0010" in sample_stems else sample_stems[0]
    selected_sample = st.selectbox("Demo sample", sample_stems, index=sample_stems.index(default_sample))
    s2_path, corridor_path = resolve_input_paths(
        sample=selected_sample,
        sample_dir=SAMPLE_DIR,
    )
else:
    s2_upload = st.file_uploader("Sentinel-2 GeoTIFF", type=["tif", "tiff"])
    corridor_upload = st.file_uploader("Corridor mask GeoTIFF", type=["tif", "tiff"])
    s2_path = save_upload(s2_upload, "uploaded_s2_path")
    corridor_path = save_upload(corridor_upload, "uploaded_corridor_path")

preview_ready = s2_path is not None and corridor_path is not None

if preview_ready:
    try:
        left, right = st.columns(2)
        with left:
            st.image(preview_s2(s2_path), caption="Sentinel-2 preview", use_container_width=True)
        with right:
            corridor_img = preview_corridor(corridor_path)
            st.image(corridor_img, caption="Corridor mask", clamp=True, use_container_width=True)
    except ModuleNotFoundError as exc:
        st.error(str(exc))
else:
    st.info("Provide both files to preview and run the model.")

run = st.button("Run model", type="primary", disabled=not preview_ready)

if run and preview_ready:
    with st.spinner("Loading model and running inference..."):
        model, device = get_model(str(WEIGHTS_PATH), device_name)
        s2_tensor, corridor_tensor = load_input_tensors(s2_path, corridor_path, device)
        result = predict(model, s2_tensor, corridor_tensor, threshold=DEFAULT_THRESHOLD)

    st.subheader("Output")
    st.code(json.dumps(result, indent=2), language="json")
    st.write(
        f"Prediction: {'Encroached' if result['encroached'] else 'Not encroached'}"
        f" | Probability: {result['probability']:.4f}"
        f" | Threshold: {result['threshold_used']:.2f}"
    )
