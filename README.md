# Power Line Corridor Encroachment Detection

This repository packages an inference-first demo for corridor encroachment
detection on Sentinel-2 patches using a fine-tuned TerraMind backbone.

The checkpoint in `checkpoints/best_model.pt` is bundled with the repository.
The judge workflow is notebook-based only: run it on Kaggle or Colab with a T4
GPU.

## Judge Quickstart

Target workflow:
- open a Kaggle or Colab notebook
- enable a T4 GPU
- install dependencies in the notebook
- run one demo command or one inference command
- get JSON output in under 10 minutes total setup + run time

Recommended runtime:
- Python 3.11.x
- Kaggle notebook with a T4 GPU, or Colab notebook with a T4 GPU
- `runtime.txt` records the target interpreter family used for the packaged path

## Judge Setup

Use one of the notebook flows in `docs/platform_setup.md`.

Run the bundled smoke test:

```bash
python demo.py
```

Run one bundled sample:

```bash
python infer.py --sample karnataka_nc_g1_r0010
```

Run one custom sample pair:

```bash
python infer.py --s2 path/to/patch_s2.tif --corridor path/to/patch_corridor.tif
```

If your Sentinel-2 file follows the `*_s2.tif` naming convention and the
matching `*_corridor.tif` sits next to it, `--corridor` is optional:

```bash
python infer.py --s2 path/to/patch_s2.tif
```

## What The Demo Expects

This model does not take a raw image alone. It requires:
- one Sentinel-2 patch GeoTIFF
- one corridor mask GeoTIFF for the same patch

The CLI now makes that explicit and can auto-resolve the corridor mask when the
file names follow the packaged naming convention.

Bundled sample files live in `sample_input/`:
- `*_s2.tif` for the Sentinel-2 patch
- `*_corridor.tif` for the corridor mask
- `sample_labels.csv` for expected labels on the demo set

## Repository Entry Points

- `infer.py`: single-sample inference CLI
- `demo.py`: bundled multi-sample smoke test with JSON summary
- `src/model.py`: model definition and checkpoint loader
- `src/inference.py`: reusable preprocessing and prediction helpers
- `train.py`: points to the original notebook-based training flow

## Model Summary

Architecture:
- TerraMind-1.0-base backbone
- frozen backbone features
- corridor encoder CNN
- pooled backbone + corridor features
- NDVI difference statistics
- MLP binary classifier head

Packaged config summary from `configs/train_config.json`:
- AOI split: train on 6 AOIs, val on `karnataka_ne`, test on `karnataka_nc`
- image size: `224`
- threshold: `0.30`
- best packaged test metrics:
  - AUC: `0.7866`
  - AP: `0.9048`
  - F1: `0.8305`
  - accuracy: `0.7674`

## Reproducibility Scope

This repository supports reproducible notebook execution for the packaged
checkpoint and bundled samples on Kaggle or Colab with a T4 GPU.

Full training reproducibility is still notebook-based, not script-based:
- `train.py` is only a stub
- the full training flow lives in `notebooks/notebook- aeon.ipynb`
- the notebook expects notebook-hosted data and internet-backed model access

So there are two different claims:
- packaged demo reproducibility in Kaggle or Colab: supported here
- full retraining reproducibility from scratch: not yet fully productized

## Known Limits

- The model needs a corridor mask, not just a raw satellite patch.
- The current demo path is optimized for verification, not retraining.
- Jetson latency numbers in `docs/edge_feasibility.md` are still estimates.
