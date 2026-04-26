#!/usr/bin/env python3
"""
train.py — Fine-tuning TerraMind for encroachment detection

Usage:
    python train.py

Reads data from paths configured in CONFIG dict at bottom of file.
Saves best checkpoint to checkpoints/best_model.pt

See README.md for full pipeline description.
"""
# Full training code lives in the Kaggle notebook.
# This file is the reference entry point for reproducibility.
# To retrain: open notebooks/training.ipynb on Kaggle with 2x T4 GPUs
# and run all cells in order.
#
# Key hyperparameters:
#   epochs=60, batch_size=16, lr=2e-4, weight_decay=1e-2
#   warmup_epochs=5, patience=15
#   focal_loss gamma=2.0, label_smoothing=0.05
#   WeightedRandomSampler (enc=0 oversampled 7.8x)
#
# AOI split:
#   Train: karnataka_c, karnataka_n, karnataka_nw, karnataka_sw, karnataka_se, karnataka_w
#   Val  : karnataka_ne
#   Test : karnataka_nc

print("See notebooks/training.ipynb for full training code.")
print("Run on Kaggle: https://kaggle.com  with 2x T4 GPU accelerator.")
