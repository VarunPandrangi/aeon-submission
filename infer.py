#!/usr/bin/env python3
"""
Single-patch inference for the TerraMind encroachment detector.

Examples:
    python infer.py --sample karnataka_nc_g1_r0010
    python infer.py --s2 path/to/patch_s2.tif --corridor path/to/patch_corridor.tif
    python infer.py --s2 path/to/patch_s2.tif

The packaged checkpoint already contains the backbone weights. By default the
script stays offline and avoids any remote model download.
"""

import argparse
import json

import torch

from src.inference import (
    DEFAULT_THRESHOLD,
    load_input_tensors,
    load_model,
    predict,
    resolve_input_paths,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample",
        help="Sample stem from sample_input/, for example 'karnataka_nc_g1_r0010'.",
    )
    parser.add_argument(
        "--sample-dir",
        default="sample_input",
        help="Directory containing bundled or user-provided sample pairs.",
    )
    parser.add_argument(
        "--s2",
        help="Path to a Sentinel-2 GeoTIFF. If it ends with '_s2.tif', the matching corridor file is inferred automatically.",
    )
    parser.add_argument(
        "--corridor",
        help="Path to the corridor mask GeoTIFF. Optional when --s2 follows the '*_s2.tif' naming convention.",
    )
    parser.add_argument("--weights", default="checkpoints/best_model.pt")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--allow-backbone-download",
        action="store_true",
        help="Allow TerraTorch to fetch pretrained backbone weights from the internet. Default is offline execution with the bundled checkpoint.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    s2_path, corridor_path = resolve_input_paths(
        s2_path=args.s2,
        corridor_path=args.corridor,
        sample=args.sample,
        sample_dir=args.sample_dir,
    )

    model = load_model(
        args.weights,
        device=device,
        allow_backbone_download=args.allow_backbone_download,
    )
    s2_tensor, corridor_tensor = load_input_tensors(s2_path, corridor_path, device)
    result = predict(
        model,
        s2_tensor,
        corridor_tensor,
        threshold=args.threshold,
    )
    result["s2_path"] = str(s2_path)
    result["corridor_path"] = str(corridor_path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
