#!/usr/bin/env python3
"""
Run the bundled sample set end-to-end and report predictions plus latency.
"""

import argparse
import csv
import json
import time
from pathlib import Path

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
    parser.add_argument("--sample-dir", default="sample_input")
    parser.add_argument("--labels", default="sample_input/sample_labels.csv")
    parser.add_argument("--weights", default="checkpoints/best_model.pt")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--allow-backbone-download",
        action="store_true",
        help="Allow TerraTorch to fetch pretrained backbone weights from the internet.",
    )
    parser.add_argument(
        "--output",
        help="Optional path to write the JSON summary.",
    )
    return parser.parse_args()


def load_labels(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
    return rows


def main():
    args = parse_args()
    device = torch.device(args.device)
    labels = load_labels(args.labels)
    model = load_model(
        args.weights,
        device=device,
        allow_backbone_download=args.allow_backbone_download,
    )

    predictions = []
    correct = 0
    total_latency_ms = 0.0

    for row in labels:
        stem = row["stem"]
        expected = int(row["encroached"])
        s2_path, corridor_path = resolve_input_paths(
            sample=stem,
            sample_dir=args.sample_dir,
        )

        start = time.perf_counter()
        s2_tensor, corridor_tensor = load_input_tensors(s2_path, corridor_path, device)
        result = predict(
            model,
            s2_tensor,
            corridor_tensor,
            threshold=args.threshold,
        )
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        total_latency_ms += latency_ms

        row_result = {
            "stem": stem,
            "expected_encroached": expected,
            "predicted_encroached": result["encroached"],
            "probability": result["probability"],
            "latency_ms": latency_ms,
        }
        predictions.append(row_result)
        correct += int(expected == result["encroached"])

    summary = {
        "device": str(device),
        "weights": str(Path(args.weights)),
        "threshold_used": args.threshold,
        "num_samples": len(predictions),
        "num_correct": correct,
        "accuracy": round(correct / len(predictions), 4) if predictions else 0.0,
        "total_latency_ms": round(total_latency_ms, 2),
        "avg_latency_ms": round(total_latency_ms / len(predictions), 2)
        if predictions
        else 0.0,
        "predictions": predictions,
    }

    payload = json.dumps(summary, indent=2)
    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
