# Edge Inference Feasibility

## Model Size
| Format       | Size     |
|--------------|----------|
| fp32         | ~334 MB  |
| fp16         | ~167 MB  |  ← recommended for deployment
| int8 (est.)  | ~84 MB   |

Checkpoint file (head only, fp32): see checkpoints/best_model.pt

## Inference Latency
Measured on T4 GPU (Kaggle, single patch, batch=1):
- See `docs/edge_feasibility.txt` generated during evaluation cell

## Jetson AGX Orin Estimate
- T4 peak fp16: ~65 TFLOPS
- Jetson AGX Orin peak fp16: ~67 TFLOPS (similar peak, but thermal throttled on orbit)
- Conservative estimate: 3× T4 latency on Orin
- Jetson NX (8 GB RAM): model fits comfortably (167 MB << 8 GB)

## Bandwidth Saving
| Item                          | Size        |
|-------------------------------|-------------|
| Model output (1 flag + score) | ~50 bytes   |
| Raw Sentinel-2 patch (256x256x12 fp16) | ~2.4 MB |
| **Bandwidth saving**          | **~48,000×** |

## Deployment Path
1. Export model to fp16 with `model.half()`
2. Quantize head to int8 with `torch.quantization` (backbone already frozen)
3. Export to ONNX for TensorRT deployment on Jetson
4. Estimated TensorRT speedup over PyTorch: 2–4×

## What We Did Not Measure
- Actual latency on Jetson hardware (not available)
- Power draw per inference
- TensorRT-optimized latency

These are honest gaps. The back-of-envelope numbers above are based on
published Jetson AGX Orin specs vs T4 FLOP counts.
