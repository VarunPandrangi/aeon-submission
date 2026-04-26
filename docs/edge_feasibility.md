# Edge Inference Feasibility

## Packaged Artifact

The repository ships `checkpoints/best_model.pt`, which contains the full model
state dict used by the inference demo. The single-sample inference path can run
offline once the Python environment is installed.

## Model Size

| Format | Size |
|---|---:|
| Packaged checkpoint (fp32) | ~335 MB |
| Estimated fp16 export | ~167 MB |
| Estimated int8 export | ~84 MB |

## Current Deployment Claim

What is supported by this repository today:
- local CPU or GPU inference for packaged samples
- local checkpoint loading without a runtime backbone download
- one-command smoke testing with `python demo.py`

What is still an estimate or future work:
- Jetson latency measurements on real hardware
- TensorRT export and benchmark numbers
- power draw per inference

## Notes

The existing orbital-compute argument is still directionally valid, but the
reproducible artifact in this repo is the local inference path rather than a
fully benchmarked embedded deployment package.
