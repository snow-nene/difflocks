# Pixi setup

This environment targets Linux x86_64, Python 3.10, and CUDA 12.4. It pins the PyTorch and torchvision builds used by the DiffLocks inference code and installs the remaining project dependencies through Pixi.

```bash
pixi install
pixi run python inference_difflocks.py \
  --img_path ./samples/medium_11.png \
  --out_path ./outputs_inference \
  --enable_monitoring
```

Pixi creates the environment from `pixi.toml`. The NATTEN and FlashAttention packages are dependencies; no wheel file is stored in this repository. Model checkpoints and Blender are separate prerequisites. See [Performance monitoring](PERFORMANCE_MONITORING.md) for exported files and Blender options.
