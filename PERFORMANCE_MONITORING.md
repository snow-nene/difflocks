# Inference performance monitoring

The inference entry point supports optional per-stage wall time and process/GPU memory reporting. Monitoring is disabled by default. Install the project dependencies, then pass `--enable_monitoring`:

```bash
python inference_difflocks.py \
  --img_path ./samples/medium_11.png \
  --out_path ./outputs_inference \
  --enable_monitoring
```

The report is written to `performance_report.txt`; the measurements are also saved as `performance_metrics.json` in the output directory. Tracked stages include model initialization, face landmark detection, DINOv2 feature extraction, scalp texture diffusion, strand decoding, material prediction, binary export, and optional Blender export. GPU memory measurements use PyTorch's allocator counters; CPU measurements sample process RSS at stage boundaries.

## Hair export

Inference writes `hair_binary.data` with a little-endian strand count followed by each strand's vertex count and float32 XYZ coordinates. By default it exports up to 10,000 strands, sampled evenly from the generated set. Set `--max_num_strands` to change that limit. The usual NPZ/JSON/PNG intermediates are still written unless `--slim_output` is selected.

To create a Blender scene or Alembic export, provide `--blender_path` and optionally `--export_alembic`. Blender export reads the NPZ intermediate, so do not combine it with `--slim_output`.
