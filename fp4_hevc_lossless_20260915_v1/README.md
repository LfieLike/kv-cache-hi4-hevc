# FP4 + lossless HEVC KV-cache codec

This package contains the earlier rotated FP4 E2M1 + exact lossless x265 evaluation implementation. It is distinct from the later high-nibble-only HEVC experiment.

## Pipeline

1. Load synchronized KV slabs and fitted per-layer/per-head orthogonal rotations.
2. Rotate activations and quantize to monotonic-mapped FP4 E2M1 codes, with scale metadata.
3. Store the 4-bit symbols as gray8 pixels in canonical layer-major/head-minor views.
4. Encode with lossless x265/HEVC and verify byte-exact decode; report bitstream plus scale-metadata size and QK/Key distortion.

## Files

- `src/evaluate_best_h265.py`: main exact FP4 + HEVC evaluation entry point.
- `src/best_codec_core.py`, `src/codec_aware_rotation.py`: quantization, rotation application and distortion/rate metrics.
- `src/codec_rotation_data.py`, `src/cross_layer_data.py`: synchronized data loading.
- `src/codec_exact_h265.py`, `src/evaluate_h265_cross_layer_codec.py`: raw view layout, x265 encoding and exact round-trip verification.
- `src/evaluate_cross_layer_alignment.py`, `src/alignment_math.py`, `src/fp8_byte_layouts.py`, `src/int8_deadzone.py`, `src/x265_defaults.py`: supporting modules required by the implementation.
- `src/summarize_best_h265.py`: summarize an evaluation JSON.
- `results_reference/infinitebench_key_qs_klt_fp4_20260823.json`: recorded InfiniteBench Split-B Key FP4 Q-S KLT results (a separate comparison run, not an output of `evaluate_best_h265.py`).
- `scripts/80_eval_infinitebench_qs_klt_fp4.sh`: historical repo launcher for the Q-S KLT comparison; it expects the original repository environment and external experiment data.

## Requirements

- Python 3.10+
- PyTorch
- `safetensors`
- `ffmpeg` built with `libx265`
- Input manifests/shards and a compatible rotation `.safetensors` file (not included)

Install Python dependencies with `pip install -r requirements.txt`.

## Run

From the extracted package directory, for example:

```bash
PYTHONPATH=src python src/evaluate_best_h265.py \
  --manifests /path/to/layer-manifest.json [additional-manifests...] \
  --rotations /path/to/cross_layer_rotations.safetensors \
  --output-dir /path/to/fp4-hevc-results \
  --device cuda
```

The manifest files must cover the requested layers and synchronized sequences. Defaults evaluate layers `3 7 11 15 19 23 27 31` and use 2048-token slabs. Use `--max-sequences 1 --max-slabs 1` for a smoke test. Set `--device cpu` if CUDA is unavailable. The exact evaluation writes `best_h265_summary.json` and `.csv`; temporary raw/decoded gray8 files are removed unless `--keep-raw` is supplied.

To print a compact summary:

```bash
python src/summarize_best_h265.py /path/to/fp4-hevc-results
```

The included reference JSON reports `codec: lossless x265 GOP-32 gray8`; do not conflate its InfiniteBench figures with the separate official RULER FP4 figures.
