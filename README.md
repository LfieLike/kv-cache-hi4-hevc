# Our no-rotation high4-HEVC KV-cache codec

This is a distinct variant of the no-rotation pipeline. It keeps the existing
target-scaled symmetric INT8 quantizer and 8-token-per-frame layout, then splits
each uint8 code into nibbles:

- high 4 bits: placed in the Y plane and compressed by lossless libx265 HEVC;
- low 4 bits: packed two symbols per byte and stored verbatim in the packet.

Decode extracts the high-nibble plane, unpacks the raw low nibbles, recombines
the exact INT8 codes, and then dequantizes. Thus HEVC adds no distortion; the
only lossy stage remains the INT8 quantizer. This variant reports actual NRMSE
but does **not** enforce a strict global error bound or retry on overshoot.

## Run

From the project root, with a completed Qwen/GLM/DeepSeek source run:

```bash
python -m paper_bench.bench_no_rotation_hi4 \
  --source /path/to/completed-run \
  --output /path/to/new-output \
  --ffmpeg /path/to/ffmpeg
```

Requires PyTorch with CUDA, NumPy, safetensors, Matplotlib, and FFmpeg built
with `libx265`. The source cache is read-only. Use a fresh output directory
when changing codec variants.

The RD output separates `low4_bytes` from `hevc_bytes`; total bpv includes
both, packet headers, and BF16 scales. No compression-ratio or quality claim
is made until this variant is run on the measured cache corpus.

## Included implementation

- `paper_bench/hevc_token_hi4.py`: packet, nibble split/merge, HEVC encode and
  decode.
- `paper_bench/bench_no_rotation_hi4.py`: entry point for the shared RD harness.
- `paper_bench/bench_kvcodec.py`, `paper_bench/hevc_token.py`,
  `paper_bench/summarize.py`: shared quantizer, cache-dump evaluation
  harness, original token layout helpers, and report generation.
- `paper_bench/fit_gpa_from_dump.py` plus `sz_kv_bench/fresh_group4_fit.py`
  and `sz_kv_bench/group4_gpa_math.py`: the repository's current group-GPA
  fitting method, packaged as a standalone fit stage for an existing
  `paper_bench/run.py collect` dump.

## Optional GPA fit

To fit one transform per group from the collected calibration Split-A data:

```bash
python -m paper_bench.fit_gpa_from_dump \
  --dump-root /path/to/completed-collection \
  --output /path/to/new-fit \
  --device cuda
```

The fitter checks source hashes, uses 100 generalized-Procrustes iterations
with `1e-6` convergence tolerance by default, and writes `groupXX.npz` files
plus `fit.json` (including the pooled FP8-reference error). It centers each
view, unit-normalizes token rows for fitting, and anchors the first view to
identity to fix the common-rotation gauge. **This optional fit is not consumed
by the no-rotation high4 codec above**; it is included for a future rotated
variant/ablation and does not change the current encoder's packet format.
