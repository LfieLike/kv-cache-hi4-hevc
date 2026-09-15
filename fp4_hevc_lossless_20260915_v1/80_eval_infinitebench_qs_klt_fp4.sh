#!/usr/bin/env bash
# InfiniteBench Key FP4 Query-sensitive V_r sweep.
# Reuses the book GPA A and qs_r* matrices from script 79. Evaluates
# unrotated / A / every qs_r on book Split-B under fp4-e2m1.
# Writes BOOK_QS_KLT_FP4_*. Does not overwrite the recorded INT8 tree
# or official RULER FP4 2.953 / 2.069 / 1.693.
#
# STAGES="eval summarize"   (default; skip align/fit)
# QUANTIZER=fp4-e2m1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export QUANTIZER="${QUANTIZER:-fp4-e2m1}"
export STAGES="${STAGES:-eval summarize}"

if [ "$QUANTIZER" != "fp4-e2m1" ]; then
  echo "script 80 is FP4-only; got QUANTIZER=$QUANTIZER"
  exit 1
fi

exec bash "$SCRIPT_DIR/79_eval_infinitebench_qs_klt.sh"
