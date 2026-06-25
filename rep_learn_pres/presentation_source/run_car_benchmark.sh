#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${CAR_BENCHMARK_OUTPUT_DIR:-${SCRIPT_DIR}/data/car_benchmark}"
DEVICE="${CAR_BENCHMARK_DEVICE:-cuda}"
PREPROCESS_BACKEND="${CAR_BENCHMARK_PREPROCESS_BACKEND:-cudf}"
SEEDS="${CAR_BENCHMARK_SEEDS:-7,17,29,43,71,101,131,173,211,257}"
NUM_WORKERS="${CAR_BENCHMARK_NUM_WORKERS:-0}"

cd "${SCRIPT_DIR}"

python -m run_categorical_benchmark \
  --datasets car-price \
  --full-dataset \
  --embedding-size 32 \
  --hidden-sizes 1024,8192,128 \
  --seeds "${SEEDS}" \
  --device "${DEVICE}" \
  --preprocess-backend "${PREPROCESS_BACKEND}" \
  --num-workers "${NUM_WORKERS}" \
  --output-dir "${OUTPUT_DIR}" \
  "$@"
