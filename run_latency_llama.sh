#!/usr/bin/env bash
# Figure-3-style end-to-end latency sweep on Llama-3.2-1B: dense FlashAttention vs
# SSA (chunk16) vs H-SSA (chunk4-chunk4). Correctness guard first, then one sweep
# per config (appended to one CSV), then the two-panel TTFT/TPOT plot.
#
# Usage:
#   bash run_latency_llama.sh
#   LENGTHS="4096 8192 16384" REPEATS=5 bash run_latency_llama.sh
#   SKIP_CHECK=1 bash run_latency_llama.sh
#
# Needs g++ for the one-time JIT build of kernels/sparse_decode_attn.cu (~30s, cached).
# Set TORCH_CUDA_ARCH_LIST to your GPU (8.9 = L4, 9.0 = H100).
set -e
cd "$(dirname "$0")"
export PYTHONPATH="$(pwd):$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
# torch JIT-builds the decode kernel with $CXX (default `c++`, absent on some boxes)
export CXX="${CXX:-$(command -v c++ || command -v g++ || command -v g++-12)}"
export PREFILL_KERNEL="${PREFILL_KERNEL:-flex}"
export DECODE_SELECT="${DECODE_SELECT:-head}"
PY="${PY:-.venv/bin/python}"
export PATH="$(pwd)/.venv/bin:$PATH"   # ninja for the JIT build

LENGTHS="${LENGTHS:-4096 8192 12288 16384 24576 32768}"
GEN_TOKENS="${GEN_TOKENS:-64}"
REPEATS="${REPEATS:-3}"
OUT="${OUT:-results/latency_llama1b_$(date +%Y%m%d_%H%M%S).csv}"

if [ "${SKIP_CHECK:-0}" != "1" ]; then
  echo ">>> correctness: sparse (head) vs eager reference"
  "$PY" benchmark/check_llama_sparse.py
fi

for cfg in dense ssa hssa; do
  echo ">>> sweep: $cfg"
  "$PY" benchmark/latency_sweep.py --config "$cfg" --lengths $LENGTHS \
    --gen_tokens "$GEN_TOKENS" --repeats "$REPEATS" --out "$OUT"
done

"$PY" benchmark/plot_latency.py "$OUT"
