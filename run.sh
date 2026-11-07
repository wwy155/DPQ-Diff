#!/bin/bash
# DPQ-Diff with the default configuration of the paper on one benchmark:
#   bash run.sh <dataset> [seed]
# <dataset>: ETTh1 ETTh2 ETTm1 ETTm2 Electricity ExchangeRate ILI SolarEnergy Traffic
set -euo pipefail
DATASET=${1:?usage: bash run.sh <dataset> [seed]}
SEED=${2:-1}
ROOT=$(cd "$(dirname "$0")" && pwd)
export DIFFPTS_ROOT=${DIFFPTS_ROOT:-$ROOT/DiffPTS}   # official DiffPTS implementation
export DPQ_DATA=${DPQ_DATA:-$ROOT/data}             # datasets, downloaded on first use
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
PY=${PY:-python}
BATCH_SIZE=${BATCH_SIZE:-32}   # windows per batch; lower it if the GPU runs out of memory
CKPT=$ROOT/checkpoints/$DATASET/$SEED
OUT=$ROOT/results/$DATASET/$SEED
SRC=$ROOT/dpqdiff

if [ ! -d "$DIFFPTS_ROOT/src" ]; then
  git clone https://github.com/wwy155/DiffPTS "$DIFFPTS_ROOT"
  git -C "$DIFFPTS_ROOT" checkout -q e9a7d1d56bd40b4bcfced29b430f6c5be6f2a424
fi
cd "$DIFFPTS_ROOT"

# full-precision DiffPTS forecaster
[ -f "$CKPT/model.pth" ] || $PY "$SRC/train.py" --dataset "$DATASET" --seed "$SEED" \
    --out "$CKPT" --runs "$ROOT/results/fp32_runs"

for BITS in 8 4; do
  # post-training stage: output-preserving rounding and step-aware activation ranges
  $PY "$SRC/post_training.py" --dataset "$DATASET" --seed "$SEED" --ckpt "$CKPT" \
      --out "$OUT" --w-bits $BITS --batch-size "$BATCH_SIZE"
  # distribution-preserving fine-tuning
  $PY "$SRC/finetune.py" --dataset "$DATASET" --seed "$SEED" --ckpt "$CKPT" \
      --out "$OUT" --w-bits $BITS --batch-size "$BATCH_SIZE"
done
$PY "$SRC/report.py" "$OUT"
