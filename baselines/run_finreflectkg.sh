#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# One-shot runner: preprocess FinReflectKG -> train TTransE -> evaluate.
#
# Usage:
#   cd Methods/RE-Net/baselines
#   bash run_finreflectkg.sh                 # quick smoke run (10 epochs, dim 50)
#   FULL=1 bash run_finreflectkg.sh          # full run (500 epochs, dim 100)
#
# Env vars you can override:
#   SRC      path to the TSV (default: AI_SupplyChain_RA/.../three_relations_sample2000.tsv)
#   DIM      embedding dimension          (default: 50  / FULL=100)
#   EPOCHS   training epochs              (default: 10  / FULL=500)
#   BS       batch size                   (default: 128)
#   LR       learning rate                (default: 0.001)
#   MARGIN   margin for ranking loss      (default: 1.0)
#   CUDA     CUDA device id (or "" = CPU) (default: "")
# -----------------------------------------------------------------------------

set -euo pipefail

# Resolve repo paths regardless of where the script is invoked.
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

SRC="${SRC:-$HERE/../../../FinReflect/finreflectkg_subset/three_relations_sample2000.tsv}"
DATA_DIR="$HERE/data/FinReflectKG_TTransE"
MODEL_DIR="$HERE/model/FinReflectKG"

if [[ "${FULL:-0}" == "1" ]]; then
    DIM="${DIM:-100}"
    EPOCHS="${EPOCHS:-500}"
else
    DIM="${DIM:-50}"
    EPOCHS="${EPOCHS:-10}"
fi
BS="${BS:-128}"
LR="${LR:-0.001}"
MARGIN="${MARGIN:-1.0}"
CUDA="${CUDA:-}"

echo "=========================================================="
echo "[run_finreflectkg] src      = $SRC"
echo "[run_finreflectkg] data dir = $DATA_DIR"
echo "[run_finreflectkg] model dir= $MODEL_DIR"
echo "[run_finreflectkg] dim=$DIM  epochs=$EPOCHS  bs=$BS  lr=$LR  margin=$MARGIN"
echo "=========================================================="

# 1) Preprocess (idempotent — re-runs are cheap)
python3 preprocess_finreflectkg.py --src "$SRC" --out "$DATA_DIR" --seed 42

# 2) Make sure the model checkpoint dir exists
mkdir -p "$MODEL_DIR"

# 3) Train + evaluate.
#    Note: TTransE.py reads from './data/<dataset>_TTransE', so we pass -d FinReflectKG.
CUDA_VISIBLE_DEVICES="$CUDA" python3 TTransE.py \
    -d FinReflectKG \
    -L 1 \
    -em "$DIM" \
    -bs "$BS" \
    -l "$LR" \
    -m "$MARGIN" \
    -n "$EPOCHS" \
    -f 1
