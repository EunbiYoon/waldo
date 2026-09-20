#!/bin/bash

set -euo pipefail

PROJECT_DIR=/work/pi_dagarwal_umass_edu/project_16/eunbi
DATASET_DIR=$PROJECT_DIR/waldo_merged_resplit

cd "$PROJECT_DIR" || exit 1

mkdir -p logs
mkdir -p results

module load conda/latest
conda activate waldo

CONFIGS=(
"--query 'Waldo wearing red and white striped shirt' --lr 1.1e-4 --epochs 12 --lambda_attr 0.1"
)

TASK_ID=0

CONFIG="${CONFIGS[$TASK_ID]}"
OUTDIR="$PROJECT_DIR/results/local_rcnn_attbind_${TASK_ID}"

mkdir -p "$OUTDIR"

echo "=================================================="
echo "Task ID: $TASK_ID"
echo "Config: $CONFIG"
echo "Output dir: $OUTDIR"
echo "Host: $(hostname)"
echo "Start time: $(date)"
echo "=================================================="

echo "Python:"
which python
python --version

echo "CUDA / GPU check:"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu count:", torch.cuda.device_count())
    print("gpu name:", torch.cuda.get_device_name(0))
else:
    raise RuntimeError("CUDA is not available.")
PY

echo "Dataset check:"
ls -lh "$DATASET_DIR/train/_annotations.coco.json"

echo "=================================================="
echo "Starting training"
echo "=================================================="

eval python -u train_frcnn_waldo.py \
    --dataset_root "$DATASET_DIR" \
    --output_dir "$OUTDIR" \
    --use_attribute_binding \
    $CONFIG

echo "=================================================="
echo "Finished successfully"
echo "End time: $(date)"
echo "=================================================="