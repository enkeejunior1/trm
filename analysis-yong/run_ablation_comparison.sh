#!/bin/bash
#SBATCH --job-name=ablation_cmp
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=b200-mig45,b200-mig90,dgx-b200-old-driver
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00

# =============================================================================
# Ablation Comparison: Original vs SAE-Ablated Predictions
# =============================================================================
# This script compares model predictions:
# 1. Original predictions (no SAE ablation)
# 2. Predictions after ablating top-k SAE features
#
# Usage:
#   sbatch analysis-yong/run_ablation_comparison.sh
#   OR
#   bash analysis-yong/run_ablation_comparison.sh
#   OR with custom parameters:
#   bash analysis-yong/run_ablation_comparison.sh --topk 20 --max_batches 5
# =============================================================================

# Environment setup
module purge
source /vast/projects/jgu32/lab/yhpark/miniconda3/etc/profile.d/conda.sh
conda activate trm

set -e
cd /vast/projects/jgu32/lab/yhpark/trm
mkdir -p logs

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_PATH="data/arc1concept-aug-0"
MODEL_DATA_PATH="data/arc1concept-aug-1000"
CHECKPOINT="ckpt/arc_v1_public/step_518071"
SAE_CHECKPOINT="weights/sae/best_val.pt"
BATCH_SIZE=32
TOPK_ABLATE=20
MAX_BATCHES=5
MAX_EXAMPLES=10
OUTPUT_NAME="ablation_comparison"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --topk) TOPK_ABLATE="$2"; shift 2 ;;
        --max_batches) MAX_BATCHES="$2"; shift 2 ;;
        --max_examples) MAX_EXAMPLES="$2"; shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --output_name) OUTPUT_NAME="$2"; shift 2 ;;
        --sae_checkpoint) SAE_CHECKPOINT="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

echo "=============================================="
echo "Ablation Comparison: Original vs SAE-Ablated"
echo "=============================================="
echo "Data: ${DATA_PATH}"
echo "Model: ${CHECKPOINT}"
echo "SAE: ${SAE_CHECKPOINT}"
echo "Top-k ablate: ${TOPK_ABLATE}"
echo "Max batches: ${MAX_BATCHES}"
echo "Max examples/batch: ${MAX_EXAMPLES}"
echo "=============================================="

python ablation_comparison_eval.py \
    data_paths="['${MODEL_DATA_PATH}']" \
    data_paths_test="['${DATA_PATH}']" \
    load_checkpoint="${CHECKPOINT}" \
    checkpoint_path="ckpt/${OUTPUT_NAME}" \
    global_batch_size=${BATCH_SIZE} \
    +sae_checkpoint="${SAE_CHECKPOINT}" \
    +topk_ablate=${TOPK_ABLATE} \
    +max_batches=${MAX_BATCHES} \
    +max_examples_per_batch=${MAX_EXAMPLES}

echo ""
echo "=============================================="
echo "Done! Results saved to:"
echo "  results/${OUTPUT_NAME}/ablation_comparison/"
echo "=============================================="
