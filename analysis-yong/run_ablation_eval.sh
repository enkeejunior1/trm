#!/bin/bash
#SBATCH --job-name=ablation_eval
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=b200-mig45,b200-mig90,dgx-b200-old-driver
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00

# =============================================================================
# Ablation Evaluation Script with Visualization
# =============================================================================
# This script runs ablation-eval.py which:
# 1. Loads the TRM model and runs inference
# 2. Collects z_L trajectories for SAE analysis
# 3. Saves batch data (predictions, trajectories, metrics)
# 4. Creates visualizations for each prediction
# 5. Generates summary statistics
#
# Usage:
#   sbatch analysis-yong/run_ablation_eval.sh
#   OR
#   bash analysis-yong/run_ablation_eval.sh
# =============================================================================

# Environment setup
module purge
source /vast/projects/jgu32/lab/yhpark/miniconda3/etc/profile.d/conda.sh
conda activate trm

set -e

# Change to project directory
cd /vast/projects/jgu32/lab/yhpark/trm

# Create logs directory
mkdir -p logs

# =============================================================================
# CONFIGURATION - Modify these as needed
# =============================================================================

# Data paths
DATA_PATH="data/arc1concept-aug-0"           # Non-augmented test data
MODEL_DATA_PATH="data/arc1concept-aug-1000"  # Augmented data (for model training)

# Model checkpoint
CHECKPOINT="ckpt/arc_v1_public/step_518071"

# Batch size (adjust based on GPU memory)
BATCH_SIZE=32

# Output directory name (will be created under results/)
OUTPUT_NAME="ablation_eval_viz"

# =============================================================================
# Parse command line arguments (optional overrides)
# =============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_path)
            DATA_PATH="$2"
            shift 2
            ;;
        --model_data_path)
            MODEL_DATA_PATH="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --output_name)
            OUTPUT_NAME="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--data_path PATH] [--model_data_path PATH] [--checkpoint PATH] [--batch_size N] [--output_name NAME]"
            exit 1
            ;;
    esac
done

# =============================================================================
# Print configuration
# =============================================================================
echo "=============================================="
echo "Ablation Evaluation with Visualization"
echo "=============================================="
echo "Data path: ${DATA_PATH}"
echo "Model data path: ${MODEL_DATA_PATH}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Batch size: ${BATCH_SIZE}"
echo "Output name: ${OUTPUT_NAME}"
echo "=============================================="
echo ""

# =============================================================================
# Run ablation-eval.py with hydra config overrides
# =============================================================================
# The script uses hydra, so we pass config overrides as key=value pairs

python ablation-eval.py \
    data_paths="['${MODEL_DATA_PATH}']" \
    data_paths_test="['${DATA_PATH}']" \
    load_checkpoint="${CHECKPOINT}" \
    checkpoint_path="ckpt/${OUTPUT_NAME}" \
    global_batch_size=${BATCH_SIZE}

echo ""
echo "=============================================="
echo "Ablation evaluation complete!"
echo "=============================================="
echo ""
echo "Results saved to:"
echo "  - Batch data: results/${OUTPUT_NAME}/"
echo "  - Visualizations: results/${OUTPUT_NAME}/visualizations/"
echo "  - Summary: results/${OUTPUT_NAME}/visualizations/evaluation_summary.png"
echo "  - Results JSON: results/${OUTPUT_NAME}/visualizations/visualization_results.json"
echo "=============================================="
