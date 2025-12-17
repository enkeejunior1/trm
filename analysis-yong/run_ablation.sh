#!/bin/bash
#SBATCH --job-name=trm
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=b200-mig45,b200-mig90,dgx-b200-old-driver
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00

# Environment setup
module purge
source /vast/projects/jgu32/lab/yhpark/miniconda3/etc/profile.d/conda.sh
conda activate trm

set -e

# Default parameters
DATA_PATH="data/arc1concept-aug-0"
MODEL_DATA_PATH="data/arc1concept-aug-1000"
CHECKPOINT="ckpt/arc_v1_public/step_518071"
CONFIG_PATH="ckpt/arc_v1_public"
SAE_CHECKPOINT="weights/sae/best_val.pt"
OUTPUT_DIR="results-analysis-noaug/ablation_visualizations"
NUM_EXAMPLES=10
BATCH_SIZE=32
TOPK_ABLATE=20
IMPORTANCE_METHOD="mean_activation"

# SAE model parameters (should match training config)
SAE_DEPTH=16
SAE_D_MODEL=512
SAE_N_FEATURES=4096
SAE_TOPK=64

# Parse command line arguments
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
        --config_path)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --sae_checkpoint)
            SAE_CHECKPOINT="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --num_examples)
            NUM_EXAMPLES="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --topk_ablate)
            TOPK_ABLATE="$2"
            shift 2
            ;;
        --importance_method)
            IMPORTANCE_METHOD="$2"
            shift 2
            ;;
        --only_incorrect)
            ONLY_INCORRECT="--only_incorrect"
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "Running Ablation Analysis"
echo "=============================================="
echo "Data path: ${DATA_PATH}"
echo "Model data path: ${MODEL_DATA_PATH}"
echo "TRM checkpoint: ${CHECKPOINT}"
echo "SAE checkpoint: ${SAE_CHECKPOINT}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Number of examples: ${NUM_EXAMPLES}"
echo "Batch size: ${BATCH_SIZE}"
echo "Top-k ablate: ${TOPK_ABLATE}"
echo "Importance method: ${IMPORTANCE_METHOD}"
echo "=============================================="

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Run ablation analysis
python analysis-yong/visualize_ablation.py \
    --data_path "${DATA_PATH}" \
    --model_data_path "${MODEL_DATA_PATH}" \
    --checkpoint "${CHECKPOINT}" \
    --config_path "${CONFIG_PATH}" \
    --sae_checkpoint "${SAE_CHECKPOINT}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_examples "${NUM_EXAMPLES}" \
    --batch_size "${BATCH_SIZE}" \
    --topk_ablate "${TOPK_ABLATE}" \
    --importance_method "${IMPORTANCE_METHOD}" \
    --sae_depth "${SAE_DEPTH}" \
    --sae_d_model "${SAE_D_MODEL}" \
    --sae_n_features "${SAE_N_FEATURES}" \
    --sae_topk "${SAE_TOPK}" \
    ${ONLY_INCORRECT:-}

echo "=============================================="
echo "Ablation analysis complete!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "=============================================="
