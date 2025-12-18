#!/bin/bash
#SBATCH --job-name=trm-viz
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=b200-mig45,b200-mig90
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00

# Environment setup
module purge
source /vast/projects/jgu32/lab/yhpark/miniconda3/etc/profile.d/conda.sh
conda activate trm

# Get the script directory
# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# BASE_DIR="$(dirname "$SCRIPT_DIR")"
# cd "$BASE_DIR"
cd $SLURM_SUBMIT_DIR

echo "=============================================="
echo "ARC Puzzle Visualization with TRM Predictions"
echo "=============================================="
# echo "Base directory: $BASE_DIR"
# echo ""

# Visualize first 20 puzzles with model predictions
python analysis-yong/visualize_with_predictions.py \
  --data_path data/arc1concept-aug-1000 \
  --model_data_path data/arc1concept-aug-1000 \
  --checkpoint ckpt/arc_v1_public/step_518071 \
  --config_path ckpt/arc_v1_public \
  --output_dir results-analysis-noaug/visualizations_with_predictions \
  --num_puzzles 20

echo ""
echo "=============================================="
echo "Visualization complete!"
echo "Results: results-analysis-noaug/visualizations_with_predictions/"
echo "=============================================="

# To visualize specific puzzle IDs, use:
# python analysis-yong/visualize_with_predictions.py \
#   --data_path data/arc1concept-aug-0 \
#   --model_data_path data/arc1concept-aug-1000 \
#   --checkpoint ckpt/arc_v1_public/step_518071 \
#   --config_path ckpt/arc_v1_public \
#   --output_dir results-analysis-noaug/visualizations_with_predictions \
#   --puzzle_ids "1,2,3,4,5,6,7,8,9,10"

# To visualize a range of puzzle IDs:
# python analysis-yong/visualize_with_predictions.py \
#   --data_path data/arc1concept-aug-0 \
#   --model_data_path data/arc1concept-aug-1000 \
#   --checkpoint ckpt/arc_v1_public/step_518071 \
#   --config_path ckpt/arc_v1_public \
#   --output_dir results-analysis-noaug/visualizations_with_predictions \
#   --start_id 0 --end_id 100
