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

CHECKPOINT_PATH=ckpt/arc_v1_public/step_518071

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"

# Change to base directory
cd "$BASE_DIR"

echo "Running ARC-AGI-1 visualization..."
echo "Base directory: $BASE_DIR"
echo "Checkpoint: $CHECKPOINT_PATH"
echo ""

# Run visualization script from analysis-yong directory
cd $SLURM_SUBMIT_DIR

# NOTE: Using aug-1000 to match eval-stat.py (checkpoint was trained on aug-1000)
python analysis-yong/visualize_test_data.py \
  --data_path data/arc1concept-aug-1000 \
  --num_puzzles 5

# To visualize specific puzzle IDs (matching eval-stat.py):
# python analysis-yong/visualize_test_data.py \
#   --data_path data/arc1concept-aug-1000 \
#   --puzzle_ids "1,10,50,100"

echo ""
echo "Visualization complete!"
echo "Results saved to: results-analysis/visualizations/"
echo ""
echo "Now run eval-stat.py with the same puzzle_ids to get accuracy!"
echo "Example: python eval-stat.py ... ++puzzle_ids=[1,10,50,100]"