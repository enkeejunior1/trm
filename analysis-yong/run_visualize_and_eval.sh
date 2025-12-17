#!/bin/bash
#SBATCH --job-name=trm-viz-eval
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=b200-mig45,b200-mig90,dgx-b200-old-driver
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00

# Combined script: Visualize puzzles AND evaluate them
# This ensures we measure accuracy on the EXACT same puzzles we visualize

# Environment setup
module purge
source /vast/projects/jgu32/lab/yhpark/miniconda3/etc/profile.d/conda.sh
conda activate trm

cd $SLURM_SUBMIT_DIR
CHECKPOINT_PATH=ckpt/arc_v1_public/step_518071

# Change to base directory
echo "============================================================"
echo "Step 1: Visualizing ARC-AGI-1 puzzles"
echo "============================================================"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Dataset: data/arc1concept-aug-1000"
echo ""

# Choose ONE of the following options:

# Option A: Visualize first N puzzles
PUZZLE_IDS="1,2,3,4,5,6,7,8,9,10"
python3 analysis-yong/visualize_test_data.py \
  --data_path data/arc1concept-aug-1000 \
  --puzzle_ids "$PUZZLE_IDS"

# Option B: Visualize specific puzzle IDs
# PUZZLE_IDS="1,10,50,100,200"
# python3 analysis-yong/visualize_test_data.py \
#   --data_path data/arc1concept-aug-1000 \
#   --puzzle_ids "$PUZZLE_IDS"

# Option C: Visualize random puzzles
# python3 analysis-yong/visualize_test_data.py \
#   --data_path data/arc1concept-aug-1000 \
#   --num_puzzles 10 \
#   --random \
#   --seed 42

echo ""
echo "Visualization complete!"
echo "Results saved to: results-analysis/visualizations/"
echo ""

echo "============================================================"
echo "Step 2: Evaluating exact accuracy on visualized puzzles"
echo "============================================================"

# Convert comma-separated IDs to hydra list format
# "1,2,3,4,5" -> "[1,2,3,4,5]"
PUZZLE_IDS_HYDRA="[$PUZZLE_IDS]"

python3 eval-stat.py \
  --config-path=ckpt/arc_v1_public \
  --config-name=all_config \
  load_checkpoint=ckpt/arc_v1_public/step_518071 \
  data_paths="[data/arc1concept-aug-1000]" \
  data_paths_test="[data/arc1concept-aug-1000]" \
  global_batch_size=1 \
  checkpoint_path=results/arc_v1_public/step_518071_eval_visualized \
  evaluators="[]" \
  ++puzzle_ids="$PUZZLE_IDS_HYDRA"

echo ""
echo "============================================================"
echo "Complete! Results:"
echo "============================================================"
echo "Visualizations: results-analysis/visualizations/"
echo "Accuracy stats: results/arc_v1_public/step_518071_eval_visualized/puzzle_id_stats.yaml"
echo "============================================================"

