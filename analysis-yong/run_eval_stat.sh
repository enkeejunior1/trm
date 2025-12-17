#!/bin/bash
#SBATCH --job-name=trm-eval
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=b200-mig45,b200-mig90,dgx-b200-old-driver
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00

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

echo "Running ARC-AGI-1 evaluation with statistics..."
echo "Base directory: $BASE_DIR"
echo "Checkpoint: $CHECKPOINT_PATH"
echo ""

# Run eval-stat.py
# NOTE: checkpoint_path must be a directory (different from checkpoint file name)
# IMPORTANT: Use aug-1000 for accurate results (matches training data)

# Option 1: Evaluate with aug-1000 (RECOMMENDED - matches training data)
python eval-stat.py \
  --config-path=ckpt/arc_v1_public \
  --config-name=all_config \
  load_checkpoint=ckpt/arc_v1_public/step_518071 \
  data_paths="[data/arc1concept-aug-1000]" \
  data_paths_test="[data/arc1concept-aug-1000]" \
  global_batch_size=4096 \
  checkpoint_path=results/arc_v1_public/step_518071_eval_aug1000 \
  evaluators="[]"

# Option 2: Evaluate specific puzzle_ids with aug-1000
# python eval-stat.py \
#   --config-path=ckpt/arc_v1_public \
#   --config-name=all_config \
#   load_checkpoint=ckpt/arc_v1_public/step_518071 \
#   data_paths="[data/arc1concept-aug-1000]" \
#   data_paths_test="[data/arc1concept-aug-1000]" \
#   global_batch_size=4096 \
#   checkpoint_path=results/arc_v1_public/step_518071_eval_aug1000 \
#   evaluators="[]" \
#   ++puzzle_ids=[1,10,50,100]

# Option 3: Evaluate with aug-0 (WARNING: Will FAIL if checkpoint trained on different dataset!)
# NOTE: This checkpoint was trained on aug-1000, so using aug-0 will raise ValueError
# python eval-stat.py \
#   --config-path=ckpt/arc_v1_public \
#   --config-name=all_config \
#   load_checkpoint=ckpt/arc_v1_public/step_518071 \
#   data_paths="[data/arc1concept-aug-0]" \
#   data_paths_test="[data/arc1concept-aug-0]" \
#   global_batch_size=4096 \
#   checkpoint_path=results/arc_v1_public/step_518071_eval_aug0 \
#   evaluators="[]"

echo ""
echo "Evaluation complete!"
echo "Results saved to: results/arc_v1_public/step_518071_eval_aug1000/"
echo "  - puzzle_id_stats.yaml: Per-puzzle statistics"
echo "  - metrics.yaml: Overall metrics"
echo "  - batch_data_*.pt: Batch-level data with trajectories"

