#!/bin/bash
#SBATCH --job-name=trm
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=b200-mig45,dgx-b200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00

# Environment setup
module purge
source /vast/projects/jgu32/lab/yhpark/miniconda3/etc/profile.d/conda.sh
conda activate trm
cd $SLURM_SUBMIT_DIR

# Option 1: Evaluate specific puzzles (matching visualization)
# NOTE: Change puzzle_ids to match what you visualized
python eval-stat.py \
  --config-path=ckpt/arc_v1_public \
  --config-name=all_config \
  load_checkpoint=ckpt/arc_v1_public/step_518071 \
  data_paths="[data/arc1concept-aug-1000]" \
  data_paths_test="[data/arc1concept-aug-1000]" \
  global_batch_size=1 \
  checkpoint_path=results/arc_v1_public/step_518071_eval_aug1000 \
  evaluators="[]" \
  ++puzzle_ids="[1,2,3,4,5,6,7,8,9,10]"

# python eval.py \
#   --config-path=ckpt/arc_v1_public \
#   --config-name=all_config \
#   load_checkpoint=ckpt/arc_v1_public/step_518071 \
#   data_paths="[data/arc1concept-aug-1000]" \
#   data_paths_test="[data/arc1concept-aug-1000]" \
#   global_batch_size=512 \
#   evaluators="[]"