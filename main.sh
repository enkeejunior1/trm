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

# ============================================================================
# Option 1: Evaluate on AUGMENTED data (aug-1000) - NO DEMOS
# The arc1concept augmentation creates ~1000 examples per base puzzle.
# Total: 402 unique base puzzles, 368,150 examples
#
# CORRECTED Dataset structure (aug-1000):
#   - idx 0-777:      7c9b52a0 (778 aug)
#   - idx 778-1778:   2a5f8217 (1001 aug)
#   - idx 1779-2779:  477d2879 (1001 aug)
#   - idx 2780-3780:  b15fca0b (1001 aug)
#   - idx 3781-4781:  e5790162 (1001 aug)
#   - idx 4782-5782:  4364c1c4 (1001 aug)
#   - idx 5784-6784:  8ba14f53 (1001 aug)  # Note: 5783 is <blank>
#   - idx 6785-6856:  b9630600 (72 aug)
#   - idx 6857-7857:  15696249 (1001 aug)
#   - ... see puzzle_boundaries_corrected.txt for full list
#
# Visualization options:
#   - ++puzzle_ids="[0,778,1779,...]": Dataset indices to visualize DIFFERENT puzzles
#   - ++num_visualize=20: Limit number of visualizations
#   - ++random_visualize=true: Randomly select from dataset (uses seed)
# ============================================================================
python eval.py \
  --config-path=ckpt/arc_v1_public \
  --config-name=all_config \
  load_checkpoint=ckpt/arc_v1_public/step_518071 \
  data_paths="[data/arc1concept-aug-1000]" \
  data_paths_test="[data/arc1concept-aug-1000]" \
  global_batch_size=1 \
  checkpoint_path=results/eval_viz \
  ++visualize=true \
  ++num_visualize=20 \
  ++puzzle_ids="[0,778,1779,2780,3781,4782,5784,6785,6857,7858,8859,9860,10861,11862,12863,13864,14865,15866,16867,17443]" \
  ++visualization_output_dir=results/visualizations

# ++random_visualize=true \
# ++seed=42 \

# ============================================================================
# Option 2 (RECOMMENDED for visualization WITH demos):
# Evaluate on NON-AUGMENTED data (aug-0)
# Demos will correctly match the test examples
# ============================================================================
# python eval.py \
#   --config-path=ckpt/arc_v1_public \
#   --config-name=all_config \
#   load_checkpoint=ckpt/arc_v1_public/step_518071 \
#   data_paths="[data/arc1concept-aug-1000]" \
#   data_paths_test="[data/arc1concept-aug-0]" \
#   global_batch_size=1 \
#   checkpoint_path=results/eval_viz_with_demos \
#   ++visualize=true \
#   ++num_visualize=20 \
#   ++visualization_output_dir=results/visualizations_with_demos