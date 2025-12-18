#!/bin/bash
#SBATCH --job-name=trm-noaug
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

echo "=============================================="
echo "Non-Augmented Validation Evaluation"
echo "=============================================="
echo "This script evaluates the TRM model on non-augmented validation data"
echo "while using the puzzle embeddings trained on the augmented dataset."
echo ""
echo "Key insight:"
echo "  - data_paths: aug-1000 (for model initialization - puzzle embeddings)"
echo "  - data_paths_test: aug-0 (actual test data - non-augmented)"
echo "  - Puzzle identifiers are remapped from aug-0 to aug-1000 indices"
echo ""
echo "Embedding modes:"
echo "  - 'original': Use ORIGINAL (non-augmented) embedding from aug-1000"
echo "  - 'first_augmented': Use FIRST augmented embedding (consistent + trained)"
echo "  - 'random_augmented': Random augmented embedding (causes variance!)"
echo "=============================================="
echo ""

# Embedding mode: "original", "first_augmented", or "random_augmented"
# - "original": Use ORIGINAL embedding from aug-1000 (barely trained - 1/1001 of data)
# - "first_augmented": Use FIRST augmented embedding (well-trained, consistent)
# - "random_augmented": Random augmented embedding (causes variance)
EMBEDDING_MODE="first_augmented"

# Run non-augmented evaluation
python eval_noaug.py \
  --config-path=ckpt/arc_v1_public \
  --config-name=all_config \
  load_checkpoint=ckpt/arc_v1_public/step_518071 \
  data_paths="[data/arc1concept-aug-1000]" \
  data_paths_test="[data/arc1concept-aug-1000]" \
  global_batch_size=1 \
  checkpoint_path=results/arc_v1_public/step_518071_eval_noaug_${EMBEDDING_MODE} \
  evaluators="[]" \
  ++embedding_mode="${EMBEDDING_MODE}"

echo ""
echo "=============================================="
echo "Evaluation complete!"
echo "Results saved to: results/arc_v1_public/step_518071_eval_noaug"
echo "=============================================="

# Optional: Evaluate specific puzzle IDs (uncomment to use)
# python eval_noaug.py \
#   --config-path=ckpt/arc_v1_public \
#   --config-name=all_config \
#   load_checkpoint=ckpt/arc_v1_public/step_518071 \
#   data_paths="[data/arc1concept-aug-1000]" \
#   data_paths_test="[data/arc1concept-aug-0]" \
#   global_batch_size=1 \
#   checkpoint_path=results/arc_v1_public/step_518071_eval_noaug_subset \
#   evaluators="[]" \
#   embedding_mode="original" \
#   ++puzzle_ids="[1,2,3,4,5,6,7,8,9,10]"

# ============================================
# Compare all embedding modes (uncomment to run)
# ============================================
# for MODE in original first_augmented random_augmented; do
#   echo "Running with embedding_mode=${MODE}"
#   python eval_noaug.py \
#     --config-path=ckpt/arc_v1_public \
#     --config-name=all_config \
#     load_checkpoint=ckpt/arc_v1_public/step_518071 \
#     data_paths="[data/arc1concept-aug-1000]" \
#     data_paths_test="[data/arc1concept-aug-0]" \
#     global_batch_size=1 \
#     checkpoint_path=results/arc_v1_public/step_518071_eval_noaug_${MODE} \
#     evaluators="[]" \
#     embedding_mode="${MODE}"
# done
