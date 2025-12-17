#!/bin/bash
#SBATCH --job-name=trm-compare
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=b200-mig45,dgx-b200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=4:00:00

# Environment setup
module purge
source /vast/projects/jgu32/lab/yhpark/miniconda3/etc/profile.d/conda.sh
conda activate trm
cd $SLURM_SUBMIT_DIR

echo "=============================================="
echo "Comparing Embedding Modes for Non-Augmented Evaluation"
echo "=============================================="

for MODE in original first_augmented; do
  echo ""
  echo "========================================"
  echo "Running with embedding_mode=${MODE}"
  echo "========================================"
  
  python eval_noaug.py \
    --config-path=ckpt/arc_v1_public \
    --config-name=all_config \
    load_checkpoint=ckpt/arc_v1_public/step_518071 \
    data_paths="[data/arc1concept-aug-1000]" \
    data_paths_test="[data/arc1concept-aug-0]" \
    global_batch_size=1 \
    checkpoint_path=results/arc_v1_public/step_518071_eval_noaug_${MODE} \
    evaluators="[]" \
    ++embedding_mode="${MODE}"
    
  echo ""
  echo "Completed ${MODE} mode"
done

echo ""
echo "=============================================="
echo "Comparison complete! Check:"
echo "  - results/arc_v1_public/step_518071_eval_noaug_original/"
echo "  - results/arc_v1_public/step_518071_eval_noaug_first_augmented/"
echo "=============================================="
