#!/bin/bash
#SBATCH --job-name=trm
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=b200-mig45
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --array=1

# Environment setup
module purge
source /vast/projects/jgu32/lab/yhpark/miniconda3/etc/profile.d/conda.sh
conda activate trm
cd $SLURM_SUBMIT_DIR

# Actual work
echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"

if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
  python sae.py \
    --config-path=ckpt/arc_v1_public \
    --config-name=all_config \
    load_checkpoint=ckpt/arc_v1_public/step_518071 \
    data_paths="[data/arc1concept-aug-1000]" \
    data_paths_test="[data/arc1concept-aug-1000]" \
    global_batch_size=512 \
    arch.halt_max_steps=16 \
    checkpoint_path=ckpt/arc_v1_public \
    +num_epochs=1000 \
    +split=train
fi 

if [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
  python tsae.py \
    --config-path=ckpt/arc_v1_public \
    --config-name=all_config \
    load_checkpoint=ckpt/arc_v1_public/step_518071 \
    data_paths="[data/arc1concept-aug-1000]" \
    data_paths_test="[data/arc1concept-aug-1000]" \
    global_batch_size=512 \
    arch.halt_max_steps=16 \
    checkpoint_path=ckpt/arc_v1_public \
    +num_epochs=1000 \
    +split=train
fi