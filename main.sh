#!/bin/bash
#SBATCH --job-name=trm
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=b200-mig45,b200-mig90
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00

# Environment setup
module purge
source /vast/projects/jgu32/lab/yhpark/miniconda3/etc/profile.d/conda.sh
conda activate trm
cd $SLURM_SUBMIT_DIR

# ============================================================================
# python eval.py \
#   --config-path=ckpt/arc_v1_public \
#   --config-name=all_config \
#   load_checkpoint=ckpt/arc_v1_public/step_518071 \
#   data_paths="[data/arc1concept-aug-1000]" \
#   data_paths_test="[data/arc1concept-aug-1000]" \
#   global_batch_size=1 \
#   checkpoint_path=results/eval_viz \
#   ++visualize=true \
#   ++num_visualize=20 \
#   ++puzzle_ids="[0,778,1779,2780,3781,4782,5784,6785,6857,7858,8859,9860,10861,11862,12863,13864,14865,15866,16867,17443]" \
#   ++visualization_output_dir=results/visualizations

# ============================================================================
# SAE Feature Visualization
# Generates:
#   1. sae_viz_XXXX_<name>.png - Top M features with 30x30 spatial heatmaps
#   2. sae_top_XXXX_<name>.png - Top 20 most active features overview
#   3. spatial/<name>/ - Full (64x64)×(30×30) spatial maps for iterations 1,8,16
#
# Available SAE model types: sae,sae_fix,tsae,tsae_fix
# ============================================================================
# # SAE Model Type (change this to visualize different models)
# for SAE_MODEL_TYPE in "sae" "sae_fix" "tsae" "tsae_fix"; do

# # Sort features by activation at this iteration (1-16),or 0 for total
# SORT_BY_ITERATION=1

# # Only consider answer tokens (where label != -100) for feature selection
# ANSWER_ONLY=true

# # Output directory (will include model type)
# OUTPUT_DIR="sae_visualizations_${SAE_MODEL_TYPE}"

# rm -rf ${OUTPUT_DIR}
# python visualize_sae_features.py \
#   --config-path=ckpt/arc_v1_public \
#   --config-name=all_config \
#   load_checkpoint=ckpt/arc_v1_public/step_518071 \
#   data_paths="[data/arc1concept-aug-1000]" \
#   data_paths_test="[data/arc1concept-aug-1000]" \
#   global_batch_size=1 \
#   ++sae_model_type=${SAE_MODEL_TYPE} \
#   ++output_dir=${OUTPUT_DIR} \
#   ++num_visualize=20 \
#   ++top_m_features=50 \
#   ++sort_by_iteration=${SORT_BY_ITERATION} \
#   ++answer_only=${ANSWER_ONLY} \
#   ++puzzle_ids="[0,778,1779,2780,3781,4782,5784,6785,6857,7858,8859,9860,10861,11862,12863,13864,14865,15866,16867,17443]"
# done

# ============================================================================
# SAE Feature Ablation Study
# 
# This script performs ablation studies to understand causal role of SAE features:
#   1. Select features to ablate (top_k,random,specific,bottom_k)
#   2. Remove their contribution from z_L during inference
#   3. Compare predictions before/after ablation
#
# Ablation modes:
#   - top_k: Ablate top-k most active features (default)
#   - bottom_k: Ablate least active (but non-zero) features
#   - random: Randomly select features to ablate
#   - specific: Ablate user-specified feature indices
#
# Available SAE model types: sae,sae_fix,tsae,tsae_fix
# ============================================================================

# SAE Model Type
SAE_MODEL_TYPE="sae_fix"

# ============================================================================
# Mode: all_individually - Ablate EACH feature one by one
# This tests every active feature individually and saves visualizations
# only when accuracy changes (improves or degrades)
# ============================================================================
# ABLATION_MODE="all_individually"

# # Output directory
# OUTPUT_DIR="ablation_individual_${SAE_MODEL_TYPE}"

# rm -rf ${OUTPUT_DIR}
# python ablation_sae_features.py \
#   --config-path=ckpt/arc_v1_public \
#   --config-name=all_config \
#   load_checkpoint=ckpt/arc_v1_public/step_518071 \
#   data_paths="[data/arc1concept-aug-1000]" \
#   data_paths_test="[data/arc1concept-aug-1000]" \
#   global_batch_size=1 \
#   ++sae_model_type=${SAE_MODEL_TYPE} \
#   ++output_dir=${OUTPUT_DIR} \
#   ++ablation_mode=${ABLATION_MODE} \
#   ++only_active_features=true \
#   ++save_only_on_change=true \
#   ++answer_only=true \
#   ++num_visualize=20 \
#   ++puzzle_ids="[0,778,1779,2780,3781,4782,5784,6785,6857,7858,8859,9860,10861,11862,12863,13864,14865,15866,16867,17443]"

# ============================================================================
# Mode: progressive - Progressive Ablation with K=1,2,3,...
# Visualizes ONLY when prediction changes from K to K+1
# ============================================================================
SAE_MODEL_TYPE="sae"
MAX_K=4096
ABLATION_STEP=100     # Group size: ablate N features at a time (1=one by one, 10=groups of 10)
MAX_ERROR_CHANGES=0   # Stop after N error-change points (0=no limit)
SAVE_ALL_IMAGES=true  # Save images for all K (not just when errors change)
SORT_BY_ITERATION=1
ANSWER_ONLY=false
PUZZLE_IDS="[0,778,1779,2780,3781,4782,5784,6785,6857,7858,8859,9860,10861,11862,12863,13864,14865,15866,16867,17443]"
NUM_VISUALIZE=20

# ------------------------------------------------------------------------------
# Metric 1: avg_activation - Rank by sum of activation values
# Higher activation sum = More important feature
# ------------------------------------------------------------------------------
RANKING_METRIC="avg_activation"
OUTPUT_DIR="ablation_progressive_${SAE_MODEL_TYPE}_${RANKING_METRIC}_${ANSWER_ONLY}"

echo "=============================================="
echo "Running Progressive Ablation with ${RANKING_METRIC}"
echo "=============================================="

rm -rf ${OUTPUT_DIR}
python ablation_sae_features.py \
  --config-path=ckpt/arc_v1_public \
  --config-name=all_config \
  load_checkpoint=ckpt/arc_v1_public/step_518071 \
  data_paths="[data/arc1concept-aug-1000]" \
  data_paths_test="[data/arc1concept-aug-1000]" \
  global_batch_size=1 \
  ++sae_model_type=${SAE_MODEL_TYPE} \
  ++output_dir=${OUTPUT_DIR} \
  ++ablation_mode=progressive \
  ++ranking_metric=${RANKING_METRIC} \
  ++max_k_features=${MAX_K} \
  ++ablation_step=${ABLATION_STEP} \
  ++max_error_changes=${MAX_ERROR_CHANGES} \
  ++save_all_images=${SAVE_ALL_IMAGES} \
  ++sort_by_iteration=${SORT_BY_ITERATION} \
  ++answer_only=${ANSWER_ONLY} \
  ++num_visualize=${NUM_VISUALIZE} \
  ++puzzle_ids="${PUZZLE_IDS}"

# ------------------------------------------------------------------------------
# Metric 2: activation_freq - Rank by how often feature is in top-K (64)
# Higher frequency = More important feature
# ------------------------------------------------------------------------------
RANKING_METRIC="activation_freq"
OUTPUT_DIR="ablation_progressive_${SAE_MODEL_TYPE}_${RANKING_METRIC}_${ANSWER_ONLY}"

echo "=============================================="
echo "Running Progressive Ablation with ${RANKING_METRIC}"
echo "=============================================="

rm -rf ${OUTPUT_DIR}
python ablation_sae_features.py \
  --config-path=ckpt/arc_v1_public \
  --config-name=all_config \
  load_checkpoint=ckpt/arc_v1_public/step_518071 \
  data_paths="[data/arc1concept-aug-1000]" \
  data_paths_test="[data/arc1concept-aug-1000]" \
  global_batch_size=1 \
  ++sae_model_type=${SAE_MODEL_TYPE} \
  ++output_dir=${OUTPUT_DIR} \
  ++ablation_mode=progressive \
  ++ranking_metric=${RANKING_METRIC} \
  ++max_k_features=${MAX_K} \
  ++ablation_step=${ABLATION_STEP} \
  ++max_error_changes=${MAX_ERROR_CHANGES} \
  ++save_all_images=${SAVE_ALL_IMAGES} \
  ++sort_by_iteration=${SORT_BY_ITERATION} \
  ++answer_only=${ANSWER_ONLY} \
  ++num_visualize=${NUM_VISUALIZE} \
  ++puzzle_ids="${PUZZLE_IDS}"

echo "=============================================="
echo "Progressive Ablation Complete!"
echo "Results saved to:"
echo "  - ablation_progressive_${SAE_MODEL_TYPE}_avg_activation/"
echo "  - ablation_progressive_${SAE_MODEL_TYPE}_activation_freq/"
echo "=============================================="

# ============================================================================
# (Commented) Mode: top_k - Ablate top K most active features together
# ============================================================================
# SAE_MODEL_TYPE="sae"
# ABLATION_MODE="top_k"
# NUM_FEATURES_TO_ABLATE=10
# SORT_BY_ITERATION=1
# ANSWER_ONLY=true
# OUTPUT_DIR="ablation_results_${SAE_MODEL_TYPE}_${ABLATION_MODE}_${NUM_FEATURES_TO_ABLATE}"
#
# rm -rf ${OUTPUT_DIR}
# python ablation_sae_features.py \
#   --config-path=ckpt/arc_v1_public \
#   --config-name=all_config \
#   load_checkpoint=ckpt/arc_v1_public/step_518071 \
#   data_paths="[data/arc1concept-aug-1000]" \
#   data_paths_test="[data/arc1concept-aug-1000]" \
#   global_batch_size=1 \
#   ++sae_model_type=${SAE_MODEL_TYPE} \
#   ++output_dir=${OUTPUT_DIR} \
#   ++ablation_mode=${ABLATION_MODE} \
#   ++num_features_to_ablate=${NUM_FEATURES_TO_ABLATE} \
#   ++num_visualize=20 \
#   ++top_m_features=50 \
#   ++sort_by_iteration=${SORT_BY_ITERATION} \
#   ++answer_only=${ANSWER_ONLY} \
#   ++puzzle_ids="[0,778,1779,2780,3781,4782,5784,6785,6857,7858,8859,9860,10861,11862,12863,13864,14865,15866,16867,17443]"

# ============================================================================
# Example: Ablate specific features (e.g.,features 100,200,300)
# ============================================================================
# SAE_MODEL_TYPE="sae"
# python ablation_sae_features.py \
#   --config-path=ckpt/arc_v1_public \
#   --config-name=all_config \
#   load_checkpoint=ckpt/arc_v1_public/step_518071 \
#   data_paths="[data/arc1concept-aug-1000]" \
#   data_paths_test="[data/arc1concept-aug-1000]" \
#   global_batch_size=1 \
#   ++sae_model_type=$SAE_MODEL_TYPE \
#   ++output_dir=ablation_specific \
#   ++ablation_mode=specific \
#   ++specific_features="[2739,3762,1670,6,3601,1584,738,1747,1602,1505,2690,2891,3572,3124,2631,243,1049,3454,628,3804]" \
#   ++puzzle_ids="[0,778,1779,2780,3781,4782,5784,6785,6857,7858,8859,9860,10861,11862,12863,13864,14865,15866,16867,17443]" \
#   ++num_visualize=20

# ============================================================================
# Example: Ablate only at specific iterations (e.g.,iterations 0,7,15)
# ============================================================================
# python ablation_sae_features.py \
#   --config-path=ckpt/arc_v1_public \
#   --config-name=all_config \
#   load_checkpoint=ckpt/arc_v1_public/step_518071 \
#   data_paths="[data/arc1concept-aug-1000]" \
#   global_batch_size=1 \
#   ++sae_model_type=sae_fix \
#   ++output_dir=ablation_iter_specific \
#   ++ablation_mode=top_k \
#   ++num_features_to_ablate=10 \
#   ++ablation_iterations="[0,7,15]" \
#   ++num_visualize=10