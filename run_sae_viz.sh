#!/bin/bash
# Run SAE feature visualization
#
# Usage:
#   ./run_sae_viz.sh                           # Use sae_fix, visualize first 10 puzzles
#   ./run_sae_viz.sh sae_fix 10                # Use sae_fix, visualize first 10 puzzles
#   ./run_sae_viz.sh tsae 5                    # Use tsae, visualize first 5 puzzles
#   ./run_sae_viz.sh sae_fix 10 "0,1,2,3"      # Visualize specific puzzle IDs
#
# Available SAE model types: sae, sae_fix, tsae, tsae_fix
#
# Generated visualizations:
#   1. sae_viz_XXXX_<name>.png - Top M features with 30x30 spatial heatmaps
#   2. sae_top_XXXX_<name>.png - Top 20 most active features overview
#   3. spatial/<name>/ - Full (64x64)×(30x30) spatial maps for iterations 1, 8, 16

# Parse arguments
SAE_MODEL_TYPE=${1:-"sae_fix"}
NUM_VIZ=${2:-10}
PUZZLE_IDS=${3:-""}

# TRM checkpoint  
TRM_CKPT="ckpt/arc_v1_public/step_518071"

# Config path (for hydra)
CONFIG_PATH="ckpt/arc_v1_public"
CONFIG_NAME="all_config"

# Data path
DATA_PATH="data/arc1concept-aug-1000"

# Output directory (includes model type)
OUTPUT_DIR="sae_visualizations_${SAE_MODEL_TYPE}"

echo "========================================"
echo "SAE Feature Visualization"
echo "========================================"
echo "SAE Model Type: $SAE_MODEL_TYPE"
echo "TRM checkpoint: $TRM_CKPT"
echo "Data path: $DATA_PATH"
echo "Num visualize: $NUM_VIZ"
echo "Puzzle IDs: ${PUZZLE_IDS:-'(first $NUM_VIZ)'}"
echo "Output dir: $OUTPUT_DIR"
echo "========================================"

# Build command
CMD="python visualize_sae_features.py \
    --config-path=$CONFIG_PATH \
    --config-name=$CONFIG_NAME \
    load_checkpoint=$TRM_CKPT \
    data_paths=[$DATA_PATH] \
    data_paths_test=[$DATA_PATH] \
    ++sae_model_type=$SAE_MODEL_TYPE \
    ++output_dir=$OUTPUT_DIR \
    ++num_visualize=$NUM_VIZ \
    ++top_m_features=50 \
    global_batch_size=1"

# Add puzzle_ids if specified
if [ -n "$PUZZLE_IDS" ]; then
    CMD="$CMD ++puzzle_ids=[$PUZZLE_IDS]"
fi

echo ""
echo "Running: $CMD"
echo ""

eval $CMD
