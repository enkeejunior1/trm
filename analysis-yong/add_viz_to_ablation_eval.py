"""
Code snippet to add visualization to ablation_eval.py

This file shows exactly what code to add and where to add it.
Copy the relevant sections into your ablation_eval.py.

Author: Claude (helping with visualization integration)
"""

# =============================================================================
# STEP 1: Add these imports at the TOP of ablation_eval.py (after existing imports)
# =============================================================================
"""
# Add these lines after your existing imports:
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'analysis-yong'))
try:
    from viz_utils import (
        decode_arc_grid,
        visualize_prediction_comparison,
        visualize_batch_results,
        create_summary_figure,
        save_results_json,
        check_prediction_correct,
    )
    VIZ_AVAILABLE = True
except ImportError:
    print("Warning: viz_utils not available, visualization disabled")
    VIZ_AVAILABLE = False
"""


# =============================================================================
# STEP 2: Add a helper function for visualization (add before evaluate())
# =============================================================================
def add_visualization_to_evaluation():
    """
    Add this function to your ablation_eval.py and call it after getting predictions.
    """
    # This is pseudo-code showing the structure - adapt variable names to your code
    
    # Assuming you have these variables from your ablation pipeline:
    # - inputs: np.ndarray [B, 900] - tokenized test inputs
    # - labels: np.ndarray [B, 900] - tokenized test labels  
    # - original_predictions: np.ndarray [B, 900] - predictions before ablation
    # - ablated_predictions: np.ndarray [B, 900] - predictions after ablation
    # - original_losses: list of floats - loss per example before ablation
    # - ablated_losses: list of floats - loss per example after ablation
    # - puzzle_identifiers: np.ndarray [B] - puzzle ID for each example
    # - identifiers_map: list[str] - maps puzzle ID to name
    # - test_puzzles: dict - puzzle_name -> {train: [...], test: [...]}
    # - ablated_feature_indices: list[int] - indices of ablated features
    # - OUTPUT_DIR: str - where to save visualizations
    
    pass  # This is just documentation


# =============================================================================
# STEP 3: Add this code block inside your main evaluation loop
# =============================================================================
VISUALIZATION_CODE_BLOCK = """
# =============================================================================
# VISUALIZATION CODE - Add this after getting original and ablated predictions
# =============================================================================

if VIZ_AVAILABLE:
    print("\\n" + "="*60)
    print("Creating Visualizations")
    print("="*60)
    
    VIZ_DIR = os.path.join(RESULT_DIR, "visualizations")
    os.makedirs(VIZ_DIR, exist_ok=True)
    
    viz_results = []
    
    # Load puzzle metadata for demo examples
    with open(os.path.join(args.data_path, "identifiers.json"), 'r') as f:
        identifiers_map = json.load(f)
    
    with open(os.path.join(args.data_path, "test_puzzles.json"), 'r') as f:
        test_puzzles = json.load(f)
    
    # Visualize each example
    for i in range(min(len(original_predictions), args.max_viz)):
        # Get puzzle info
        puzzle_id = int(puzzle_ids[i]) if puzzle_ids is not None else i
        if puzzle_id < len(identifiers_map):
            puzzle_name = identifiers_map[puzzle_id]
        else:
            puzzle_name = f"puzzle_{i}"
        
        # Get base name (without augmentation suffix)
        base_name = puzzle_name.split("|||")[0] if "|||" in puzzle_name else puzzle_name
        
        # Get demo examples from test_puzzles.json
        demos = test_puzzles.get(base_name, {}).get("train", [])
        
        # Create comparison visualization
        result = visualize_prediction_comparison(
            test_input=inputs[i],
            test_label=labels[i],
            original_pred=original_predictions[i],
            ablated_pred=ablated_predictions[i],
            demo_examples=demos[:4],  # Limit to 4 demos
            puzzle_name=base_name,
            output_path=os.path.join(VIZ_DIR, f"ablation_{i:03d}_{base_name}.png"),
            original_loss=original_losses[i] if original_losses else None,
            ablated_loss=ablated_losses[i] if ablated_losses else None,
            ablated_features=ablated_feature_indices,
            puzzle_id=puzzle_id,
        )
        viz_results.append(result)
        
        # Print progress
        change_str = result.get('change', 'unchanged')
        orig_status = "✓" if result['original_correct'] else "✗"
        abl_status = "✓" if result['ablated_correct'] else "✗"
        print(f"  [{i+1}] {base_name}: {orig_status} → {abl_status} ({change_str})")
    
    # Create summary figure
    create_summary_figure(
        viz_results,
        os.path.join(VIZ_DIR, "ablation_summary.png"),
        title="Ablation Analysis Summary",
        topk=len(ablated_feature_indices)
    )
    print(f"\\nSaved summary to: {os.path.join(VIZ_DIR, 'ablation_summary.png')}")
    
    # Save results to JSON
    save_results_json(
        viz_results,
        config={
            'topk_ablate': len(ablated_feature_indices),
            'importance_method': args.importance_method if hasattr(args, 'importance_method') else 'mean_activation',
            'num_examples': len(viz_results),
            'ablated_features': ablated_feature_indices,
        },
        output_path=os.path.join(VIZ_DIR, "ablation_results.json")
    )
    
    # Print final summary
    n_improved = sum(1 for r in viz_results if r.get('change') == 'improved')
    n_degraded = sum(1 for r in viz_results if r.get('change') == 'degraded')
    n_unchanged = len(viz_results) - n_improved - n_degraded
    
    print("\\n" + "="*60)
    print("VISUALIZATION SUMMARY")
    print("="*60)
    print(f"Total visualized: {len(viz_results)}")
    print(f"Improved (wrong→correct): {n_improved}")
    print(f"Degraded (correct→wrong): {n_degraded}")
    print(f"Unchanged: {n_unchanged}")
    print(f"Visualizations saved to: {VIZ_DIR}")
    print("="*60)
"""


# =============================================================================
# STEP 4: Alternative - Minimal integration (just add after predictions)
# =============================================================================
MINIMAL_INTEGRATION = """
# Minimal visualization - add right after getting predictions

import sys
sys.path.insert(0, 'analysis-yong')
from viz_utils import decode_arc_grid, visualize_grid
import matplotlib.pyplot as plt

# Quick visualization of a single prediction
def quick_viz(input_tokens, label_tokens, pred_tokens, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    input_grid = decode_arc_grid(input_tokens)
    label_grid = decode_arc_grid(label_tokens)
    pred_grid = decode_arc_grid(pred_tokens)
    
    visualize_grid(input_grid, axes[0], "Input")
    visualize_grid(label_grid, axes[1], "Target")
    
    is_correct = np.array_equal(label_grid, pred_grid)
    color = '#2ECC40' if is_correct else '#FF4136'
    status = "✓ Correct" if is_correct else "✗ Wrong"
    visualize_grid(pred_grid, axes[2], f"Prediction ({status})", highlight_color=color)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return is_correct

# Usage:
# is_correct = quick_viz(inputs[i], labels[i], predictions[i], f"viz/pred_{i}.png")
"""


# =============================================================================
# EXAMPLE: Complete integration pattern
# =============================================================================
if __name__ == "__main__":
    print("This file contains code snippets for integration.")
    print("Copy the relevant sections into your ablation_eval.py")
    print("\nSee VISUALIZATION_GUIDE.md for detailed instructions.")
