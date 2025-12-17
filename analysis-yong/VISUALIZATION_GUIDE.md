# Visualization Guide for Ablation Analysis

This guide explains how to add visualization to your ablation pipeline.

## Quick Start

### Option 1: Run standalone visualization on existing results

If you already have `ablation_results.json`:

```bash
python analysis-yong/visualize_ablation_predictions.py \
    --results_json results-analysis-noaug/ablation_visualizations/ablation_results.json \
    --data_path data/arc1concept-aug-0 \
    --output_dir results-analysis-noaug/new_visualizations
```

### Option 2: Integrate into ablation_eval.py

Add these imports at the top of your `ablation_eval.py`:

```python
# Add after existing imports
import sys
sys.path.insert(0, 'analysis-yong')
from viz_utils import (
    decode_arc_grid,
    visualize_prediction_comparison,
    visualize_batch_results,
    create_summary_figure,
    save_results_json,
    check_prediction_correct
)
```

Then add visualization after your ablation loop:

```python
# After getting original_pred and ablated_pred for each example:
result = visualize_prediction_comparison(
    test_input=inputs[i],           # tokenized input [900]
    test_label=labels[i],           # tokenized label [900]
    original_pred=original_pred,    # original prediction [900]
    ablated_pred=ablated_pred,      # ablated prediction [900]
    demo_examples=demo_examples,    # list of dicts with 'input', 'output'
    puzzle_name=puzzle_name,
    output_path=f"{output_dir}/ablation_{i:03d}_{puzzle_name}.png",
    original_loss=orig_loss,
    ablated_loss=abl_loss,
    ablated_features=topk_indices.tolist(),
    puzzle_id=puzzle_id,
)
results.append(result)

# After all examples:
create_summary_figure(results, f"{output_dir}/summary.png", topk=20)
save_results_json(results, config_dict, f"{output_dir}/results.json")
```

## File Descriptions

### `viz_utils.py`
Core visualization utilities:
- `decode_arc_grid(tokens)` - Decode tokens [900] to grid [H, W]
- `visualize_grid(grid, ax, title)` - Draw ARC grid on matplotlib axis
- `visualize_prediction_comparison(...)` - Full comparison figure
- `visualize_batch_results(...)` - Visualize entire batch
- `create_summary_figure(results, output_path)` - Summary statistics
- `save_results_json(results, config, output_path)` - Save to JSON
- `check_prediction_correct(pred, label)` - Check if correct

### `visualize_ablation_predictions.py`
Standalone script for visualization:
- Works with `ablation_results.json` files
- Works with `batch_data_*.pt` files
- Creates individual puzzle visualizations
- Creates summary figures

### `visualize_ablation.py`
Full ablation pipeline with visualization (existing):
- Loads models, runs inference, ablates features
- Creates all visualizations automatically

## Integration Example

Here's a complete integration snippet for your `ablation_eval.py`:

```python
# Add to evaluate() function after collecting predictions

# Import visualization utilities
import sys
sys.path.insert(0, 'analysis-yong')
from viz_utils import (
    visualize_prediction_comparison,
    create_summary_figure,
    save_results_json
)

# Inside your evaluation loop, after ablation:
viz_results = []

for i, (orig_pred, abl_pred, loss_orig, loss_abl) in enumerate(ablation_results):
    # Get puzzle info
    puzzle_id = int(puzzle_ids_arr[i])
    puzzle_name = identifiers_map[puzzle_id] if puzzle_id < len(identifiers_map) else f"puzzle_{i}"
    base_name = puzzle_name.split("|||")[0]
    
    # Get demo examples
    demos = test_puzzles.get(base_name, {}).get("train", [])
    
    # Create visualization
    result = visualize_prediction_comparison(
        test_input=inputs[i],
        test_label=labels[i],
        original_pred=orig_pred,
        ablated_pred=abl_pred,
        demo_examples=demos,
        puzzle_name=base_name,
        output_path=f"{OUTPUT_DIR}/ablation_{i:03d}_{base_name}.png",
        original_loss=loss_orig,
        ablated_loss=loss_abl,
        ablated_features=topk_features,
        puzzle_id=puzzle_id,
    )
    viz_results.append(result)
    
    print(f"[{i+1}] {base_name}: {result['change']}")

# Create summary
create_summary_figure(
    viz_results, 
    f"{OUTPUT_DIR}/summary.png",
    title="Ablation Analysis Summary",
    topk=len(topk_features)
)

# Save results
save_results_json(
    viz_results,
    {'topk_ablate': len(topk_features), 'method': 'mean_activation'},
    f"{OUTPUT_DIR}/ablation_results.json"
)
```

## Understanding the Visualization

Each visualization shows:

1. **Demo Examples** (top rows)
   - Input → Output pairs showing the pattern to learn

2. **Test Target** (middle row)
   - Input | Target (Ground Truth) with blue border

3. **Original Prediction** (row after target)
   - Model's prediction before ablation
   - Green border = correct, Red border = incorrect
   - White X marks show cells that differ from target

4. **Ablated Prediction** (bottom row)
   - Model's prediction after ablating top-k features
   - Shows loss change compared to original

## Summary Figure

The summary figure shows:
- **Accuracy comparison**: Before/after ablation bar chart
- **Loss distribution**: Histogram of loss changes
- **Change breakdown**: Pie chart of improved/degraded/unchanged
- **Loss scatter**: Original vs ablated loss per example

## Tips

1. **For incorrect predictions**: Focus on examples where ablation changes the outcome
2. **Loss interpretation**: Lower loss after ablation = features were hurting
3. **Feature importance**: Top features by mean activation on incorrect examples
4. **Grid differences**: White X markers show where prediction differs from target
