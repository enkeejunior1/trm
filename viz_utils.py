"""
Visualization utilities for ablation analysis.

This module provides reusable visualization functions that can be imported
into ablation_eval.py or other analysis scripts.

Usage in ablation_eval.py:
    from analysis-yong.viz_utils import (
        decode_arc_grid,
        visualize_prediction_comparison,
        visualize_batch_results,
        create_summary_figure
    )
    
    # After getting predictions:
    visualize_prediction_comparison(
        test_input, test_label, original_pred, ablated_pred,
        demo_examples, puzzle_name, output_path,
        original_loss, ablated_loss
    )
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from typing import Dict, List, Tuple, Optional, Union
import json

# ARC color palette (standard 10 colors used in ARC-AGI)
ARC_COLORS = [
    '#000000',  # 0: black
    '#0074D9',  # 1: blue
    '#FF4136',  # 2: red
    '#2ECC40',  # 3: green
    '#FFDC00',  # 4: yellow
    '#AAAAAA',  # 5: grey
    '#F012BE',  # 6: magenta/fuchsia
    '#FF851B',  # 7: orange
    '#7FDBFF',  # 8: light blue/sky
    '#870C25',  # 9: maroon/dark red
]


def decode_arc_grid(tokens: np.ndarray, debug: bool = False) -> np.ndarray:
    """
    Decode tokenized ARC grid back to 2D grid.
    
    ARC tokenization:
    - PAD: 0
    - EOS: 1  
    - Colors 0-9: tokens 2-11
    
    Args:
        tokens: [900] or [30, 30] array of token IDs
        debug: if True, print debugging info
    
    Returns:
        grid: [H, W] numpy array of color IDs (0-9), cropped to actual content
    """
    if debug:
        unique_values = np.unique(tokens)
        print(f"  Token shape: {tokens.shape}, unique values: {unique_values}")
    
    # Handle different input shapes
    if tokens.ndim == 1:
        if len(tokens) != 900:
            if len(tokens) > 900:
                tokens = tokens[:900]
            else:
                tokens = np.pad(tokens, (0, 900 - len(tokens)), constant_values=0)
        grid_30x30 = tokens.reshape(30, 30).astype(np.int32)
    else:
        grid_30x30 = tokens.astype(np.int32)
    
    # Find maximum valid rectangle (tokens 2-11 are valid colors)
    max_area = 0
    max_size = (0, 0)
    num_c = 30
    
    for num_r in range(1, 31):
        for c in range(1, num_c + 1):
            x = grid_30x30[num_r - 1, c - 1]
            if (x < 2) or (x > 11):
                num_c = c - 1
                break
        
        area = num_r * num_c
        if area > max_area:
            max_area = area
            max_size = (num_r, num_c)
    
    if max_size[0] == 0 or max_size[1] == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    
    cropped = grid_30x30[:max_size[0], :max_size[1]]
    cropped = (cropped - 2).astype(np.uint8)
    
    return cropped


def visualize_grid(
    grid: np.ndarray, 
    ax: plt.Axes, 
    title: str = "", 
    highlight_color: str = None,
    show_diff: np.ndarray = None,
    fontsize: int = 10
):
    """
    Visualize an ARC grid.
    
    Args:
        grid: [H, W] numpy array of color IDs (0-9)
        ax: matplotlib axis
        title: title for the subplot
        highlight_color: border color for highlighting
        show_diff: boolean mask of cells to mark as different
        fontsize: font size for title
    """
    H, W = grid.shape
    cmap = ListedColormap(ARC_COLORS)
    
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=9, aspect='equal')
    
    # Grid lines
    for i in range(H + 1):
        ax.axhline(i - 0.5, color='white', linewidth=0.5, alpha=0.3)
    for j in range(W + 1):
        ax.axvline(j - 0.5, color='white', linewidth=0.5, alpha=0.3)
    
    # Difference markers
    if show_diff is not None and show_diff.shape == grid.shape:
        for i in range(H):
            for j in range(W):
                if show_diff[i, j]:
                    ax.plot(j, i, 'x', color='white', markersize=6, markeredgewidth=2)
    
    ax.set_xticks([])
    ax.set_yticks([])
    
    if highlight_color:
        for spine in ax.spines.values():
            spine.set_edgecolor(highlight_color)
            spine.set_linewidth(3)
    
    ax.set_title(title, fontsize=fontsize, fontweight='bold')
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)


def compute_diff(grid1: np.ndarray, grid2: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Compute difference between two grids.
    
    Returns:
        diff_mask: boolean array of differing cells
        num_diff: count of differing cells
    """
    h1, w1 = grid1.shape
    h2, w2 = grid2.shape
    
    # Pad to same size
    max_h, max_w = max(h1, h2), max(w1, w2)
    p1 = np.full((max_h, max_w), -1, dtype=np.int32)
    p2 = np.full((max_h, max_w), -1, dtype=np.int32)
    p1[:h1, :w1] = grid1
    p2[:h2, :w2] = grid2
    
    diff = p1 != p2
    return diff[:h1, :w1], int(np.sum(diff))


def check_prediction_correct(pred: np.ndarray, label: np.ndarray) -> bool:
    """Check if prediction matches label."""
    pred_grid = decode_arc_grid(pred)
    label_grid = decode_arc_grid(label)
    return np.array_equal(pred_grid, label_grid)


def visualize_prediction_comparison(
    test_input: np.ndarray,
    test_label: np.ndarray,
    original_pred: np.ndarray,
    ablated_pred: np.ndarray,
    demo_examples: List[Dict],
    puzzle_name: str,
    output_path: str,
    original_loss: float = None,
    ablated_loss: float = None,
    ablated_features: List[int] = None,
    puzzle_id: int = 0,
):
    """
    Create a comprehensive visualization comparing original and ablated predictions.
    
    Args:
        test_input: tokenized test input [900]
        test_label: tokenized test label [900]
        original_pred: original model prediction [900]
        ablated_pred: prediction after ablation [900]
        demo_examples: list of demo dicts with 'input' and 'output' keys
        puzzle_name: name of the puzzle
        output_path: path to save the figure
        original_loss: loss before ablation
        ablated_loss: loss after ablation
        ablated_features: list of ablated feature indices
        puzzle_id: puzzle identifier number
    
    Returns:
        dict with comparison results
    """
    num_demo = min(len(demo_examples), 4)  # Limit demos
    total_rows = num_demo + 3  # demos + target + original + ablated
    
    fig = plt.figure(figsize=(14, total_rows * 2.8))
    gs = gridspec.GridSpec(total_rows, 3, width_ratios=[1, 1, 1.2])
    
    # Demo examples
    for row, demo in enumerate(demo_examples[:num_demo]):
        ax_in = fig.add_subplot(gs[row, 0])
        ax_out = fig.add_subplot(gs[row, 1])
        ax_info = fig.add_subplot(gs[row, 2])
        
        in_grid = np.array(demo['input'], dtype=np.uint8)
        out_grid = np.array(demo['output'], dtype=np.uint8)
        
        visualize_grid(in_grid, ax_in, f"Demo {row+1} - Input")
        visualize_grid(out_grid, ax_out, f"Demo {row+1} - Output")
        ax_info.axis('off')
    
    # Decode all grids
    input_grid = decode_arc_grid(test_input)
    label_grid = decode_arc_grid(test_label)
    orig_grid = decode_arc_grid(original_pred)
    abl_grid = decode_arc_grid(ablated_pred)
    
    # Target row
    target_row = num_demo
    ax_t_in = fig.add_subplot(gs[target_row, 0])
    ax_t_out = fig.add_subplot(gs[target_row, 1])
    ax_t_info = fig.add_subplot(gs[target_row, 2])
    
    visualize_grid(input_grid, ax_t_in, "Test - Input")
    visualize_grid(label_grid, ax_t_out, "Test - Target (Ground Truth)", highlight_color='#0074D9')
    ax_t_info.axis('off')
    ax_t_info.text(0.5, 0.5, f"Target Grid\n{label_grid.shape[0]}×{label_grid.shape[1]}", 
                  ha='center', va='center', fontsize=11, transform=ax_t_info.transAxes)
    
    # Original prediction row
    orig_row = num_demo + 1
    ax_o_in = fig.add_subplot(gs[orig_row, 0])
    ax_o_out = fig.add_subplot(gs[orig_row, 1])
    ax_o_info = fig.add_subplot(gs[orig_row, 2])
    
    orig_correct = np.array_equal(label_grid, orig_grid)
    orig_status = "✓ CORRECT" if orig_correct else "✗ WRONG"
    orig_color = '#2ECC40' if orig_correct else '#FF4136'
    
    orig_diff, orig_ndiff = compute_diff(label_grid, orig_grid)
    
    visualize_grid(input_grid, ax_o_in, "Original - Input")
    visualize_grid(orig_grid, ax_o_out, f"Original ({orig_status})", 
                  highlight_color=orig_color, show_diff=orig_diff if not orig_correct else None)
    
    ax_o_info.axis('off')
    info_txt = f"Original Prediction\n{orig_status}"
    if original_loss is not None:
        info_txt += f"\nLoss: {original_loss:.4f}"
    if not orig_correct:
        info_txt += f"\n{orig_ndiff} cell(s) wrong"
    ax_o_info.text(0.5, 0.5, info_txt, ha='center', va='center', 
                  fontsize=11, color=orig_color, fontweight='bold',
                  transform=ax_o_info.transAxes)
    
    # Ablated prediction row
    abl_row = num_demo + 2
    ax_a_in = fig.add_subplot(gs[abl_row, 0])
    ax_a_out = fig.add_subplot(gs[abl_row, 1])
    ax_a_info = fig.add_subplot(gs[abl_row, 2])
    
    abl_correct = np.array_equal(label_grid, abl_grid)
    abl_status = "✓ CORRECT" if abl_correct else "✗ WRONG"
    abl_color = '#2ECC40' if abl_correct else '#FF4136'
    
    abl_diff, abl_ndiff = compute_diff(label_grid, abl_grid)
    
    visualize_grid(input_grid, ax_a_in, "Ablated - Input")
    visualize_grid(abl_grid, ax_a_out, f"Ablated ({abl_status})", 
                  highlight_color=abl_color, show_diff=abl_diff if not abl_correct else None)
    
    ax_a_info.axis('off')
    info_txt = f"After Ablation\n{abl_status}"
    if ablated_loss is not None:
        loss_delta = ablated_loss - (original_loss or 0)
        info_txt += f"\nLoss: {ablated_loss:.4f} ({'+' if loss_delta >= 0 else ''}{loss_delta:.4f})"
    if not abl_correct:
        info_txt += f"\n{abl_ndiff} cell(s) wrong"
    if ablated_features:
        info_txt += f"\n\n{len(ablated_features)} features ablated"
    ax_a_info.text(0.5, 0.5, info_txt, ha='center', va='center', 
                  fontsize=11, color=abl_color, fontweight='bold',
                  transform=ax_a_info.transAxes)
    
    # Title
    change_str = ""
    if orig_correct and not abl_correct:
        change_str = " [DEGRADED]"
    elif not orig_correct and abl_correct:
        change_str = " [IMPROVED!]"
    
    title = f"Ablation: {puzzle_name} (ID: {puzzle_id}){change_str}"
    if original_loss is not None and ablated_loss is not None:
        title += f"\nLoss: {original_loss:.4f} → {ablated_loss:.4f}"
    
    plt.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return {
        'puzzle_name': puzzle_name,
        'puzzle_id': puzzle_id,
        'original_correct': orig_correct,
        'ablated_correct': abl_correct,
        'original_loss': original_loss,
        'ablated_loss': ablated_loss,
        'original_diff_count': orig_ndiff,
        'ablated_diff_count': abl_ndiff,
        'change': 'improved' if not orig_correct and abl_correct else 
                  'degraded' if orig_correct and not abl_correct else 'unchanged'
    }


def visualize_batch_results(
    inputs: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    puzzle_ids: np.ndarray,
    identifiers_map: List[str],
    test_puzzles: Dict,
    output_dir: str,
    batch_idx: int = 0,
    losses: np.ndarray = None,
):
    """
    Visualize all predictions in a batch.
    
    Args:
        inputs: [B, 900] tokenized inputs
        labels: [B, 900] tokenized labels
        predictions: [B, 900] model predictions
        puzzle_ids: [B] puzzle identifier indices
        identifiers_map: list mapping index to puzzle name
        test_puzzles: dict of puzzle_name -> {train: [...], test: [...]}
        output_dir: directory to save visualizations
        batch_idx: batch number for naming
        losses: optional [B] array of losses per example
    
    Returns:
        list of result dicts
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []
    
    B = len(inputs)
    print(f"\nVisualizing batch {batch_idx}: {B} examples")
    
    for i in range(B):
        # Get puzzle name
        pid = int(puzzle_ids[i]) if puzzle_ids is not None else i
        if pid < len(identifiers_map):
            name = identifiers_map[pid]
        else:
            name = f"puzzle_{pid}"
        
        base_name = name.split("|||")[0] if "|||" in name else name
        
        # Get demos
        demos = test_puzzles.get(base_name, {}).get("train", [])
        
        # Decode and check
        pred_grid = decode_arc_grid(predictions[i])
        label_grid = decode_arc_grid(labels[i])
        input_grid = decode_arc_grid(inputs[i])
        
        is_correct = np.array_equal(pred_grid, label_grid)
        diff, ndiff = compute_diff(label_grid, pred_grid)
        
        # Create simple visualization
        num_demo = min(len(demos), 3)
        total_rows = num_demo + 2
        
        fig, axes = plt.subplots(total_rows, 2, figsize=(8, total_rows * 2.5))
        if total_rows == 1:
            axes = axes[np.newaxis, :]
        
        # Demos
        for row, demo in enumerate(demos[:num_demo]):
            in_g = np.array(demo['input'], dtype=np.uint8)
            out_g = np.array(demo['output'], dtype=np.uint8)
            visualize_grid(in_g, axes[row, 0], f"Demo {row+1} - In")
            visualize_grid(out_g, axes[row, 1], f"Demo {row+1} - Out")
        
        # Target
        visualize_grid(input_grid, axes[num_demo, 0], "Test - Input")
        visualize_grid(label_grid, axes[num_demo, 1], "Test - Target", highlight_color='#0074D9')
        
        # Prediction
        status = "✓" if is_correct else f"✗ ({ndiff} diff)"
        color = '#2ECC40' if is_correct else '#FF4136'
        visualize_grid(input_grid, axes[num_demo+1, 0], "Pred - Input")
        visualize_grid(pred_grid, axes[num_demo+1, 1], f"Pred - {status}", 
                      highlight_color=color, show_diff=diff if not is_correct else None)
        
        title = f"{base_name} (ID: {pid})"
        if losses is not None:
            title += f" | Loss: {losses[i]:.4f}"
        plt.suptitle(title, fontsize=11, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        save_path = os.path.join(output_dir, f"batch{batch_idx:03d}_{i:03d}_{base_name}.png")
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close()
        
        results.append({
            'puzzle_name': base_name,
            'is_correct': is_correct,
            'num_diff': ndiff,
            'loss': float(losses[i]) if losses is not None else None
        })
        
        print(f"  [{i+1}/{B}] {base_name}: {'✓' if is_correct else f'✗ ({ndiff} diff)'}")
    
    return results


def create_summary_figure(
    results: List[Dict],
    output_path: str,
    title: str = "Ablation Summary",
    topk: int = 20
):
    """
    Create a summary figure for ablation results.
    
    Args:
        results: list of dicts with 'original_correct', 'ablated_correct', 
                 'original_loss', 'ablated_loss' keys
        output_path: path to save the figure
        title: figure title
        topk: number of features that were ablated
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Stats
    n = len(results)
    orig_correct = sum(1 for r in results if r.get('original_correct', False))
    abl_correct = sum(1 for r in results if r.get('ablated_correct', False))
    
    # 1. Accuracy comparison
    ax = axes[0, 0]
    x = ['Original', 'Ablated']
    correct = [orig_correct, abl_correct]
    incorrect = [n - orig_correct, n - abl_correct]
    
    ax.bar(x, correct, color='#2ECC40', alpha=0.8, label='Correct')
    ax.bar(x, incorrect, bottom=correct, color='#FF4136', alpha=0.8, label='Incorrect')
    ax.set_ylabel('Count')
    ax.set_title(f'Accuracy (Top-{topk} Ablation)')
    ax.legend()
    
    for i, (c, ic) in enumerate(zip(correct, incorrect)):
        if c > 0:
            ax.text(i, c/2, str(c), ha='center', va='center', fontweight='bold', color='white')
        if ic > 0:
            ax.text(i, c + ic/2, str(ic), ha='center', va='center', fontweight='bold', color='white')
    
    # 2. Loss change histogram
    ax = axes[0, 1]
    deltas = [r.get('ablated_loss', 0) - r.get('original_loss', 0) 
              for r in results if r.get('original_loss') is not None]
    if deltas:
        ax.hist(deltas, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', label='No change')
        ax.axvline(np.mean(deltas), color='orange', label=f'Mean: {np.mean(deltas):+.4f}')
        ax.set_xlabel('Loss Change')
        ax.set_ylabel('Count')
        ax.set_title('Loss Change Distribution')
        ax.legend()
    
    # 3. Change breakdown pie
    ax = axes[1, 0]
    stayed_correct = sum(1 for r in results if r.get('original_correct') and r.get('ablated_correct'))
    became_correct = sum(1 for r in results if not r.get('original_correct') and r.get('ablated_correct'))
    stayed_wrong = sum(1 for r in results if not r.get('original_correct') and not r.get('ablated_correct'))
    became_wrong = sum(1 for r in results if r.get('original_correct') and not r.get('ablated_correct'))
    
    labels = ['Stayed Correct', 'Improved', 'Stayed Wrong', 'Degraded']
    sizes = [stayed_correct, became_correct, stayed_wrong, became_wrong]
    colors = ['#2ECC40', '#7FDBFF', '#FFDC00', '#FF4136']
    
    non_zero = [(s, l, c) for s, l, c in zip(sizes, labels, colors) if s > 0]
    if non_zero:
        sizes_nz, labels_nz, colors_nz = zip(*non_zero)
        ax.pie(sizes_nz, labels=labels_nz, colors=colors_nz,
               autopct=lambda p: f'{int(p*n/100)}', startangle=90)
    ax.set_title('Prediction Changes')
    
    # 4. Loss scatter
    ax = axes[1, 1]
    orig_losses = [r.get('original_loss') for r in results if r.get('original_loss') is not None]
    abl_losses = [r.get('ablated_loss') for r in results if r.get('ablated_loss') is not None]
    colors_scatter = ['#2ECC40' if r.get('original_correct') else '#FF4136' 
                     for r in results if r.get('original_loss') is not None]
    
    if orig_losses:
        ax.scatter(orig_losses, abl_losses, c=colors_scatter, alpha=0.6, s=40, edgecolors='k', linewidths=0.5)
        lims = [min(orig_losses + abl_losses), max(orig_losses + abl_losses)]
        ax.plot(lims, lims, 'k--', alpha=0.5)
        ax.set_xlabel('Original Loss')
        ax.set_ylabel('Ablated Loss')
        ax.set_title('Loss Comparison')
    
    # Title
    acc_orig = orig_correct / n * 100 if n > 0 else 0
    acc_abl = abl_correct / n * 100 if n > 0 else 0
    plt.suptitle(f'{title}\nAccuracy: {orig_correct}/{n} ({acc_orig:.1f}%) → {abl_correct}/{n} ({acc_abl:.1f}%)',
                fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_results_json(results: List[Dict], config: Dict, output_path: str):
    """Save results to JSON file."""
    n = len(results)
    orig_correct = sum(1 for r in results if r.get('original_correct', False))
    abl_correct = sum(1 for r in results if r.get('ablated_correct', False))
    
    orig_losses = [r['original_loss'] for r in results if r.get('original_loss') is not None]
    abl_losses = [r['ablated_loss'] for r in results if r.get('ablated_loss') is not None]
    
    output = {
        'config': config,
        'summary': {
            'total_examples': n,
            'original_accuracy': orig_correct / n if n > 0 else 0,
            'ablated_accuracy': abl_correct / n if n > 0 else 0,
            'avg_original_loss': float(np.mean(orig_losses)) if orig_losses else None,
            'avg_ablated_loss': float(np.mean(abl_losses)) if abl_losses else None,
            'improved_count': sum(1 for r in results if r.get('change') == 'improved'),
            'degraded_count': sum(1 for r in results if r.get('change') == 'degraded'),
        },
        'results': results
    }
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Results saved to: {output_path}")


# ============================================================================
# Quick demo/test
# ============================================================================
if __name__ == "__main__":
    # Quick test with synthetic data
    print("Testing viz_utils...")
    
    # Create test grid (3x3 with some colors)
    test_tokens = np.zeros(900, dtype=np.int32)
    # Fill top-left 3x3 with colors (tokens 2-11 represent colors 0-9)
    for i in range(3):
        for j in range(3):
            test_tokens[i * 30 + j] = 2 + (i * 3 + j) % 10  # colors 0-9
    
    grid = decode_arc_grid(test_tokens, debug=True)
    print(f"Decoded grid shape: {grid.shape}")
    print(f"Grid:\n{grid}")
    
    # Test visualization
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    visualize_grid(grid, ax, "Test Grid")
    plt.savefig("/tmp/viz_test.png", dpi=100)
    plt.close()
    print("Test visualization saved to /tmp/viz_test.png")
