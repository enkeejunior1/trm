"""
Visualization script for Ablation Analysis predictions.

This script visualizes predictions from ablation_eval.py or batch_data files:
- Demo examples (input → output)
- Test example (input → target ground truth)
- Original prediction (before ablation)
- Ablated prediction (after ablating top-k features)
- Comparison metrics (loss change, correctness)

Can work with:
1. Pre-computed batch_data files from ablation_eval.py
2. Pre-computed ablation_results.json
3. Live inference with model + SAE

Usage examples:
    # Visualize from ablation results JSON
    python analysis-yong/visualize_ablation_predictions.py \
        --results_json results-analysis-noaug/ablation_visualizations/ablation_results.json \
        --data_path data/arc1concept-aug-0 \
        --output_dir results-analysis-noaug/ablation_viz

    # Visualize from batch_data files
    python analysis-yong/visualize_ablation_predictions.py \
        --batch_data_dir results/batch_data/ \
        --data_path data/arc1concept-aug-0 \
        --output_dir results-analysis-noaug/ablation_viz

    # Run live inference visualization
    python analysis-yong/visualize_ablation_predictions.py \
        --data_path data/arc1concept-aug-0 \
        --model_data_path data/arc1concept-aug-1000 \
        --checkpoint ckpt/arc_v1_public/step_518071 \
        --config_path ckpt/arc_v1_public \
        --sae_checkpoint weights/sae/best_val.pt \
        --output_dir results-analysis-noaug/ablation_viz \
        --num_examples 10 \
        --topk_ablate 20
"""
import os
import sys
import json
import argparse
import ast
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

import torch
from torch import nn
from torch.nn import functional as F

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from puzzle_dataset import PuzzleDatasetConfig, PuzzleDataset
    from dataset.common import PuzzleDatasetMetadata
    from models.losses import IGNORE_LABEL_ID
    from utils.functions import load_model_class
except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")
    print("Some features may not be available.")

# Global dtype for SAE
DTYPE = torch.bfloat16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


# ============================================================================
# Grid Decoding and Visualization Utilities
# ============================================================================
def decode_arc_grid(tokens: np.ndarray, debug: bool = False) -> np.ndarray:
    """
    Decode tokenized ARC grid back to 2D grid.
    
    ARC tokenization:
    - PAD: 0
    - EOS: 1  
    - Colors 0-9: tokens 2-11
    
    Grid is 30x30, with EOS tokens marking the boundary of actual content.
    """
    if debug:
        unique_values = np.unique(tokens)
        print(f"  Token shape: {tokens.shape}, unique values: {unique_values}")
    
    # Ensure tokens are the right shape
    if tokens.ndim == 1:
        if len(tokens) != 900:
            # Try to handle different formats
            if len(tokens) > 900:
                tokens = tokens[:900]
            else:
                tokens = np.pad(tokens, (0, 900 - len(tokens)), constant_values=0)
        grid_30x30 = tokens.reshape(30, 30).astype(np.int32)
    else:
        grid_30x30 = tokens.astype(np.int32)
    
    # Find maximum-sized rectangle without any EOS token (value < 2 or value > 11) inside
    max_area = 0
    max_size = (0, 0)
    nr, nc = 30, 30
    
    num_c = nc
    for num_r in range(1, nr + 1):
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


def visualize_arc_grid(grid: np.ndarray, ax: plt.Axes, title: str = "", 
                       highlight_color: str = None, show_diff: np.ndarray = None):
    """
    Visualize an ARC grid with optional highlighting and difference overlay.
    
    Args:
        grid: [H, W] numpy array of color IDs (0-9)
        ax: matplotlib axis to plot on
        title: title for the plot
        highlight_color: optional color for border highlighting
        show_diff: optional boolean array marking cells that differ from reference
    """
    H, W = grid.shape
    cmap = ListedColormap(ARC_COLORS)
    
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=9, aspect='equal')
    
    # Add grid lines
    for i in range(H + 1):
        ax.axhline(i - 0.5, color='white', linewidth=0.5, alpha=0.3)
    for j in range(W + 1):
        ax.axvline(j - 0.5, color='white', linewidth=0.5, alpha=0.3)
    
    # Overlay difference markers if provided
    if show_diff is not None and show_diff.shape == grid.shape:
        for i in range(H):
            for j in range(W):
                if show_diff[i, j]:
                    # Draw X marker for differences
                    ax.plot(j, i, 'x', color='white', markersize=8, markeredgewidth=2)
    
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Highlight border if specified
    if highlight_color:
        for spine in ax.spines.values():
            spine.set_edgecolor(highlight_color)
            spine.set_linewidth(3)
    
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)


def compute_grid_diff(grid1: np.ndarray, grid2: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """
    Compute difference between two grids.
    
    Returns:
        diff_mask: boolean array where True means cells differ
        num_diff: number of differing cells
        total_cells: total number of cells in comparison
    """
    # Pad to same size if needed
    h1, w1 = grid1.shape
    h2, w2 = grid2.shape
    max_h, max_w = max(h1, h2), max(w1, w2)
    
    padded1 = np.full((max_h, max_w), -1, dtype=np.int32)
    padded2 = np.full((max_h, max_w), -1, dtype=np.int32)
    padded1[:h1, :w1] = grid1
    padded2[:h2, :w2] = grid2
    
    diff_mask = padded1 != padded2
    num_diff = np.sum(diff_mask)
    total_cells = max_h * max_w
    
    return diff_mask[:h1, :w1], int(num_diff), total_cells


# ============================================================================
# Data Loading Utilities
# ============================================================================
def load_puzzle_metadata(data_path: str) -> Tuple[List[str], Dict]:
    """Load puzzle identifiers and test puzzles metadata."""
    identifiers_path = os.path.join(data_path, "identifiers.json")
    test_puzzles_path = os.path.join(data_path, "test_puzzles.json")
    
    with open(identifiers_path, 'r') as f:
        identifiers_map = json.load(f)
    
    with open(test_puzzles_path, 'r') as f:
        test_puzzles = json.load(f)
    
    return identifiers_map, test_puzzles


def load_test_data(data_path: str) -> Dict[str, np.ndarray]:
    """Load test dataset arrays."""
    test_data_dir = os.path.join(data_path, "test")
    
    return {
        'inputs': np.load(os.path.join(test_data_dir, "all__inputs.npy")),
        'labels': np.load(os.path.join(test_data_dir, "all__labels.npy")),
        'puzzle_ids': np.load(os.path.join(test_data_dir, "all__puzzle_identifiers.npy")),
        'puzzle_indices': np.load(os.path.join(test_data_dir, "all__puzzle_indices.npy")),
    }


def load_batch_data(batch_data_path: str) -> Dict:
    """Load a batch_data file from ablation_eval.py."""
    data = torch.load(batch_data_path, map_location='cpu', weights_only=False)
    return data


def load_ablation_results(results_json_path: str) -> Dict:
    """Load ablation results JSON file."""
    with open(results_json_path, 'r') as f:
        return json.load(f)


# ============================================================================
# Visualization Functions
# ============================================================================
def visualize_single_prediction(
    puzzle_name: str,
    puzzle_id: int,
    demo_examples: List[Dict],
    test_input: np.ndarray,
    test_label: np.ndarray,
    prediction: np.ndarray,
    output_dir: str,
    example_idx: int,
    loss: float = None,
    metrics: Dict = None,
):
    """
    Visualize a single puzzle with prediction.
    
    Layout:
    - Demo rows: Input | Output
    - Test row: Input | Target (Ground Truth)
    - Pred row: Input | Prediction (with correctness status)
    """
    num_demo = len(demo_examples)
    total_rows = num_demo + 2
    
    fig = plt.figure(figsize=(10, total_rows * 3))
    gs = gridspec.GridSpec(total_rows, 2, width_ratios=[1, 1])
    
    # Plot demo examples
    for ex_row, demo in enumerate(demo_examples):
        ax_in = fig.add_subplot(gs[ex_row, 0])
        ax_out = fig.add_subplot(gs[ex_row, 1])
        
        input_grid = np.array(demo['input'], dtype=np.uint8)
        output_grid = np.array(demo['output'], dtype=np.uint8)
        
        visualize_arc_grid(input_grid, ax_in, f"Demo {ex_row+1} - Input")
        visualize_arc_grid(output_grid, ax_out, f"Demo {ex_row+1} - Output")
    
    # Decode grids
    test_input_grid = decode_arc_grid(test_input)
    test_label_grid = decode_arc_grid(test_label)
    pred_grid = decode_arc_grid(prediction)
    
    # Test row
    test_row = num_demo
    ax_test_in = fig.add_subplot(gs[test_row, 0])
    ax_test_out = fig.add_subplot(gs[test_row, 1])
    
    visualize_arc_grid(test_input_grid, ax_test_in, "Test - Input")
    visualize_arc_grid(test_label_grid, ax_test_out, "Test - Target", highlight_color='blue')
    
    # Prediction row
    pred_row = num_demo + 1
    ax_pred_in = fig.add_subplot(gs[pred_row, 0])
    ax_pred_out = fig.add_subplot(gs[pred_row, 1])
    
    is_correct = np.array_equal(test_label_grid, pred_grid)
    status = "✓ CORRECT" if is_correct else "✗ INCORRECT"
    status_color = 'green' if is_correct else 'red'
    
    # Compute diff for visualization
    diff_mask, num_diff, _ = compute_grid_diff(test_label_grid, pred_grid)
    
    visualize_arc_grid(test_input_grid, ax_pred_in, "Model - Input")
    visualize_arc_grid(pred_grid, ax_pred_out, f"Prediction ({status})",
                       highlight_color=status_color, 
                       show_diff=diff_mask if not is_correct else None)
    
    # Title with optional loss info
    title_str = f"Puzzle: {puzzle_name} (ID: {puzzle_id})"
    if loss is not None:
        title_str += f"\nLoss: {loss:.4f}"
    if not is_correct:
        title_str += f" | {num_diff} cell(s) different"
    
    plt.suptitle(title_str, fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    save_path = os.path.join(output_dir, f"pred_{example_idx:03d}_{puzzle_name}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return is_correct, num_diff


def visualize_ablation_comparison(
    puzzle_name: str,
    puzzle_id: int,
    demo_examples: List[Dict],
    test_input: np.ndarray,
    test_label: np.ndarray,
    original_pred: np.ndarray,
    ablated_pred: np.ndarray,
    original_loss: float,
    ablated_loss: float,
    ablated_features: List[int],
    output_dir: str,
    example_idx: int,
):
    """
    Visualize comparison between original and ablated predictions.
    
    Layout:
    - Demo rows: Input | Output | (empty)
    - Test row: Input | Target | Info
    - Original row: Input | Prediction | Info
    - Ablated row: Input | Prediction | Info
    """
    num_demo = len(demo_examples)
    total_rows = num_demo + 3
    
    fig = plt.figure(figsize=(14, total_rows * 3))
    gs = gridspec.GridSpec(total_rows, 3, width_ratios=[1, 1, 1.2])
    
    # Plot demo examples
    for ex_row, demo in enumerate(demo_examples):
        ax_in = fig.add_subplot(gs[ex_row, 0])
        ax_out = fig.add_subplot(gs[ex_row, 1])
        ax_empty = fig.add_subplot(gs[ex_row, 2])
        
        input_grid = np.array(demo['input'], dtype=np.uint8)
        output_grid = np.array(demo['output'], dtype=np.uint8)
        
        visualize_arc_grid(input_grid, ax_in, f"Demo {ex_row+1} - Input")
        visualize_arc_grid(output_grid, ax_out, f"Demo {ex_row+1} - Output")
        ax_empty.axis('off')
    
    # Decode grids
    test_input_grid = decode_arc_grid(test_input)
    test_label_grid = decode_arc_grid(test_label)
    original_pred_grid = decode_arc_grid(original_pred)
    ablated_pred_grid = decode_arc_grid(ablated_pred)
    
    # Test row
    test_row = num_demo
    ax_test_in = fig.add_subplot(gs[test_row, 0])
    ax_test_out = fig.add_subplot(gs[test_row, 1])
    ax_test_info = fig.add_subplot(gs[test_row, 2])
    
    visualize_arc_grid(test_input_grid, ax_test_in, "Test - Input")
    visualize_arc_grid(test_label_grid, ax_test_out, "Test - Target", highlight_color='blue')
    ax_test_info.axis('off')
    ax_test_info.text(0.5, 0.5, f"Ground Truth\nSize: {test_label_grid.shape[0]}×{test_label_grid.shape[1]}", 
                      ha='center', va='center', fontsize=11, transform=ax_test_info.transAxes)
    
    # Original prediction row
    orig_row = num_demo + 1
    ax_orig_in = fig.add_subplot(gs[orig_row, 0])
    ax_orig_out = fig.add_subplot(gs[orig_row, 1])
    ax_orig_info = fig.add_subplot(gs[orig_row, 2])
    
    is_orig_correct = np.array_equal(test_label_grid, original_pred_grid)
    orig_status = "✓ CORRECT" if is_orig_correct else "✗ INCORRECT"
    orig_color = 'green' if is_orig_correct else 'red'
    
    orig_diff, orig_num_diff, _ = compute_grid_diff(test_label_grid, original_pred_grid)
    
    visualize_arc_grid(test_input_grid, ax_orig_in, "Original - Input")
    visualize_arc_grid(original_pred_grid, ax_orig_out, f"Original Pred ({orig_status})",
                       highlight_color=orig_color,
                       show_diff=orig_diff if not is_orig_correct else None)
    ax_orig_info.axis('off')
    orig_info_text = f"Original Prediction\nLoss: {original_loss:.4f}\n{orig_status}"
    if not is_orig_correct:
        orig_info_text += f"\n{orig_num_diff} cell(s) wrong"
    ax_orig_info.text(0.5, 0.5, orig_info_text, ha='center', va='center', 
                      fontsize=11, color=orig_color, transform=ax_orig_info.transAxes)
    
    # Ablated prediction row
    abl_row = num_demo + 2
    ax_abl_in = fig.add_subplot(gs[abl_row, 0])
    ax_abl_out = fig.add_subplot(gs[abl_row, 1])
    ax_abl_info = fig.add_subplot(gs[abl_row, 2])
    
    is_abl_correct = np.array_equal(test_label_grid, ablated_pred_grid)
    abl_status = "✓ CORRECT" if is_abl_correct else "✗ INCORRECT"
    abl_color = 'green' if is_abl_correct else 'red'
    
    abl_diff, abl_num_diff, _ = compute_grid_diff(test_label_grid, ablated_pred_grid)
    
    visualize_arc_grid(test_input_grid, ax_abl_in, "Ablated - Input")
    visualize_arc_grid(ablated_pred_grid, ax_abl_out, f"Ablated Pred ({abl_status})",
                       highlight_color=abl_color,
                       show_diff=abl_diff if not is_abl_correct else None)
    
    ax_abl_info.axis('off')
    loss_change = ablated_loss - original_loss
    loss_change_str = f"+{loss_change:.4f}" if loss_change > 0 else f"{loss_change:.4f}"
    abl_info_text = (f"After Ablation\n"
                     f"Loss: {ablated_loss:.4f} ({loss_change_str})\n"
                     f"{abl_status}")
    if not is_abl_correct:
        abl_info_text += f"\n{abl_num_diff} cell(s) wrong"
    abl_info_text += f"\n\nAblated {len(ablated_features)} features"
    ax_abl_info.text(0.5, 0.5, abl_info_text, ha='center', va='center', 
                     fontsize=11, color=abl_color, transform=ax_abl_info.transAxes)
    
    # Title
    change_indicator = ""
    if is_orig_correct != is_abl_correct:
        if is_abl_correct:
            change_indicator = " [IMPROVED!]"
        else:
            change_indicator = " [DEGRADED!]"
    
    plt.suptitle(
        f"Ablation Analysis: {puzzle_name} (ID: {puzzle_id}){change_indicator}\n"
        f"Top-{len(ablated_features)} features ablated | Loss: {original_loss:.4f} → {ablated_loss:.4f}",
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    
    save_path = os.path.join(output_dir, f"ablation_{example_idx:03d}_{puzzle_name}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return is_orig_correct, is_abl_correct, orig_num_diff, abl_num_diff


def visualize_feature_importance(
    importance_scores: np.ndarray,
    top_k: int,
    output_path: str,
    title: str = "Feature Importance Analysis"
):
    """Visualize feature importance distribution and top-k features."""
    if isinstance(importance_scores, torch.Tensor):
        importance = importance_scores.cpu().float().numpy()
    else:
        importance = importance_scores
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Histogram of all importance scores
    axes[0].hist(importance, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    threshold = np.sort(importance)[-top_k] if top_k <= len(importance) else 0
    axes[0].axvline(threshold, color='red', linestyle='--', 
                   label=f'Top-{top_k} threshold')
    axes[0].set_xlabel('Importance Score')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Distribution of Feature Importance')
    axes[0].legend()
    
    # 2. Top-k features bar chart
    top_indices = np.argsort(importance)[-top_k:][::-1]
    top_values = importance[top_indices]
    
    y_pos = np.arange(len(top_indices))
    axes[1].barh(y_pos, top_values, color='steelblue')
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels([f'F{i}' for i in top_indices])
    axes[1].set_xlabel('Importance Score')
    axes[1].set_title(f'Top-{top_k} Most Important Features')
    axes[1].invert_yaxis()
    
    # 3. Cumulative importance
    sorted_importance = np.sort(importance)[::-1]
    cumulative = np.cumsum(sorted_importance) / np.sum(sorted_importance)
    
    axes[2].plot(range(len(cumulative)), cumulative, color='steelblue')
    axes[2].axvline(top_k, color='red', linestyle='--', label=f'Top-{top_k}')
    if top_k < len(cumulative):
        axes[2].axhline(cumulative[top_k-1], color='red', linestyle=':', alpha=0.5)
        axes[2].text(top_k + 5, cumulative[top_k-1], f'{cumulative[top_k-1]:.2%}', fontsize=10)
    axes[2].set_xlabel('Number of Features')
    axes[2].set_ylabel('Cumulative Importance')
    axes[2].set_title('Cumulative Feature Importance')
    axes[2].legend()
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_ablation_summary(
    results: List[Dict],
    output_path: str,
    topk_ablated: int
):
    """
    Visualize summary of ablation experiment.
    
    Shows:
    - Accuracy comparison (before/after)
    - Loss change distribution
    - Prediction change breakdown (pie chart)
    - Loss scatter plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Extract statistics
    orig_correct = sum(1 for r in results if r.get('original_correct', False))
    orig_incorrect = len(results) - orig_correct
    abl_correct = sum(1 for r in results if r.get('ablated_correct', False))
    abl_incorrect = len(results) - abl_correct
    
    # 1. Accuracy comparison bar chart
    ax = axes[0, 0]
    x = ['Original', 'After Ablation']
    correct_counts = [orig_correct, abl_correct]
    incorrect_counts = [orig_incorrect, abl_incorrect]
    
    bars1 = ax.bar(x, correct_counts, label='Correct', color='#2ECC40', alpha=0.8)
    bars2 = ax.bar(x, incorrect_counts, bottom=correct_counts, label='Incorrect', color='#FF4136', alpha=0.8)
    
    ax.set_ylabel('Number of Examples')
    ax.set_title(f'Prediction Accuracy (Top-{topk_ablated} Ablation)')
    ax.legend(loc='upper right')
    
    for bar, c, i in zip(bars1, correct_counts, incorrect_counts):
        if c > 0:
            ax.text(bar.get_x() + bar.get_width()/2, c/2, f'{c}', 
                   ha='center', va='center', fontsize=12, fontweight='bold', color='white')
        if i > 0:
            ax.text(bar.get_x() + bar.get_width()/2, c + i/2, f'{i}', 
                   ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    
    # 2. Loss change distribution
    ax = axes[0, 1]
    loss_changes = [r.get('ablated_loss', 0) - r.get('original_loss', 0) for r in results]
    
    ax.hist(loss_changes, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='No change')
    mean_change = np.mean(loss_changes)
    ax.axvline(mean_change, color='orange', linestyle='-', linewidth=2, 
               label=f'Mean: {mean_change:+.4f}')
    ax.set_xlabel('Loss Change (Ablated - Original)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Loss Changes')
    ax.legend()
    
    # 3. Prediction change breakdown (pie chart)
    ax = axes[1, 0]
    categories = ['Stayed Correct', 'Became Correct', 'Stayed Incorrect', 'Became Incorrect']
    counts = [
        sum(1 for r in results if r.get('original_correct', False) and r.get('ablated_correct', False)),
        sum(1 for r in results if not r.get('original_correct', False) and r.get('ablated_correct', False)),
        sum(1 for r in results if not r.get('original_correct', False) and not r.get('ablated_correct', False)),
        sum(1 for r in results if r.get('original_correct', False) and not r.get('ablated_correct', False)),
    ]
    colors = ['#2ECC40', '#7FDBFF', '#FFDC00', '#FF4136']
    
    # Filter out zero counts for cleaner pie chart
    non_zero = [(c, cat, col) for c, cat, col in zip(counts, categories, colors) if c > 0]
    if non_zero:
        counts_nz, categories_nz, colors_nz = zip(*non_zero)
        wedges, texts, autotexts = ax.pie(counts_nz, labels=categories_nz, colors=colors_nz, 
                                           autopct=lambda p: f'{p:.1f}%\n({int(p*len(results)/100)})',
                                           startangle=90, textprops={'fontsize': 9})
    ax.set_title('Prediction Change After Ablation')
    
    # 4. Loss scatter plot
    ax = axes[1, 1]
    orig_losses = [r.get('original_loss', 0) for r in results]
    abl_losses = [r.get('ablated_loss', 0) for r in results]
    colors_scatter = ['#2ECC40' if r.get('original_correct', False) else '#FF4136' for r in results]
    
    ax.scatter(orig_losses, abl_losses, c=colors_scatter, alpha=0.6, s=50, edgecolors='black', linewidths=0.5)
    
    # Diagonal line (no change)
    all_losses = orig_losses + abl_losses
    if all_losses:
        min_loss, max_loss = min(all_losses), max(all_losses)
        ax.plot([min_loss, max_loss], [min_loss, max_loss], 'k--', alpha=0.5, label='No change')
    
    ax.set_xlabel('Original Loss')
    ax.set_ylabel('Ablated Loss')
    ax.set_title('Loss Comparison')
    ax.legend()
    
    # Add legend for colors
    green_patch = mpatches.Patch(color='#2ECC40', label='Originally Correct')
    red_patch = mpatches.Patch(color='#FF4136', label='Originally Incorrect')
    ax.legend(handles=[green_patch, red_patch], loc='upper left')
    
    # Super title
    acc_orig = orig_correct / len(results) * 100 if results else 0
    acc_abl = abl_correct / len(results) * 100 if results else 0
    
    plt.suptitle(
        f'Ablation Summary: Top-{topk_ablated} Features\n'
        f'Original Accuracy: {orig_correct}/{len(results)} ({acc_orig:.1f}%) → '
        f'Ablated Accuracy: {abl_correct}/{len(results)} ({acc_abl:.1f}%)',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_batch_predictions(
    batch_data: Dict,
    identifiers_map: List[str],
    test_puzzles: Dict,
    output_dir: str,
    batch_idx: int = 0,
):
    """
    Visualize predictions from a batch_data file.
    
    batch_data structure:
    - batch_index: int
    - loss: tensor
    - trajectories_L: [num_steps, B, L, H]
    - trajectories_H: [num_steps, B, L, H]
    - metrics: dict
    - predictions: dict with 'preds' key
    - batch_info: dict with 'inputs', 'labels', 'puzzle_identifiers'
    """
    results = []
    
    # Extract data
    predictions = batch_data.get('predictions', {})
    batch_info = batch_data.get('batch_info', {})
    metrics = batch_data.get('metrics', {})
    loss = batch_data.get('loss', None)
    
    if 'preds' in predictions:
        preds = predictions['preds'].numpy()
    elif isinstance(predictions, torch.Tensor):
        preds = predictions.numpy()
    else:
        print("Warning: No predictions found in batch_data")
        return results
    
    inputs = batch_info.get('inputs', None)
    labels = batch_info.get('labels', None)
    puzzle_ids = batch_info.get('puzzle_identifiers', None)
    
    if inputs is None or labels is None:
        print("Warning: Missing inputs or labels in batch_info")
        return results
    
    inputs = inputs.numpy() if isinstance(inputs, torch.Tensor) else inputs
    labels = labels.numpy() if isinstance(labels, torch.Tensor) else labels
    puzzle_ids = puzzle_ids.numpy() if isinstance(puzzle_ids, torch.Tensor) else puzzle_ids
    
    batch_size = len(inputs)
    print(f"\nVisualizing batch {batch_idx} with {batch_size} examples...")
    
    for i in range(batch_size):
        # Get puzzle name
        if puzzle_ids is not None and i < len(puzzle_ids):
            pid = int(puzzle_ids[i])
            if pid < len(identifiers_map):
                puzzle_name = identifiers_map[pid]
            else:
                puzzle_name = f"puzzle_{pid}"
        else:
            puzzle_name = f"example_{i}"
        
        # Skip augmented puzzles for cleaner visualization
        base_puzzle_name = puzzle_name.split("|||")[0] if "|||" in puzzle_name else puzzle_name
        
        # Get demo examples
        demo_examples = test_puzzles.get(base_puzzle_name, {}).get("train", [])
        
        # Visualize
        is_correct, num_diff = visualize_single_prediction(
            puzzle_name=base_puzzle_name,
            puzzle_id=pid if puzzle_ids is not None else i,
            demo_examples=demo_examples[:3],  # Limit demos for space
            test_input=inputs[i],
            test_label=labels[i],
            prediction=preds[i],
            output_dir=output_dir,
            example_idx=batch_idx * 1000 + i,
            loss=float(loss) if loss is not None else None,
        )
        
        results.append({
            'puzzle_name': base_puzzle_name,
            'is_correct': is_correct,
            'num_diff': num_diff,
        })
        
        status = "✓" if is_correct else f"✗ ({num_diff} diff)"
        print(f"  [{i+1}/{batch_size}] {base_puzzle_name}: {status}")
    
    return results


# ============================================================================
# Main Functions
# ============================================================================
def visualize_from_results_json(args):
    """Visualize ablation results from pre-computed JSON file."""
    print("="*70)
    print("Visualizing Ablation Results from JSON")
    print("="*70)
    
    # Load results
    results_data = load_ablation_results(args.results_json)
    print(f"Loaded {len(results_data['results'])} results")
    print(f"Config: {results_data['config']}")
    
    # Load puzzle metadata
    identifiers_map, test_puzzles = load_puzzle_metadata(args.data_path)
    test_data = load_test_data(args.data_path)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create summary visualization
    summary_path = os.path.join(args.output_dir, "ablation_summary.png")
    visualize_ablation_summary(
        results_data['results'],
        summary_path,
        results_data['config']['topk_ablate']
    )
    print(f"Saved summary: {summary_path}")
    
    # Print summary statistics
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total examples: {len(results_data['results'])}")
    print(f"Original accuracy: {results_data['summary']['original_accuracy']*100:.1f}%")
    print(f"Ablated accuracy: {results_data['summary']['ablated_accuracy']*100:.1f}%")
    print(f"Avg original loss: {results_data['summary']['avg_original_loss']:.4f}")
    print(f"Avg ablated loss: {results_data['summary']['avg_ablated_loss']:.4f}")
    print(f"Ablated features: {results_data['ablated_features'][:10]}..." if len(results_data['ablated_features']) > 10 else f"Ablated features: {results_data['ablated_features']}")
    print("="*70)


def visualize_from_batch_data(args):
    """Visualize predictions from batch_data files."""
    print("="*70)
    print("Visualizing Batch Data Files")
    print("="*70)
    
    # Load puzzle metadata
    identifiers_map, test_puzzles = load_puzzle_metadata(args.data_path)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find batch data files
    batch_files = sorted([
        f for f in os.listdir(args.batch_data_dir) 
        if f.startswith('batch_data_') and f.endswith('.pt')
    ])
    
    if not batch_files:
        print(f"No batch_data files found in {args.batch_data_dir}")
        return
    
    print(f"Found {len(batch_files)} batch data files")
    
    all_results = []
    for batch_idx, batch_file in enumerate(batch_files[:args.max_batches]):
        batch_path = os.path.join(args.batch_data_dir, batch_file)
        print(f"\nProcessing: {batch_file}")
        
        batch_data = load_batch_data(batch_path)
        results = visualize_batch_predictions(
            batch_data=batch_data,
            identifiers_map=identifiers_map,
            test_puzzles=test_puzzles,
            output_dir=args.output_dir,
            batch_idx=batch_idx,
        )
        all_results.extend(results)
    
    # Summary
    correct_count = sum(1 for r in all_results if r['is_correct'])
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total visualized: {len(all_results)}")
    print(f"Correct: {correct_count} ({correct_count/len(all_results)*100:.1f}%)")
    print(f"Incorrect: {len(all_results) - correct_count}")
    print(f"Output: {args.output_dir}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Visualize Ablation Analysis Predictions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From ablation results JSON:
  python visualize_ablation_predictions.py --results_json results.json --data_path data/arc1concept-aug-0

  # From batch_data files:
  python visualize_ablation_predictions.py --batch_data_dir results/ --data_path data/arc1concept-aug-0
        """
    )
    
    # Input sources (mutually exclusive)
    input_group = parser.add_argument_group('Input Sources')
    input_group.add_argument('--results_json', type=str, default=None,
                            help='Path to ablation_results.json file')
    input_group.add_argument('--batch_data_dir', type=str, default=None,
                            help='Directory containing batch_data_*.pt files')
    
    # Data paths
    data_group = parser.add_argument_group('Data Paths')
    data_group.add_argument('--data_path', type=str, 
                           default='data/arc1concept-aug-0',
                           help='Path to dataset with identifiers.json and test_puzzles.json')
    data_group.add_argument('--output_dir', type=str,
                           default='results-analysis-noaug/ablation_predictions_viz',
                           help='Output directory for visualizations')
    
    # Options
    opt_group = parser.add_argument_group('Options')
    opt_group.add_argument('--max_batches', type=int, default=10,
                          help='Maximum number of batch files to process')
    opt_group.add_argument('--max_examples', type=int, default=50,
                          help='Maximum number of examples to visualize')
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.results_json is None and args.batch_data_dir is None:
        # Try to find results automatically
        default_results = 'results-analysis-noaug/ablation_visualizations/ablation_results.json'
        if os.path.exists(default_results):
            args.results_json = default_results
            print(f"Auto-detected results file: {default_results}")
        else:
            parser.error("Please specify --results_json or --batch_data_dir")
    
    # Run appropriate visualization
    if args.results_json:
        visualize_from_results_json(args)
    elif args.batch_data_dir:
        visualize_from_batch_data(args)


if __name__ == "__main__":
    main()
