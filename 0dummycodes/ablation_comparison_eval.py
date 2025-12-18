"""
Ablation Comparison Evaluation with Visualization

This script compares model predictions:
1. Original predictions (no SAE ablation)
2. Predictions after ablating top-k SAE features

Workflow:
1. Load TRM model and SAE
2. Run inference to get original predictions and z_L trajectories
3. Encode z_L through SAE to get feature activations
4. Compute feature importance (for incorrect predictions)
5. Ablate top-k features and decode back through SAE
6. Inject reconstructed z_L and get ablated predictions
7. Visualize original vs ablated predictions side-by-side

Usage:
    python ablation_comparison_eval.py \
        data_paths="['data/arc1concept-aug-1000']" \
        data_paths_test="['data/arc1concept-aug-0']" \
        load_checkpoint="ckpt/arc_v1_public/step_518071" \
        checkpoint_path="ckpt/ablation_comparison"
"""

from typing import Optional, Any, Sequence, List, Dict, Tuple
from dataclasses import dataclass, fields
import os
import math
import yaml
import shutil
import copy
import json

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.gridspec as gridspec

import tqdm
import coolname
import hydra
import pydantic
from omegaconf import DictConfig

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from utils.functions import load_model_class, get_model_source_path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16

# =============================================================================
# ARC VISUALIZATION
# =============================================================================
ARC_COLORS = [
    '#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
    '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25',
]

def decode_arc_grid(tokens: np.ndarray) -> np.ndarray:
    """Decode tokenized ARC grid to 2D grid."""
    if tokens.ndim == 1:
        if len(tokens) != 900:
            tokens = tokens[:900] if len(tokens) > 900 else np.pad(tokens, (0, 900 - len(tokens)))
        grid = tokens.reshape(30, 30).astype(np.int32)
    else:
        grid = tokens.astype(np.int32)
    
    max_area, max_size = 0, (0, 0)
    num_c = 30
    for num_r in range(1, 31):
        for c in range(1, num_c + 1):
            x = grid[num_r - 1, c - 1]
            if x < 2 or x > 11:
                num_c = c - 1
                break
        if num_r * num_c > max_area:
            max_area, max_size = num_r * num_c, (num_r, num_c)
    
    if max_size[0] == 0 or max_size[1] == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    return (grid[:max_size[0], :max_size[1]] - 2).astype(np.uint8)


def visualize_grid(grid: np.ndarray, ax: plt.Axes, title: str = "", 
                   highlight_color: str = None, show_diff: np.ndarray = None):
    """Visualize an ARC grid."""
    H, W = grid.shape
    cmap = ListedColormap(ARC_COLORS)
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=9, aspect='equal')
    
    for i in range(H + 1):
        ax.axhline(i - 0.5, color='white', linewidth=0.5, alpha=0.3)
    for j in range(W + 1):
        ax.axvline(j - 0.5, color='white', linewidth=0.5, alpha=0.3)
    
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
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)


def compute_diff(grid1: np.ndarray, grid2: np.ndarray) -> Tuple[np.ndarray, int]:
    """Compute difference between two grids."""
    h1, w1 = grid1.shape
    h2, w2 = grid2.shape
    max_h, max_w = max(h1, h2), max(w1, w2)
    p1 = np.full((max_h, max_w), -1, dtype=np.int32)
    p2 = np.full((max_h, max_w), -1, dtype=np.int32)
    p1[:h1, :w1] = grid1
    p2[:h2, :w2] = grid2
    diff = p1 != p2
    return diff[:h1, :w1], int(np.sum(diff))


# =============================================================================
# SAE MODEL
# =============================================================================
class SAE(nn.Module):
    """Simple SAE for z_L encoding/decoding."""
    def __init__(self, d_model: int = 512, depth: int = 16, 
                 n_features: int = 4096, topk: int = 64):
        super().__init__()
        self.d_model = d_model
        self.depth = depth
        self.n_features = n_features
        self.topk = topk
        
        self.dictionary_enc = nn.Parameter(torch.randn(n_features, d_model, dtype=DTYPE) * (2.0 / (d_model + n_features)) ** 0.5)
        self.dictionary_dec = nn.Parameter(torch.randn(d_model, n_features, dtype=DTYPE) * (2.0 / (n_features + d_model)) ** 0.5)
        self.dictionary_dec.data.copy_(self.dictionary_enc.data.T)
        self.bias_pre = nn.Parameter(torch.zeros(d_model, dtype=DTYPE))
        self.bias_enc = nn.Parameter(torch.zeros(n_features, dtype=DTYPE))
        
        self.register_buffer("num_tokens_seen", torch.zeros((), dtype=torch.long))
        self.register_buffer("last_active_token", torch.zeros(n_features, dtype=torch.long))

    def topk_activation(self, x):
        if self.topk is None or self.topk >= x.size(-1):
            return x
        values, indices = torch.topk(x, self.topk, dim=-1)
        out = torch.zeros_like(x)
        out.scatter_(dim=-1, index=indices, src=values)
        return out

    def encode(self, zL: torch.Tensor) -> torch.Tensor:
        """zL: [B, D, L, H] -> z_n: [B, D, L, M]"""
        B, D, L, H = zL.shape
        x_src = zL.reshape(B * D * L, H)
        logits = F.linear(x_src - self.bias_pre[None, :], self.dictionary_enc, self.bias_enc)
        z_n = self.topk_activation(F.relu(logits))
        return z_n.view(B, D, L, self.n_features)

    def decode(self, z_n: torch.Tensor) -> torch.Tensor:
        """z_n: [B, D, L, M] -> zL: [B, D, L, H]"""
        B, D, L, M = z_n.shape
        z_n_flat = z_n.reshape(B * D * L, M)
        x_tgt = F.linear(z_n_flat, self.dictionary_dec) + self.bias_pre[None, :]
        return x_tgt.view(B, D, L, self.d_model)


def load_sae(checkpoint_path: str, d_model: int = 512, depth: int = 16,
             n_features: int = 4096, topk: int = 64) -> SAE:
    """Load SAE from checkpoint."""
    sae = SAE(d_model=d_model, depth=depth, n_features=n_features, topk=topk)
    sae = sae.to(device=device, dtype=DTYPE)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'sae_state_dict' in checkpoint:
        sae.load_state_dict(checkpoint['sae_state_dict'])
    else:
        sae.load_state_dict(checkpoint)
    sae.eval()
    return sae


# =============================================================================
# CARRY MANIPULATION
# =============================================================================
def move_carry_to_device(carry, target_device):
    """Recursively move carry to device."""
    if hasattr(carry, '__dataclass_fields__'):
        updates = {}
        for field in fields(carry):
            value = getattr(carry, field.name)
            if isinstance(value, torch.Tensor):
                updates[field.name] = value.to(target_device)
            elif hasattr(value, '__dataclass_fields__'):
                updates[field.name] = move_carry_to_device(value, target_device)
            else:
                updates[field.name] = value
        return type(carry)(**updates)
    return carry


def replace_z_L_in_carry(carry, new_z_L):
    """Replace z_L in carry with new tensor."""
    if hasattr(carry, 'inner_carry') and hasattr(carry.inner_carry, 'z_L'):
        inner_updates = {f.name: getattr(carry.inner_carry, f.name) for f in fields(carry.inner_carry)}
        inner_updates['z_L'] = new_z_L
        new_inner = type(carry.inner_carry)(**inner_updates)
        
        outer_updates = {f.name: getattr(carry, f.name) for f in fields(carry)}
        outer_updates['inner_carry'] = new_inner
        return type(carry)(**outer_updates)
    return carry


# =============================================================================
# ABLATION FUNCTIONS
# =============================================================================
def compute_feature_importance(z_n: torch.Tensor, predictions: np.ndarray, 
                                labels: np.ndarray) -> Tuple[torch.Tensor, np.ndarray]:
    """Compute feature importance for incorrect predictions."""
    B, D, L, M = z_n.shape
    
    incorrect_mask = np.zeros(B, dtype=bool)
    for b in range(B):
        pred_grid = decode_arc_grid(predictions[b])
        label_grid = decode_arc_grid(labels[b])
        if not np.array_equal(pred_grid, label_grid):
            incorrect_mask[b] = True
    
    incorrect_indices = np.where(incorrect_mask)[0]
    if len(incorrect_indices) == 0:
        return torch.ones(M, device=z_n.device) / M, incorrect_mask
    
    z_n_incorrect = z_n[incorrect_indices]
    importance = z_n_incorrect.mean(dim=(0, 1, 2))
    return importance, incorrect_mask


def ablate_features(z_n: torch.Tensor, feature_indices: torch.Tensor) -> torch.Tensor:
    """Zero out specific features."""
    z_n_ablated = z_n.clone()
    z_n_ablated[:, :, :, feature_indices] = 0
    return z_n_ablated


# =============================================================================
# VISUALIZATION
# =============================================================================
def visualize_ablation_comparison(
    puzzle_name: str, puzzle_id: int, demo_examples: List[Dict],
    test_input: np.ndarray, test_label: np.ndarray,
    original_pred: np.ndarray, ablated_pred: np.ndarray,
    original_loss: float, ablated_loss: float,
    num_ablated_features: int, output_path: str,
) -> Dict:
    """Visualize original vs ablated predictions."""
    num_demo = min(len(demo_examples), 3)
    total_rows = num_demo + 3  # demos + target + original + ablated
    
    fig = plt.figure(figsize=(12, total_rows * 2.5))
    gs = gridspec.GridSpec(total_rows, 3, width_ratios=[1, 1, 1])
    
    # Demo examples
    for row, demo in enumerate(demo_examples[:num_demo]):
        ax_in = fig.add_subplot(gs[row, 0])
        ax_out = fig.add_subplot(gs[row, 1])
        ax_info = fig.add_subplot(gs[row, 2])
        visualize_grid(np.array(demo['input'], dtype=np.uint8), ax_in, f"Demo {row+1} Input")
        visualize_grid(np.array(demo['output'], dtype=np.uint8), ax_out, f"Demo {row+1} Output")
        ax_info.axis('off')
    
    # Decode grids
    input_grid = decode_arc_grid(test_input)
    label_grid = decode_arc_grid(test_label)
    orig_grid = decode_arc_grid(original_pred)
    abl_grid = decode_arc_grid(ablated_pred)
    
    # Target row
    row = num_demo
    ax = fig.add_subplot(gs[row, 0])
    visualize_grid(input_grid, ax, "Test Input")
    ax = fig.add_subplot(gs[row, 1])
    visualize_grid(label_grid, ax, "Target (Ground Truth)", highlight_color='#0074D9')
    ax = fig.add_subplot(gs[row, 2])
    ax.axis('off')
    ax.text(0.5, 0.5, f"Target\n{label_grid.shape[0]}×{label_grid.shape[1]}", 
            ha='center', va='center', fontsize=10, transform=ax.transAxes)
    
    # Original prediction row
    row = num_demo + 1
    orig_correct = np.array_equal(label_grid, orig_grid)
    orig_diff, orig_ndiff = compute_diff(label_grid, orig_grid)
    orig_color = '#2ECC40' if orig_correct else '#FF4136'
    orig_status = "CORRECT" if orig_correct else f"WRONG ({orig_ndiff} diff)"
    
    ax = fig.add_subplot(gs[row, 0])
    visualize_grid(input_grid, ax, "Original Input")
    ax = fig.add_subplot(gs[row, 1])
    visualize_grid(orig_grid, ax, f"Original: {orig_status}", highlight_color=orig_color,
                   show_diff=orig_diff if not orig_correct else None)
    ax = fig.add_subplot(gs[row, 2])
    ax.axis('off')
    ax.text(0.5, 0.5, f"Original\nLoss: {original_loss:.4f}\n{orig_status}", 
            ha='center', va='center', fontsize=10, color=orig_color, fontweight='bold',
            transform=ax.transAxes)
    
    # Ablated prediction row
    row = num_demo + 2
    abl_correct = np.array_equal(label_grid, abl_grid)
    abl_diff, abl_ndiff = compute_diff(label_grid, abl_grid)
    abl_color = '#2ECC40' if abl_correct else '#FF4136'
    abl_status = "CORRECT" if abl_correct else f"WRONG ({abl_ndiff} diff)"
    
    ax = fig.add_subplot(gs[row, 0])
    visualize_grid(input_grid, ax, "Ablated Input")
    ax = fig.add_subplot(gs[row, 1])
    visualize_grid(abl_grid, ax, f"Ablated: {abl_status}", highlight_color=abl_color,
                   show_diff=abl_diff if not abl_correct else None)
    ax = fig.add_subplot(gs[row, 2])
    ax.axis('off')
    loss_delta = ablated_loss - original_loss
    ax.text(0.5, 0.5, f"After Ablation\nLoss: {ablated_loss:.4f} ({'+' if loss_delta >= 0 else ''}{loss_delta:.4f})\n"
            f"{abl_status}\n{num_ablated_features} features ablated", 
            ha='center', va='center', fontsize=10, color=abl_color, fontweight='bold',
            transform=ax.transAxes)
    
    # Title
    change = ""
    if orig_correct and not abl_correct:
        change = " [DEGRADED]"
    elif not orig_correct and abl_correct:
        change = " [IMPROVED!]"
    
    plt.suptitle(f"Ablation Comparison: {puzzle_name} (ID: {puzzle_id}){change}\n"
                f"Loss: {original_loss:.4f} → {ablated_loss:.4f}", fontsize=11, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()
    
    return {
        'puzzle_name': puzzle_name, 'puzzle_id': puzzle_id,
        'original_correct': orig_correct, 'ablated_correct': abl_correct,
        'original_loss': original_loss, 'ablated_loss': ablated_loss,
        'original_diff': orig_ndiff, 'ablated_diff': abl_ndiff,
        'change': 'improved' if not orig_correct and abl_correct else 
                  'degraded' if orig_correct and not abl_correct else 'unchanged'
    }


def create_summary(results: List[Dict], output_path: str, topk: int):
    """Create summary visualization."""
    n = len(results)
    if n == 0:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    orig_correct = sum(1 for r in results if r['original_correct'])
    abl_correct = sum(1 for r in results if r['ablated_correct'])
    
    # Accuracy comparison
    ax = axes[0, 0]
    x = ['Original', 'After Ablation']
    ax.bar(x, [orig_correct, abl_correct], color='#2ECC40', alpha=0.8, label='Correct')
    ax.bar(x, [n - orig_correct, n - abl_correct], bottom=[orig_correct, abl_correct], 
           color='#FF4136', alpha=0.8, label='Incorrect')
    ax.set_ylabel('Count')
    ax.set_title(f'Accuracy (Top-{topk} Ablation)')
    ax.legend()
    
    # Loss distribution
    ax = axes[0, 1]
    deltas = [r['ablated_loss'] - r['original_loss'] for r in results]
    ax.hist(deltas, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', label='No change')
    ax.axvline(np.mean(deltas), color='orange', label=f'Mean: {np.mean(deltas):+.4f}')
    ax.set_xlabel('Loss Change')
    ax.set_title('Loss Change Distribution')
    ax.legend()
    
    # Change breakdown
    ax = axes[1, 0]
    labels = ['Stayed Correct', 'Improved', 'Stayed Wrong', 'Degraded']
    sizes = [
        sum(1 for r in results if r['original_correct'] and r['ablated_correct']),
        sum(1 for r in results if not r['original_correct'] and r['ablated_correct']),
        sum(1 for r in results if not r['original_correct'] and not r['ablated_correct']),
        sum(1 for r in results if r['original_correct'] and not r['ablated_correct']),
    ]
    colors = ['#2ECC40', '#7FDBFF', '#FFDC00', '#FF4136']
    non_zero = [(s, l, c) for s, l, c in zip(sizes, labels, colors) if s > 0]
    if non_zero:
        sizes_nz, labels_nz, colors_nz = zip(*non_zero)
        ax.pie(sizes_nz, labels=labels_nz, colors=colors_nz, autopct=lambda p: f'{int(p*n/100)}')
    ax.set_title('Prediction Changes')
    
    # Loss scatter
    ax = axes[1, 1]
    orig_losses = [r['original_loss'] for r in results]
    abl_losses = [r['ablated_loss'] for r in results]
    colors_scatter = ['#2ECC40' if r['original_correct'] else '#FF4136' for r in results]
    ax.scatter(orig_losses, abl_losses, c=colors_scatter, alpha=0.6, s=40, edgecolors='k', linewidths=0.5)
    lims = [min(orig_losses + abl_losses), max(orig_losses + abl_losses)]
    ax.plot(lims, lims, 'k--', alpha=0.5)
    ax.set_xlabel('Original Loss')
    ax.set_ylabel('Ablated Loss')
    ax.set_title('Loss Comparison')
    
    plt.suptitle(f'Ablation Summary: Top-{topk} Features\n'
                f'Accuracy: {orig_correct}/{n} ({100*orig_correct/n:.1f}%) → {abl_correct}/{n} ({100*abl_correct/n:.1f}%)',
                fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# CONFIG
# =============================================================================
class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str

class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str
    loss: LossConfig

class EvalConfig(pydantic.BaseModel):
    arch: ArchConfig
    data_paths: List[str]
    data_paths_test: List[str] = []
    global_batch_size: int
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    load_checkpoint: Optional[str] = None
    checkpoint_path: Optional[str] = None
    seed: int = 0
    eval_save_outputs: List[str] = []
    # SAE settings
    sae_checkpoint: str = "weights/sae/best_val.pt"
    topk_ablate: int = 20
    max_batches: int = 5
    max_examples_per_batch: int = 10


@dataclass
class EvalState:
    model: nn.Module
    carry: Any


# =============================================================================
# MAIN EVALUATION
# =============================================================================
def create_dataloader(config: EvalConfig, split: str, **kwargs):
    dataset = PuzzleDataset(PuzzleDatasetConfig(
        seed=config.seed,
        dataset_paths=config.data_paths_test if config.data_paths_test and split == "test" else config.data_paths,
        rank=0, num_replicas=1, **kwargs
    ), split=split)
    dataloader = DataLoader(dataset, batch_size=None, num_workers=1, prefetch_factor=8, 
                           pin_memory=True, persistent_workers=True)
    return dataloader, dataset.metadata


def create_model(config: EvalConfig, metadata: PuzzleDatasetMetadata):
    model_cfg = dict(
        **config.arch.__pydantic_extra__,
        batch_size=config.global_batch_size,
        vocab_size=metadata.vocab_size,
        seq_len=metadata.seq_len,
        num_puzzle_identifiers=metadata.num_puzzle_identifiers,
        causal=False
    )
    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)
    
    with device:
        model = model_cls(model_cfg)
        model = loss_head_cls(model, **config.arch.loss.__pydantic_extra__)
        if "DISABLE_COMPILE" not in os.environ:
            model = torch.compile(model)
    
    if config.load_checkpoint:
        print(f"Loading checkpoint: {config.load_checkpoint}")
        state_dict = torch.load(config.load_checkpoint, map_location=device, weights_only=False)
        puzzle_emb_name = "_orig_mod.model.inner.puzzle_emb.weights"
        expected_shape = model.model.puzzle_emb.weights.shape
        if puzzle_emb_name in state_dict and state_dict[puzzle_emb_name].shape != expected_shape:
            print(f"Resetting puzzle embedding: {state_dict[puzzle_emb_name].shape} -> {expected_shape}")
            state_dict[puzzle_emb_name] = torch.mean(state_dict[puzzle_emb_name], dim=0, keepdim=True).expand(expected_shape).contiguous()
        model.load_state_dict(state_dict, assign=True)
    
    return model


def run_inference_with_trajectories(model, batch, max_steps=100):
    """Run inference and collect z_L trajectories."""
    trajectories = []
    
    with torch.inference_mode():
        carry = model.initial_carry(batch)
        carry = move_carry_to_device(carry, device)
        
        if hasattr(carry, 'inner_carry') and hasattr(carry.inner_carry, 'z_L'):
            trajectories.append(carry.inner_carry.z_L.detach().clone())
        
        for step in range(max_steps):
            carry, loss, metrics, preds, all_finish = model(carry=carry, batch=batch, return_keys={"preds"})
            
            if hasattr(carry, 'inner_carry') and hasattr(carry.inner_carry, 'z_L'):
                trajectories.append(carry.inner_carry.z_L.detach().clone())
            
            if all_finish:
                break
        
        predictions = preds["preds"].cpu().numpy()
        loss_val = loss.item()
    
    z_L_trajectories = torch.stack(trajectories, dim=0)  # [steps, B, L, H]
    return z_L_trajectories, predictions, loss_val


def run_inference_with_injection(model, batch, z_L_inject, inject_step=10, max_steps=100):
    """Run inference with z_L injection at specific step."""
    with torch.inference_mode():
        carry = model.initial_carry(batch)
        carry = move_carry_to_device(carry, device)
        
        for step in range(max_steps):
            if step == inject_step:
                carry = replace_z_L_in_carry(carry, z_L_inject)
            
            carry, loss, metrics, preds, all_finish = model(carry=carry, batch=batch, return_keys={"preds"})
            
            if all_finish:
                break
        
        return preds["preds"].cpu().numpy(), loss.item()


def evaluate(config: EvalConfig, model, eval_loader, metadata):
    """Main evaluation with ablation comparison."""
    RESULT_DIR = config.checkpoint_path.replace("ckpt/", "results/")
    VIZ_DIR = os.path.join(RESULT_DIR, "ablation_comparison")
    os.makedirs(VIZ_DIR, exist_ok=True)
    
    # Load SAE
    print(f"Loading SAE: {config.sae_checkpoint}")
    sae = load_sae(config.sae_checkpoint)
    
    # Load puzzle metadata
    data_path = config.data_paths_test[0] if config.data_paths_test else config.data_paths[0]
    identifiers_map, test_puzzles = [], {}
    try:
        with open(os.path.join(data_path, "identifiers.json"), 'r') as f:
            identifiers_map = json.load(f)
        with open(os.path.join(data_path, "test_puzzles.json"), 'r') as f:
            test_puzzles = json.load(f)
        print(f"Loaded {len(identifiers_map)} puzzle identifiers")
    except Exception as e:
        print(f"Warning: Could not load metadata: {e}")
    
    all_results = []
    all_z_n = []
    all_predictions = []
    all_labels = []
    processed_batches = 0
    
    print(f"\nRunning ablation comparison (top-{config.topk_ablate} features)...")
    
    with torch.inference_mode():
        for set_name, batch, _ in eval_loader:
            processed_batches += 1
            if processed_batches > config.max_batches:
                break
            
            print(f"\nBatch {processed_batches}: {set_name}")
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Run original inference
            z_L_traj, orig_preds, orig_loss = run_inference_with_trajectories(model, batch)
            print(f"  Original loss: {orig_loss:.4f}, trajectories: {z_L_traj.shape}")
            
            # Get z_L for SAE (use last step, reshape to [B, D, L, H])
            # z_L_traj is [steps, B, L, H], we want [B, steps, L, H] for SAE
            z_L_for_sae = z_L_traj.permute(1, 0, 2, 3).to(dtype=DTYPE)  # [B, D, L, H]
            
            # Encode through SAE
            z_n = sae.encode(z_L_for_sae)  # [B, D, L, M]
            print(f"  SAE features: {z_n.shape}")
            
            # Store for importance computation
            B = len(orig_preds)
            labels = batch['labels'].cpu().numpy()
            all_z_n.append(z_n.cpu())
            all_predictions.append(orig_preds)
            all_labels.append(labels)
    
    # Concatenate all batches
    all_z_n = torch.cat(all_z_n, dim=0).to(device).to(dtype=DTYPE)
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    print(f"\nTotal examples: {len(all_predictions)}")
    print(f"Computing feature importance...")
    
    # Compute feature importance
    importance, incorrect_mask = compute_feature_importance(all_z_n, all_predictions, all_labels)
    topk_indices = torch.topk(importance, config.topk_ablate).indices
    print(f"Top-{config.topk_ablate} features to ablate: {topk_indices[:10].tolist()}...")
    
    # Re-run with ablation for visualization
    print(f"\nRunning ablation and creating visualizations...")
    processed = 0
    
    for set_name, batch, _ in eval_loader:
        if processed >= min(config.max_batches * config.global_batch_size, len(all_predictions)):
            break
        
        batch = {k: v.to(device) for k, v in batch.items()}
        B = batch['inputs'].shape[0]
        
        for i in range(min(B, config.max_examples_per_batch)):
            if processed >= len(all_predictions):
                break
            
            # Single example batch
            single_batch = {k: v[i:i+1] for k, v in batch.items()}
            
            # Get original prediction
            z_L_traj, orig_pred, orig_loss = run_inference_with_trajectories(model, single_batch)
            
            # Ablate and reconstruct
            z_L_for_sae = z_L_traj.permute(1, 0, 2, 3).to(dtype=DTYPE)
            z_n = sae.encode(z_L_for_sae)
            z_n_ablated = ablate_features(z_n, topk_indices)
            z_L_reconstructed = sae.decode(z_n_ablated)
            
            # Inject and get ablated prediction
            z_L_inject = z_L_reconstructed[:, -1, :, :].float()  # Last step
            abl_pred, abl_loss = run_inference_with_injection(model, single_batch, z_L_inject, inject_step=10)
            
            # Get puzzle info
            inputs = single_batch['inputs'].cpu().numpy()[0]
            labels = single_batch['labels'].cpu().numpy()[0]
            puzzle_ids = single_batch.get('puzzle_identifiers')
            pid = int(puzzle_ids[0]) if puzzle_ids is not None else processed
            puzzle_name = identifiers_map[pid] if pid < len(identifiers_map) else f"puzzle_{pid}"
            base_name = puzzle_name.split("|||")[0]
            demos = test_puzzles.get(base_name, {}).get("train", [])
            
            # Visualize
            output_path = os.path.join(VIZ_DIR, f"ablation_{processed:03d}_{base_name}.png")
            result = visualize_ablation_comparison(
                puzzle_name=base_name, puzzle_id=pid, demo_examples=demos,
                test_input=inputs, test_label=labels,
                original_pred=orig_pred[0], ablated_pred=abl_pred[0],
                original_loss=orig_loss, ablated_loss=abl_loss,
                num_ablated_features=config.topk_ablate, output_path=output_path,
            )
            all_results.append(result)
            
            status = f"{result['change'].upper()}" if result['change'] != 'unchanged' else 'unchanged'
            print(f"  [{processed+1}] {base_name}: {status} (loss: {orig_loss:.2f} -> {abl_loss:.2f})")
            processed += 1
    
    # Create summary
    if all_results:
        summary_path = os.path.join(VIZ_DIR, "ablation_summary.png")
        create_summary(all_results, summary_path, config.topk_ablate)
        
        # Save results JSON
        with open(os.path.join(VIZ_DIR, "ablation_results.json"), 'w') as f:
            json.dump({
                'config': {'topk_ablate': config.topk_ablate, 'num_examples': len(all_results)},
                'ablated_features': topk_indices.tolist(),
                'results': all_results,
            }, f, indent=2)
        
        # Print summary
        print("\n" + "="*60)
        print("ABLATION COMPARISON SUMMARY")
        print("="*60)
        orig_acc = sum(1 for r in all_results if r['original_correct'])
        abl_acc = sum(1 for r in all_results if r['ablated_correct'])
        improved = sum(1 for r in all_results if r['change'] == 'improved')
        degraded = sum(1 for r in all_results if r['change'] == 'degraded')
        print(f"Total: {len(all_results)}")
        print(f"Original accuracy: {orig_acc}/{len(all_results)} ({100*orig_acc/len(all_results):.1f}%)")
        print(f"Ablated accuracy: {abl_acc}/{len(all_results)} ({100*abl_acc/len(all_results):.1f}%)")
        print(f"Improved: {improved}, Degraded: {degraded}")
        print(f"Visualizations: {VIZ_DIR}")
        print("="*60)


@hydra.main(config_path="config", config_name="cfg_eval", version_base=None)
def launch(hydra_config: DictConfig):
    config = EvalConfig(**hydra_config)
    
    if config.project_name is None:
        config.project_name = "ablation-comparison"
    if config.run_name is None:
        config.run_name = f"ablation_{coolname.generate_slug(2)}"
    if config.checkpoint_path is None:
        config.checkpoint_path = os.path.join("ckpt", config.project_name, config.run_name)
    
    torch.random.manual_seed(config.seed)
    
    try:
        eval_loader, metadata = create_dataloader(config, "test", test_set_mode=True, 
                                                   epochs_per_iter=1, global_batch_size=config.global_batch_size)
    except:
        print("No test data, using train")
        eval_loader, metadata = create_dataloader(config, "train", test_set_mode=False,
                                                   epochs_per_iter=1, global_batch_size=config.global_batch_size)
    
    model = create_model(config, metadata)
    model.eval()
    
    evaluate(config, model, eval_loader, metadata)


if __name__ == "__main__":
    launch()
