"""
Ablation Analysis Visualization for ARC-AGI-1 puzzles with TRM + SAE.

This script performs ablation experiments on SAE features and visualizes:
- Original predictions (before ablation)
- Predictions after ablating top-k important features
- Feature importance analysis
- Loss/accuracy changes

Workflow:
1. Load TRM model and SAE checkpoint
2. Run inference to collect z_L trajectories [B, D, L, H]
3. Pass z_L through SAE encoder to get feature activations
4. Identify top-k important features (for incorrect predictions)
5. Ablate features (set to zero) and reconstruct z_L via SAE decoder
6. Inject reconstructed z_L back into model
7. Measure accuracy/loss changes and visualize

Usage:
    python analysis-yong/visualize_ablation.py \
        --data_path data/arc1concept-aug-0 \
        --model_data_path data/arc1concept-aug-1000 \
        --checkpoint ckpt/arc_v1_public/step_518071 \
        --config_path ckpt/arc_v1_public \
        --sae_checkpoint weights/sae/best_val.pt \
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

import torch
from torch import nn
from torch.nn import functional as F

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from puzzle_dataset import PuzzleDatasetConfig, PuzzleDataset
from dataset.common import PuzzleDatasetMetadata
from models.losses import IGNORE_LABEL_ID
from utils.functions import load_model_class

# Global dtype for SAE (match sae_fix.py)
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
# SAE Model Definition (copied from sae_fix.py for standalone use)
# ============================================================================
class SAE(nn.Module):
    """
    Simple SAE without attention
    zL: [B, D, L, H] (B=batch, D=depth, L=seq len, H=d_model)
    """
    def __init__(
        self,
        d_model: int,
        depth: int,
        n_heads: int = 8,
        n_features: int = 4096,
        topk: int = 64,
        lambda_sparse: float = 1e-3,
        n_registers: int = 4,
        auxk_topk: int = 512,
        aux_alpha: float = 1.0 / 32.0,
        dead_token_threshold: int = 200_000,
    ):
        super().__init__()
        H = d_model
        D = depth

        self.depth = D
        self.d_model = H
        self.n_features = n_features
        self.topk = topk
        self.lambda_sparse = lambda_sparse
        self.n_registers = n_registers
        self.auxk_topk = auxk_topk
        self.aux_alpha = aux_alpha
        self.dead_token_threshold = dead_token_threshold

        # SAE dictionary: H -> M (encoder), M -> H (decoder)
        self.dictionary_enc = nn.Parameter(torch.randn(n_features, H, dtype=DTYPE) * (2.0 / (H + n_features)) ** 0.5)
        self.dictionary_dec = nn.Parameter(torch.randn(H, n_features, dtype=DTYPE) * (2.0 / (n_features + H)) ** 0.5)
        self.dictionary_dec.data.copy_(self.dictionary_enc.data.T)
        self.bias_pre = nn.Parameter(torch.zeros(H, dtype=DTYPE))
        self.bias_enc = nn.Parameter(torch.zeros(n_features, dtype=DTYPE))

        # AuxK statistics buffers
        self.register_buffer("num_tokens_seen", torch.zeros((), dtype=torch.long))
        self.register_buffer("last_active_token", torch.zeros(n_features, dtype=torch.long))

    def topk_activation(self, x):
        """Keep only TopK activations, set the rest to 0"""
        if self.topk is None or self.topk >= x.size(-1):
            return x
        values, indices = torch.topk(x, self.topk, dim=-1)
        out = torch.zeros_like(x)
        out.scatter_(dim=-1, index=indices, src=values)
        return out

    def encode(self, zL: torch.Tensor) -> torch.Tensor:
        """
        Encode input to sparse features.
        zL: [B, D, L, H] -> z_n: [B, D, L, M]
        """
        B, D, L, H = zL.shape
        N = B * D * L
        x_src = zL.reshape(N, H)
        
        logits = F.linear(x_src - self.bias_pre[None, :], self.dictionary_enc, self.bias_enc)
        z_n_dense = F.relu(logits)
        z_n_flat = self.topk_activation(z_n_dense)
        z_n = z_n_flat.view(B, D, L, self.n_features)
        return z_n

    def decode(self, z_n: torch.Tensor) -> torch.Tensor:
        """
        Decode sparse features back to input space.
        z_n: [B, D, L, M] -> x_tgt: [B, D, L, H]
        """
        B, D, L, M = z_n.shape
        z_n_flat = z_n.reshape(B * D * L, M)
        x_tgt_flat = F.linear(z_n_flat, self.dictionary_dec) + self.bias_pre[None, :]
        x_tgt = x_tgt_flat.view(B, D, L, self.d_model)
        return x_tgt

    def forward(self, zL, mask=None):
        """Full forward pass (encode + decode)"""
        z_n = self.encode(zL)
        x_tgt = self.decode(z_n)
        
        x_src_reshaped = zL
        recon_loss = F.mse_loss(x_tgt, x_src_reshaped)
        
        return {
            "loss": recon_loss,
            "recon_loss": recon_loss.detach(),
            "x_tgt": x_tgt,
            "x_src": x_src_reshaped,
            "z_n": z_n,
        }


# ============================================================================
# Utility Functions
# ============================================================================
def decode_arc_grid(tokens: np.ndarray, debug: bool = False) -> np.ndarray:
    """Decode tokenized ARC grid back to 2D grid"""
    if debug:
        unique_values = np.unique(tokens)
        print(f"  Token shape: {tokens.shape}, unique values: {unique_values}")
    
    grid_30x30 = tokens.reshape(30, 30).astype(np.int32)
    
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
                       highlight_color: str = None):
    """Visualize an ARC grid"""
    H, W = grid.shape
    cmap = ListedColormap(ARC_COLORS)
    
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=9, aspect='equal')
    
    for i in range(H + 1):
        ax.axhline(i - 0.5, color='white', linewidth=0.5, alpha=0.3)
    for j in range(W + 1):
        ax.axvline(j - 0.5, color='white', linewidth=0.5, alpha=0.3)
    
    ax.set_xticks([])
    ax.set_yticks([])
    
    if highlight_color:
        for spine in ax.spines.values():
            spine.set_edgecolor(highlight_color)
            spine.set_linewidth(3)
    
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)


def build_identifier_mapping(train_data_path: str, test_data_path: str, 
                             use_augmented: bool = True, seed: int = 42) -> Dict[int, int]:
    """Build mapping from test dataset puzzle_identifiers to train dataset puzzle_identifiers."""
    import random
    random.seed(seed)
    
    with open(os.path.join(test_data_path, "identifiers.json"), 'r') as f:
        test_identifiers = json.load(f)
    
    with open(os.path.join(train_data_path, "identifiers.json"), 'r') as f:
        content = f.read()
        train_identifiers = ast.literal_eval(content)
    
    mapping = {}
    for test_idx, identifier in enumerate(test_identifiers):
        if identifier == '<blank>':
            mapping[test_idx] = 0
            continue
            
        if use_augmented:
            augmented_ids = [idx for idx, name in enumerate(train_identifiers) 
                           if name.startswith(identifier + '|||')]
            
            if augmented_ids:
                selected_id = random.choice(augmented_ids)
                mapping[test_idx] = selected_id
            else:
                if identifier in train_identifiers:
                    mapping[test_idx] = train_identifiers.index(identifier)
                else:
                    mapping[test_idx] = 0
        else:
            if identifier in train_identifiers:
                mapping[test_idx] = train_identifiers.index(identifier)
            else:
                mapping[test_idx] = 0
    
    return mapping


def move_carry_to_device(carry, target_device):
    """Recursively move all tensors in a carry object to device."""
    import dataclasses
    if dataclasses.is_dataclass(carry):
        updates = {}
        for field in dataclasses.fields(carry):
            value = getattr(carry, field.name)
            if isinstance(value, torch.Tensor):
                updates[field.name] = value.to(target_device)
            elif dataclasses.is_dataclass(value):
                updates[field.name] = move_carry_to_device(value, target_device)
            elif isinstance(value, dict):
                updates[field.name] = {k: v.to(target_device) if isinstance(v, torch.Tensor) else v 
                                        for k, v in value.items()}
            else:
                updates[field.name] = value
        return type(carry)(**updates)
    return carry


def replace_z_L_in_carry(carry, new_z_L):
    """Replace z_L in carry object with new tensor."""
    import dataclasses
    if dataclasses.is_dataclass(carry):
        # Recursively update inner_carry
        if hasattr(carry, 'inner_carry'):
            inner_carry = carry.inner_carry
            if hasattr(inner_carry, 'z_L'):
                # Create new inner_carry with replaced z_L
                inner_updates = {}
                for field in dataclasses.fields(inner_carry):
                    if field.name == 'z_L':
                        inner_updates['z_L'] = new_z_L
                    else:
                        inner_updates[field.name] = getattr(inner_carry, field.name)
                new_inner_carry = type(inner_carry)(**inner_updates)
                
                # Create new carry with new inner_carry
                outer_updates = {}
                for field in dataclasses.fields(carry):
                    if field.name == 'inner_carry':
                        outer_updates['inner_carry'] = new_inner_carry
                    else:
                        outer_updates[field.name] = getattr(carry, field.name)
                return type(carry)(**outer_updates)
    return carry


# ============================================================================
# Model Loading Functions
# ============================================================================
def load_trm_model(config_path: str, checkpoint_path: str, model_metadata: PuzzleDatasetMetadata):
    """Load TRM model from checkpoint."""
    import yaml
    from omegaconf import OmegaConf
    
    config_file = os.path.join(config_path, "all_config.yaml")
    with open(config_file, 'r') as f:
        config = OmegaConf.load(f)
    
    arch_dict = OmegaConf.to_container(config.arch, resolve=True)
    arch_name = arch_dict.pop('name')
    loss_config = arch_dict.pop('loss')
    loss_name = loss_config.pop('name')
    
    model_cfg = dict(
        **arch_dict,
        batch_size=1,
        vocab_size=model_metadata.vocab_size,
        seq_len=model_metadata.seq_len,
        num_puzzle_identifiers=model_metadata.num_puzzle_identifiers,
        causal=False
    )
    
    model_cls = load_model_class(arch_name)
    loss_head_cls = load_model_class(loss_name)
    
    with device:
        model = model_cls(model_cfg)
        model = loss_head_cls(model, **loss_config)
    
    model = model.to(device)
    
    print(f"Loading TRM checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("_orig_mod.", "")
        if isinstance(v, torch.Tensor):
            new_state_dict[new_key] = v.to(device)
        else:
            new_state_dict[new_key] = v
    
    puzzle_emb_name = "model.inner.puzzle_emb.weights"
    if puzzle_emb_name in new_state_dict:
        expected_shape = model.model.puzzle_emb.weights.shape
        if new_state_dict[puzzle_emb_name].shape != expected_shape:
            raise ValueError(f"Puzzle embedding mismatch: {new_state_dict[puzzle_emb_name].shape} vs {expected_shape}")
    
    model.load_state_dict(new_state_dict, strict=True)
    model = model.to(device)
    model.eval()
    return model, config


def load_sae_model(sae_checkpoint_path: str, depth: int = 16, d_model: int = 512, 
                   n_features: int = 4096, topk: int = 64):
    """Load trained SAE from checkpoint."""
    sae = SAE(
        d_model=d_model,
        depth=depth,
        n_features=n_features,
        topk=topk,
    ).to(device=device, dtype=DTYPE)
    
    print(f"Loading SAE checkpoint: {sae_checkpoint_path}")
    checkpoint = torch.load(sae_checkpoint_path, map_location=device, weights_only=False)
    
    if 'sae_state_dict' in checkpoint:
        sae.load_state_dict(checkpoint['sae_state_dict'])
    else:
        sae.load_state_dict(checkpoint)
    
    sae.eval()
    return sae


# ============================================================================
# Ablation Functions
# ============================================================================
def compute_feature_importance_for_incorrect(
    z_n: torch.Tensor,
    predictions: np.ndarray,
    labels: np.ndarray,
    method: str = "mean_activation"
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Compute feature importance scores specifically for incorrectly predicted examples.
    
    Args:
        z_n: Feature activations [B, D, L, M]
        predictions: Model predictions [B, L]
        labels: Ground truth labels [B, L]
        method: Importance computation method
            - "mean_activation": Mean activation across incorrect positions
            - "activation_diff": Difference between incorrect vs correct positions
    
    Returns:
        importance_scores: [M] tensor of importance per feature
        incorrect_mask: [B] boolean array indicating incorrect predictions
    """
    B, D, L, M = z_n.shape
    
    # Decode predictions and labels to identify incorrect examples
    incorrect_mask = np.zeros(B, dtype=bool)
    for b in range(B):
        pred_grid = decode_arc_grid(predictions[b])
        label_grid = decode_arc_grid(labels[b])
        if not np.array_equal(pred_grid, label_grid):
            incorrect_mask[b] = True
    
    incorrect_indices = np.where(incorrect_mask)[0]
    
    if len(incorrect_indices) == 0:
        print("Warning: No incorrect predictions found!")
        # Return uniform importance if no incorrect predictions
        return torch.ones(M, device=z_n.device) / M, incorrect_mask
    
    print(f"Found {len(incorrect_indices)} incorrect predictions out of {B}")
    
    # Extract features for incorrect examples
    z_n_incorrect = z_n[incorrect_indices]  # [N_incorrect, D, L, M]
    
    if method == "mean_activation":
        # Mean activation across all positions for incorrect examples
        importance = z_n_incorrect.mean(dim=(0, 1, 2))  # [M]
    
    elif method == "activation_diff":
        # Compare activation patterns between correct and incorrect
        correct_indices = np.where(~incorrect_mask)[0]
        if len(correct_indices) > 0:
            z_n_correct = z_n[correct_indices]
            incorrect_mean = z_n_incorrect.mean(dim=(0, 1, 2))
            correct_mean = z_n_correct.mean(dim=(0, 1, 2))
            importance = (incorrect_mean - correct_mean).abs()
        else:
            importance = z_n_incorrect.mean(dim=(0, 1, 2))
    
    else:
        raise ValueError(f"Unknown importance method: {method}")
    
    return importance, incorrect_mask


def ablate_features(z_n: torch.Tensor, feature_indices: torch.Tensor, 
                    example_indices: Optional[np.ndarray] = None) -> torch.Tensor:
    """
    Ablate (zero out) specific features in the activation tensor.
    
    Args:
        z_n: Feature activations [B, D, L, M]
        feature_indices: Indices of features to ablate [K]
        example_indices: Optional indices of examples to ablate (None = all)
    
    Returns:
        z_n_ablated: Ablated feature activations [B, D, L, M]
    """
    z_n_ablated = z_n.clone()
    
    if example_indices is not None:
        # Ablate only for specific examples
        z_n_ablated[example_indices, :, :, feature_indices] = 0
    else:
        # Ablate for all examples
        z_n_ablated[:, :, :, feature_indices] = 0
    
    return z_n_ablated


def run_inference_with_z_L_injection(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    z_L_to_inject: torch.Tensor,
    inject_at_step: int = -1,
    max_steps: int = 100
) -> Tuple[np.ndarray, float, Dict]:
    """
    Run model inference with z_L injection at a specific step.
    
    Args:
        model: TRM model
        batch: Input batch dictionary
        z_L_to_inject: z_L tensor to inject [B, L, H]
        inject_at_step: Step at which to inject z_L (-1 = last step before final)
        max_steps: Maximum inference steps
    
    Returns:
        predictions: Model predictions [B, L]
        loss: Final loss value
        metrics: Dictionary of metrics
    """
    with torch.inference_mode():
        carry = model.initial_carry(batch)
        carry = move_carry_to_device(carry, device)
        
        step = 0
        while step < max_steps:
            # Check if we should inject at this step
            if inject_at_step >= 0 and step == inject_at_step:
                carry = replace_z_L_in_carry(carry, z_L_to_inject)
            
            carry, loss, metrics, preds, all_finish = model(
                carry=carry, batch=batch, return_keys={"preds"}
            )
            step += 1
            
            if all_finish:
                break
        
        predictions = preds["preds"].cpu().float().numpy()
        loss_value = loss.item() if torch.is_tensor(loss) else loss
    
    return predictions, loss_value, {k: v.cpu().float().numpy() if torch.is_tensor(v) else v 
                                      for k, v in metrics.items()}


def run_inference_collect_trajectories(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    max_steps: int = 100
) -> Tuple[torch.Tensor, np.ndarray, float, Dict]:
    """
    Run inference and collect z_L trajectories at each step.
    
    Returns:
        z_L_trajectories: [B, num_steps, L, H] tensor of z_L at each step
        predictions: Final predictions [B, L]
        loss: Final loss value
        metrics: Dictionary of metrics
    """
    trajectories = []
    
    with torch.inference_mode():
        carry = model.initial_carry(batch)
        carry = move_carry_to_device(carry, device)
        
        # Collect initial z_L
        if hasattr(carry, 'inner_carry') and hasattr(carry.inner_carry, 'z_L'):
            trajectories.append(carry.inner_carry.z_L.detach().clone())
        
        step = 0
        while step < max_steps:
            carry, loss, metrics, preds, all_finish = model(
                carry=carry, batch=batch, return_keys={"preds"}
            )
            step += 1
            
            # Collect z_L after each step
            if hasattr(carry, 'inner_carry') and hasattr(carry.inner_carry, 'z_L'):
                trajectories.append(carry.inner_carry.z_L.detach().clone())
            
            if all_finish:
                break
        
        predictions = preds["preds"].cpu().float().numpy()
        loss_value = loss.item() if torch.is_tensor(loss) else loss
    
    # Stack trajectories: [num_steps, B, L, H] -> [B, num_steps, L, H]
    z_L_trajectories = torch.stack(trajectories, dim=0).permute(1, 0, 2, 3)
    
    return z_L_trajectories, predictions, loss_value, {
        k: v.cpu().float().numpy() if torch.is_tensor(v) else v for k, v in metrics.items()
    }


# ============================================================================
# Visualization Functions
# ============================================================================
def visualize_ablation_comparison(
    puzzle_name: str,
    puzzle_id: int,
    demo_examples: List,
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
    """
    num_demo = len(demo_examples)
    
    # Create figure with 4 rows: demos, test target, original pred, ablated pred
    fig = plt.figure(figsize=(14, (num_demo + 3) * 3))
    gs = gridspec.GridSpec(num_demo + 3, 3, width_ratios=[1, 1, 1.5])
    
    # Plot demo examples
    for ex_row, demo in enumerate(demo_examples):
        ax_in = fig.add_subplot(gs[ex_row, 0])
        ax_out = fig.add_subplot(gs[ex_row, 1])
        
        input_grid = np.array(demo['input'], dtype=np.uint8)
        output_grid = np.array(demo['output'], dtype=np.uint8)
        
        visualize_arc_grid(input_grid, ax_in, f"Demo {ex_row+1} - Input")
        visualize_arc_grid(output_grid, ax_out, f"Demo {ex_row+1} - Output")
        
        # Empty third column for demos
        ax_empty = fig.add_subplot(gs[ex_row, 2])
        ax_empty.axis('off')
    
    # Decode grids
    test_input_grid = decode_arc_grid(test_input)
    test_label_grid = decode_arc_grid(test_label)
    original_pred_grid = decode_arc_grid(original_pred)
    ablated_pred_grid = decode_arc_grid(ablated_pred)
    
    # Test target row
    target_row = num_demo
    ax_target_in = fig.add_subplot(gs[target_row, 0])
    ax_target_out = fig.add_subplot(gs[target_row, 1])
    ax_target_info = fig.add_subplot(gs[target_row, 2])
    
    visualize_arc_grid(test_input_grid, ax_target_in, "Test - Input")
    visualize_arc_grid(test_label_grid, ax_target_out, "Test - Target", highlight_color='blue')
    ax_target_info.axis('off')
    ax_target_info.text(0.5, 0.5, f"Ground Truth\nSize: {test_label_grid.shape}", 
                       ha='center', va='center', fontsize=12)
    
    # Original prediction row
    orig_row = num_demo + 1
    ax_orig_in = fig.add_subplot(gs[orig_row, 0])
    ax_orig_out = fig.add_subplot(gs[orig_row, 1])
    ax_orig_info = fig.add_subplot(gs[orig_row, 2])
    
    is_orig_correct = np.array_equal(test_label_grid, original_pred_grid)
    orig_status = "✓ CORRECT" if is_orig_correct else "✗ INCORRECT"
    orig_color = 'green' if is_orig_correct else 'red'
    
    visualize_arc_grid(test_input_grid, ax_orig_in, "Original - Input")
    visualize_arc_grid(original_pred_grid, ax_orig_out, f"Original Pred ({orig_status})",
                       highlight_color=orig_color)
    ax_orig_info.axis('off')
    ax_orig_info.text(0.5, 0.5, f"Original Prediction\nLoss: {original_loss:.4f}\n{orig_status}", 
                     ha='center', va='center', fontsize=12,
                     color=orig_color)
    
    # Ablated prediction row
    abl_row = num_demo + 2
    ax_abl_in = fig.add_subplot(gs[abl_row, 0])
    ax_abl_out = fig.add_subplot(gs[abl_row, 1])
    ax_abl_info = fig.add_subplot(gs[abl_row, 2])
    
    is_abl_correct = np.array_equal(test_label_grid, ablated_pred_grid)
    abl_status = "✓ CORRECT" if is_abl_correct else "✗ INCORRECT"
    abl_color = 'green' if is_abl_correct else 'red'
    
    visualize_arc_grid(test_input_grid, ax_abl_in, "Ablated - Input")
    visualize_arc_grid(ablated_pred_grid, ax_abl_out, f"Ablated Pred ({abl_status})",
                       highlight_color=abl_color)
    ax_abl_info.axis('off')
    
    loss_change = ablated_loss - original_loss
    loss_change_str = f"+{loss_change:.4f}" if loss_change > 0 else f"{loss_change:.4f}"
    
    ax_abl_info.text(0.5, 0.5, 
                    f"After Ablation\n"
                    f"Loss: {ablated_loss:.4f} ({loss_change_str})\n"
                    f"{abl_status}\n"
                    f"Ablated {len(ablated_features)} features", 
                    ha='center', va='center', fontsize=12,
                    color=abl_color)
    
    # Title
    changed_str = "CHANGED!" if is_orig_correct != is_abl_correct else "No change"
    plt.suptitle(
        f"Ablation Analysis: {puzzle_name} (ID: {puzzle_id})\n"
        f"Top-{len(ablated_features)} features ablated | Prediction: {changed_str}",
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save
    save_path = os.path.join(output_dir, f"ablation_{example_idx:03d}_{puzzle_name}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return is_orig_correct, is_abl_correct, original_loss, ablated_loss


def visualize_feature_importance(
    importance_scores: torch.Tensor,
    top_k: int,
    output_path: str,
    title: str = "Feature Importance for Incorrect Predictions"
):
    """Visualize feature importance distribution and top-k features."""
    importance = importance_scores.cpu().float().numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Histogram of all importance scores
    axes[0].hist(importance, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    axes[0].axvline(np.sort(importance)[-top_k], color='red', linestyle='--', 
                   label=f'Top-{top_k} threshold')
    axes[0].set_xlabel('Importance Score')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Distribution of Feature Importance')
    axes[0].legend()
    
    # 2. Top-k features bar chart
    top_indices = np.argsort(importance)[-top_k:][::-1]
    top_values = importance[top_indices]
    
    axes[1].barh(range(len(top_indices)), top_values, color='steelblue')
    axes[1].set_yticks(range(len(top_indices)))
    axes[1].set_yticklabels([f'F{i}' for i in top_indices])
    axes[1].set_xlabel('Importance Score')
    axes[1].set_title(f'Top-{top_k} Most Important Features')
    axes[1].invert_yaxis()
    
    # 3. Cumulative importance
    sorted_importance = np.sort(importance)[::-1]
    cumulative = np.cumsum(sorted_importance) / np.sum(sorted_importance)
    
    axes[2].plot(range(len(cumulative)), cumulative, color='steelblue')
    axes[2].axvline(top_k, color='red', linestyle='--', label=f'Top-{top_k}')
    axes[2].axhline(cumulative[top_k-1], color='red', linestyle=':', alpha=0.5)
    axes[2].set_xlabel('Number of Features')
    axes[2].set_ylabel('Cumulative Importance')
    axes[2].set_title('Cumulative Feature Importance')
    axes[2].legend()
    axes[2].text(top_k + 5, cumulative[top_k-1], f'{cumulative[top_k-1]:.2%}', fontsize=10)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_summary(
    results: List[Dict],
    output_path: str,
    topk_ablated: int
):
    """Visualize summary of ablation experiment."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Extract data
    orig_correct = sum(1 for r in results if r['original_correct'])
    orig_incorrect = len(results) - orig_correct
    abl_correct = sum(1 for r in results if r['ablated_correct'])
    abl_incorrect = len(results) - abl_correct
    
    # 1. Accuracy comparison
    ax = axes[0, 0]
    x = ['Original', 'After Ablation']
    correct = [orig_correct, abl_correct]
    incorrect = [orig_incorrect, abl_incorrect]
    
    bars1 = ax.bar(x, correct, label='Correct', color='green', alpha=0.7)
    bars2 = ax.bar(x, incorrect, bottom=correct, label='Incorrect', color='red', alpha=0.7)
    
    ax.set_ylabel('Number of Examples')
    ax.set_title(f'Prediction Accuracy (Top-{topk_ablated} Ablation)')
    ax.legend()
    
    for bar, c, i in zip(bars1, correct, incorrect):
        ax.text(bar.get_x() + bar.get_width()/2, c/2, f'{c}', ha='center', va='center', fontsize=12)
        ax.text(bar.get_x() + bar.get_width()/2, c + i/2, f'{i}', ha='center', va='center', fontsize=12)
    
    # 2. Loss change distribution
    ax = axes[0, 1]
    loss_changes = [r['ablated_loss'] - r['original_loss'] for r in results]
    ax.hist(loss_changes, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', label='No change')
    ax.axvline(np.mean(loss_changes), color='orange', linestyle='-', label=f'Mean: {np.mean(loss_changes):.4f}')
    ax.set_xlabel('Loss Change (Ablated - Original)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Loss Changes')
    ax.legend()
    
    # 3. Prediction change breakdown
    ax = axes[1, 0]
    categories = ['Stayed Correct', 'Became Correct', 'Stayed Incorrect', 'Became Incorrect']
    counts = [
        sum(1 for r in results if r['original_correct'] and r['ablated_correct']),
        sum(1 for r in results if not r['original_correct'] and r['ablated_correct']),
        sum(1 for r in results if not r['original_correct'] and not r['ablated_correct']),
        sum(1 for r in results if r['original_correct'] and not r['ablated_correct']),
    ]
    colors = ['green', 'lightgreen', 'salmon', 'red']
    
    wedges, texts, autotexts = ax.pie(counts, labels=categories, colors=colors, 
                                       autopct=lambda p: f'{p:.1f}%\n({int(p*len(results)/100)})',
                                       startangle=90)
    ax.set_title('Prediction Change After Ablation')
    
    # 4. Loss scatter plot
    ax = axes[1, 1]
    orig_losses = [r['original_loss'] for r in results]
    abl_losses = [r['ablated_loss'] for r in results]
    colors_scatter = ['green' if r['original_correct'] else 'red' for r in results]
    
    ax.scatter(orig_losses, abl_losses, c=colors_scatter, alpha=0.6)
    ax.plot([min(orig_losses + abl_losses), max(orig_losses + abl_losses)],
            [min(orig_losses + abl_losses), max(orig_losses + abl_losses)],
            'k--', label='No change')
    ax.set_xlabel('Original Loss')
    ax.set_ylabel('Ablated Loss')
    ax.set_title('Loss Comparison')
    ax.legend()
    
    plt.suptitle(f'Ablation Summary: Top-{topk_ablated} Features\n'
                f'Original Accuracy: {orig_correct}/{len(results)} ({100*orig_correct/len(results):.1f}%) → '
                f'Ablated Accuracy: {abl_correct}/{len(results)} ({100*abl_correct/len(results):.1f}%)',
                fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# Main Function
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='Ablation Analysis Visualization for ARC + TRM + SAE')
    parser.add_argument('--data_path', type=str, 
                        default='data/arc1concept-aug-0',
                        help='Path to test dataset (non-augmented)')
    parser.add_argument('--model_data_path', type=str,
                        default='data/arc1concept-aug-1000',
                        help='Path to model dataset (for puzzle embeddings)')
    parser.add_argument('--checkpoint', type=str,
                        default='ckpt/arc_v1_public/step_518071',
                        help='Path to TRM model checkpoint')
    parser.add_argument('--config_path', type=str,
                        default='ckpt/arc_v1_public',
                        help='Path to config directory')
    parser.add_argument('--sae_checkpoint', type=str,
                        default='weights/sae/best_val.pt',
                        help='Path to SAE checkpoint')
    parser.add_argument('--output_dir', type=str,
                        default='results-analysis-noaug/ablation_visualizations',
                        help='Output directory for visualizations')
    parser.add_argument('--num_examples', type=int, default=10,
                        help='Number of examples to analyze')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for inference')
    parser.add_argument('--topk_ablate', type=int, default=20,
                        help='Number of top features to ablate')
    parser.add_argument('--importance_method', type=str, default='mean_activation',
                        choices=['mean_activation', 'activation_diff'],
                        help='Method for computing feature importance')
    parser.add_argument('--only_incorrect', action='store_true',
                        help='Only visualize incorrect predictions')
    parser.add_argument('--sae_depth', type=int, default=16,
                        help='SAE depth parameter')
    parser.add_argument('--sae_d_model', type=int, default=512,
                        help='SAE d_model parameter')
    parser.add_argument('--sae_n_features', type=int, default=4096,
                        help='SAE n_features parameter')
    parser.add_argument('--sae_topk', type=int, default=64,
                        help='SAE topk parameter')
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*70)
    print("Ablation Analysis Visualization for ARC + TRM + SAE")
    print("="*70)
    print(f"Test data: {args.data_path}")
    print(f"Model data: {args.model_data_path}")
    print(f"TRM Checkpoint: {args.checkpoint}")
    print(f"SAE Checkpoint: {args.sae_checkpoint}")
    print(f"Top-k ablate: {args.topk_ablate}")
    print(f"Output: {args.output_dir}")
    print("="*70)
    
    # Load model metadata
    print("\nLoading model metadata...")
    with open(os.path.join(args.model_data_path, "train", "dataset.json"), 'r') as f:
        model_metadata = PuzzleDatasetMetadata(**json.load(f))
    print(f"  Model expects {model_metadata.num_puzzle_identifiers} puzzle identifiers")
    
    # Build identifier mapping
    print("\nBuilding identifier mapping...")
    identifier_mapping = build_identifier_mapping(args.model_data_path, args.data_path)
    print(f"  Mapped {len(identifier_mapping)} identifiers")
    
    # Load TRM model
    print("\nLoading TRM model...")
    model, config = load_trm_model(args.config_path, args.checkpoint, model_metadata)
    print("  TRM model loaded successfully!")
    
    # Load SAE model
    print("\nLoading SAE model...")
    sae = load_sae_model(
        args.sae_checkpoint,
        depth=args.sae_depth,
        d_model=args.sae_d_model,
        n_features=args.sae_n_features,
        topk=args.sae_topk
    )
    print("  SAE model loaded successfully!")
    
    # Load test data
    print("\nLoading puzzle data...")
    with open(os.path.join(args.data_path, "identifiers.json"), 'r') as f:
        identifiers_map = json.load(f)
    
    with open(os.path.join(args.data_path, "test_puzzles.json"), 'r') as f:
        test_puzzles = json.load(f)
    
    test_data_dir = os.path.join(args.data_path, "test")
    inputs = np.load(os.path.join(test_data_dir, "all__inputs.npy"))
    labels = np.load(os.path.join(test_data_dir, "all__labels.npy"))
    puzzle_ids_arr = np.load(os.path.join(test_data_dir, "all__puzzle_identifiers.npy"))
    puzzle_indices = np.load(os.path.join(test_data_dir, "all__puzzle_indices.npy"))
    
    print(f"  Test inputs: {inputs.shape}")
    print(f"  Test labels: {labels.shape}")
    
    # Select subset of examples
    num_examples = min(args.num_examples, len(inputs))
    batch_size = min(args.batch_size, num_examples)
    
    print(f"\nProcessing {num_examples} examples in batches of {batch_size}...")
    
    # Process in batches
    all_results = []
    all_z_n = []
    all_predictions = []
    all_labels_batch = []
    
    for batch_start in range(0, num_examples, batch_size):
        batch_end = min(batch_start + batch_size, num_examples)
        batch_indices = range(batch_start, batch_end)
        
        print(f"\nProcessing batch {batch_start//batch_size + 1}: examples {batch_start}-{batch_end-1}")
        
        # Prepare batch
        batch_inputs = torch.from_numpy(inputs[batch_indices].astype(np.int32)).to(device)
        batch_labels = torch.from_numpy(labels[batch_indices].astype(np.int32)).to(device)
        batch_puzzle_ids = torch.from_numpy(
            np.array([identifier_mapping.get(int(puzzle_ids_arr[i]), 0) for i in batch_indices], dtype=np.int32)
        ).to(device)
        
        batch = {
            "inputs": batch_inputs,
            "labels": batch_labels,
            "puzzle_identifiers": batch_puzzle_ids,
        }
        
        # Run inference and collect trajectories
        print("  Running inference and collecting z_L trajectories...")
        z_L_trajectories, predictions, loss, metrics = run_inference_collect_trajectories(
            model, batch, max_steps=100
        )
        
        print(f"  z_L trajectories shape: {z_L_trajectories.shape}")
        print(f"  Original loss: {loss:.4f}")
        
        # Use final step z_L for SAE (shape: [B, L, H])
        # Need to reshape for SAE: [B, D=num_steps, L, H]
        # For simplicity, use the full trajectory
        z_L_for_sae = z_L_trajectories.to(dtype=DTYPE)  # [B, D, L, H]
        
        # Encode through SAE
        print("  Encoding through SAE...")
        with torch.no_grad():
            z_n = sae.encode(z_L_for_sae)  # [B, D, L, M]
        
        print(f"  Feature activations shape: {z_n.shape}")
        
        all_z_n.append(z_n.cpu())
        all_predictions.append(predictions)
        all_labels_batch.append(labels[batch_indices])
    
    # Concatenate all batches
    all_z_n = torch.cat(all_z_n, dim=0).to(device).to(dtype=DTYPE)
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels_batch = np.concatenate(all_labels_batch, axis=0)
    
    print(f"\nTotal feature activations shape: {all_z_n.shape}")
    
    # Compute feature importance for incorrect predictions
    print("\nComputing feature importance for incorrect predictions...")
    importance_scores, incorrect_mask = compute_feature_importance_for_incorrect(
        all_z_n, all_predictions, all_labels_batch, method=args.importance_method
    )
    
    # Get top-k features to ablate
    topk_indices = torch.topk(importance_scores, args.topk_ablate).indices
    print(f"Top-{args.topk_ablate} features to ablate: {topk_indices.tolist()}")
    
    # Visualize feature importance
    importance_path = os.path.join(args.output_dir, "feature_importance.png")
    visualize_feature_importance(importance_scores, args.topk_ablate, importance_path)
    print(f"Saved feature importance visualization to {importance_path}")
    
    # Run ablation experiment
    print("\n" + "="*70)
    print("Running Ablation Experiment")
    print("="*70)
    
    results = []
    
    for i in range(num_examples):
        if i + 1 < len(puzzle_indices):
            puzzle_idx = int(puzzle_indices[i + 1])
            if puzzle_idx < len(identifiers_map):
                puzzle_name = identifiers_map[puzzle_idx]
            else:
                puzzle_name = f"puzzle_{i}"
        else:
            puzzle_name = f"puzzle_{i}"
        
        # Skip if only_incorrect and this is correct
        if args.only_incorrect and not incorrect_mask[i]:
            continue
        
        print(f"\nExample {i+1}/{num_examples}: {puzzle_name}")
        
        # Get original prediction
        original_pred = all_predictions[i]
        original_label = all_labels_batch[i]
        
        # Ablate features for this example
        z_n_single = all_z_n[i:i+1]  # [1, D, L, M]
        z_n_ablated = ablate_features(z_n_single, topk_indices)
        
        # Reconstruct z_L from ablated features
        with torch.no_grad():
            z_L_reconstructed = sae.decode(z_n_ablated)  # [1, D, L, H]
        
        # Prepare single example batch
        single_batch = {
            "inputs": torch.from_numpy(inputs[i:i+1].astype(np.int32)).to(device),
            "labels": torch.from_numpy(labels[i:i+1].astype(np.int32)).to(device),
            "puzzle_identifiers": torch.tensor(
                [[identifier_mapping.get(int(puzzle_ids_arr[i]), 0)]], dtype=torch.int32
            ).to(device),
        }
        
        # Run original inference (for comparison loss)
        _, orig_loss, _ = run_inference_with_z_L_injection(
            model, single_batch, None, inject_at_step=-1
        )
        
        # Run inference with z_L injection
        # Use the last timestep of reconstructed z_L
        z_L_to_inject = z_L_reconstructed[:, -1, :, :].float()  # [1, L, H]
        
        # Inject at second-to-last step (step before final)
        ablated_pred, ablated_loss, _ = run_inference_with_z_L_injection(
            model, single_batch, z_L_to_inject, inject_at_step=10  # Inject mid-way
        )
        
        print(f"  Original loss: {orig_loss:.4f}, Ablated loss: {ablated_loss:.4f}")
        
        # Get demo examples
        if "|||" in puzzle_name:
            base_puzzle_name = puzzle_name.split("|||")[0]
        else:
            base_puzzle_name = puzzle_name
        demo_examples = test_puzzles.get(base_puzzle_name, {}).get("train", [])
        
        # Visualize comparison
        orig_correct, abl_correct, orig_loss_val, abl_loss_val = visualize_ablation_comparison(
            puzzle_name=puzzle_name,
            puzzle_id=puzzle_idx if i + 1 < len(puzzle_indices) else i,
            demo_examples=demo_examples,
            test_input=inputs[i],
            test_label=labels[i],
            original_pred=original_pred,
            ablated_pred=ablated_pred[0],
            original_loss=orig_loss,
            ablated_loss=ablated_loss,
            ablated_features=topk_indices.tolist(),
            output_dir=args.output_dir,
            example_idx=i,
        )
        
        results.append({
            'puzzle_name': puzzle_name,
            'original_correct': orig_correct,
            'ablated_correct': abl_correct,
            'original_loss': orig_loss,
            'ablated_loss': ablated_loss,
        })
        
        status = "Correct→Correct" if orig_correct and abl_correct else \
                 "Correct→Incorrect" if orig_correct else \
                 "Incorrect→Correct" if abl_correct else "Incorrect→Incorrect"
        print(f"  {status}")
    
    # Summary visualization
    if len(results) > 0:
        summary_path = os.path.join(args.output_dir, "ablation_summary.png")
        visualize_summary(results, summary_path, args.topk_ablate)
        print(f"\nSaved summary visualization to {summary_path}")
    
    # Print final summary
    print("\n" + "="*70)
    print("ABLATION EXPERIMENT SUMMARY")
    print("="*70)
    print(f"Total examples analyzed: {len(results)}")
    
    orig_correct_count = sum(1 for r in results if r['original_correct'])
    abl_correct_count = sum(1 for r in results if r['ablated_correct'])
    
    print(f"Original accuracy: {orig_correct_count}/{len(results)} ({100*orig_correct_count/len(results):.1f}%)")
    print(f"Ablated accuracy:  {abl_correct_count}/{len(results)} ({100*abl_correct_count/len(results):.1f}%)")
    
    print(f"\nPrediction changes:")
    print(f"  Stayed correct:    {sum(1 for r in results if r['original_correct'] and r['ablated_correct'])}")
    print(f"  Became correct:    {sum(1 for r in results if not r['original_correct'] and r['ablated_correct'])}")
    print(f"  Stayed incorrect:  {sum(1 for r in results if not r['original_correct'] and not r['ablated_correct'])}")
    print(f"  Became incorrect:  {sum(1 for r in results if r['original_correct'] and not r['ablated_correct'])}")
    
    avg_orig_loss = np.mean([r['original_loss'] for r in results])
    avg_abl_loss = np.mean([r['ablated_loss'] for r in results])
    print(f"\nAverage loss:")
    print(f"  Original: {avg_orig_loss:.4f}")
    print(f"  Ablated:  {avg_abl_loss:.4f}")
    print(f"  Change:   {avg_abl_loss - avg_orig_loss:+.4f}")
    
    print(f"\nVisualizations saved to: {args.output_dir}")
    print("="*70)
    
    # Save results to JSON
    results_path = os.path.join(args.output_dir, "ablation_results.json")
    with open(results_path, 'w') as f:
        json.dump({
            'config': {
                'topk_ablate': args.topk_ablate,
                'importance_method': args.importance_method,
                'num_examples': len(results),
            },
            'summary': {
                'original_accuracy': orig_correct_count / len(results),
                'ablated_accuracy': abl_correct_count / len(results),
                'avg_original_loss': float(avg_orig_loss),
                'avg_ablated_loss': float(avg_abl_loss),
            },
            'ablated_features': topk_indices.tolist(),
            'results': results,
        }, f, indent=2)
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
