"""
Visualize SAE features across iterations.

This script creates visualizations showing:
- Top row: Test input, ground truth, and model prediction
- Bottom section: SAE feature activation maps for all 16 iterations
  - Each iteration shown as a 64x64 grid (4096 features)
  - Each pixel intensity = aggregated activation of that feature across all spatial positions

Usage:
    python visualize_sae_features.py \
        arch=trm \
        load_checkpoint=weights/best.pt \
        data_paths=[arc_v1_public] \
        sae_checkpoint=weights/sae_fix/best_val.pt \
        puzzle_ids=[0,1,2] \
        output_dir=sae_visualizations
"""

from typing import Optional, List, Dict, Any
import os
import json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
import tqdm
import hydra
import pydantic
from omegaconf import DictConfig

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from utils.functions import load_model_class, get_model_source_path
from viz_utils import decode_arc_grid, visualize_grid, compute_diff, ARC_COLORS

# Global dtype configuration
DTYPE = torch.bfloat16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Attn(nn.Module):
    """Multi-head attention for TSAE"""
    def __init__(self, d_model, n_heads, is_causal):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.is_causal = is_causal
        
        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.o_proj = nn.Linear(self.d_model, self.d_model, bias=False)

    def forward(self, zL):
        batch_size, seq_len, d_model = zL.shape
        q = self.q_proj(zL).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(zL).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(zL).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=self.is_causal)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.o_proj(out)


class SAE(nn.Module):
    """
    Simple SAE without attention (copied from sae_fix.py for standalone use)
    zL: [B, D, L, H] (B=batch, D=depth, L=seq len, H=d_model)
    """
    def __init__(
        self,
        d_model: int = 512,
        depth: int = 16,
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

    def encode(self, zL):
        """
        Encode hidden states to sparse features.
        
        Args:
            zL: [B, D, L, H] or [B, L, H]
        
        Returns:
            z_n: sparse feature activations with same batch dims + [M]
        """
        original_shape = zL.shape
        H = self.d_model
        
        # Flatten to [N, H]
        x = zL.reshape(-1, H)
        
        # Encoder: x -> logits -> ReLU -> TopK
        logits = F.linear(x - self.bias_pre[None, :], self.dictionary_enc, self.bias_enc)
        z_n_dense = F.relu(logits)
        z_n = self.topk_activation(z_n_dense)
        
        # Reshape back
        new_shape = original_shape[:-1] + (self.n_features,)
        return z_n.view(*new_shape)

    def forward(self, zL, mask=None):
        """Full forward pass with reconstruction"""
        B, D, L, H = zL.shape
        
        # Encode
        z_n = self.encode(zL)  # [B, D, L, M]
        
        # Decode
        z_n_flat = z_n.reshape(-1, self.n_features)
        x_tgt_flat = F.linear(z_n_flat, self.dictionary_dec) + self.bias_pre[None, :]
        x_tgt = x_tgt_flat.view(B, D, L, H)
        
        return {
            "z_n": z_n,
            "x_tgt": x_tgt,
        }


class TSAE(nn.Module):
    """
    Transformer SAE with spatial and depth attention (matches tsae_fix.py)
    zL: [B, D, L, H] (B=batch, D=depth, L=seq len, H=d_model)
    """
    def __init__(
        self,
        d_model: int = 512,
        depth: int = 16,
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

        # Spatial attention (L direction)
        self.attn_l = Attn(H, n_heads, is_causal=False)
        self.norm_l = nn.LayerNorm(H)

        # Depth attention (D direction, causal)
        self.attn_d = Attn(H, n_heads, is_causal=True)
        self.norm_d = nn.LayerNorm(H)

        # SAE dictionary: H -> M (encoder), M -> H (decoder) - matches tsae_fix.py
        self.dictionary_enc = nn.Parameter(torch.randn(n_features, H, dtype=DTYPE) * (2.0 / (H + n_features)) ** 0.5)
        self.dictionary_dec = nn.Parameter(torch.randn(H, n_features, dtype=DTYPE) * (2.0 / (n_features + H)) ** 0.5)
        self.bias_pre = nn.Parameter(torch.zeros(H, dtype=DTYPE))
        self.bias_enc = nn.Parameter(torch.zeros(n_features, dtype=DTYPE))
        
        # Query token
        self.query_token = nn.Parameter(torch.zeros(H, dtype=DTYPE))

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

    def forward(self, zL, mask=None):
        """
        Forward pass returning z_n features (matches tsae_fix.py logic).
        """
        from einops import rearrange
        
        B, D, L, H = zL.shape
        
        # L-axis self-attention with query token shift
        x = torch.cat([self.query_token[None, None, None, :].expand(B, 1, L, H), zL[:, :-1, :, :]], dim=1)
        x = x.view(B * D, L, H)
        x = x + self.attn_l(self.norm_l(x))
        x = x.view(B, D, L, H)

        # D-axis causal self-attention
        x = rearrange(x, 'b d l h -> (b l) d h')
        x = x + self.attn_d(self.norm_d(x))
        x_prior = rearrange(x, '(b l) d h -> b d l h', b=B, l=L)

        # Encode residual to sparse features
        N = B * D * L
        x_src = (zL - x_prior).detach().reshape(N, H)
        
        # encoder: x -> logits -> ReLU -> TopK
        logits = F.linear(x_src - self.bias_pre[None, :], self.dictionary_enc, self.bias_enc)
        z_n_dense = F.relu(logits)
        z_n_flat = self.topk_activation(z_n_dense)
        
        z_n = z_n_flat.view(B, D, L, self.n_features)
        
        return {
            "z_n": z_n,
        }


class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str


class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str
    loss: LossConfig


class VisConfig(pydantic.BaseModel):
    # Model config
    arch: ArchConfig
    data_paths: List[str]
    data_paths_test: List[str] = []
    global_batch_size: int = 1
    
    # Checkpoints
    load_checkpoint: Optional[str] = None
    sae_checkpoint: Optional[str] = None
    
    # Visualization options
    puzzle_ids: Optional[List[int]] = None
    num_visualize: int = 10
    output_dir: str = "sae_visualizations"
    top_m_features: int = 50  # Number of top features to show in heatmap
    sort_by_iteration: int = 1  # Which iteration to sort features by (1-16, or 0 for total)
    answer_only: bool = True  # If True, only consider answer tokens (where label != -100)

    # SAE model type: 'sae', 'sae_fix', 'tsae', 'tsae_fix'
    sae_model_type: str = "sae_fix"
    
    # SAE config
    sae_depth: int = 16
    sae_d_model: int = 512
    sae_n_features: int = 4096
    sae_topk: int = 64
    
    seed: int = 0


def create_dataloader(config: VisConfig, split: str):
    """Create dataloader"""
    dataset = PuzzleDataset(PuzzleDatasetConfig(
        seed=config.seed,
        dataset_paths=config.data_paths_test if len(config.data_paths_test) > 0 and split == "test" else config.data_paths,
        rank=0,
        num_replicas=1,
        test_set_mode=True,
        epochs_per_iter=1,
        global_batch_size=config.global_batch_size
    ), split=split)
    
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
        prefetch_factor=2,
        pin_memory=True,
        persistent_workers=True
    )
    return dataloader, dataset.metadata


def create_model(config: VisConfig, metadata: PuzzleDatasetMetadata):
    """Create TRM model"""
    # Build model config from arch config
    extra_config = config.arch.__pydantic_extra__ if config.arch.__pydantic_extra__ else {}
    model_cfg = dict(
        **extra_config,
        batch_size=config.global_batch_size,
        vocab_size=metadata.vocab_size,
        seq_len=metadata.seq_len,
        num_puzzle_identifiers=metadata.num_puzzle_identifiers,
        causal=False
    )

    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)
    
    loss_extra = config.arch.loss.__pydantic_extra__ if config.arch.loss.__pydantic_extra__ else {}

    # Disable compilation for visualization (easier to inspect carry)
    os.environ["DISABLE_COMPILE"] = "1"
    
    with device:
        model = model_cls(model_cfg)
        model = loss_head_cls(model, **loss_extra)
        
        # Load checkpoint
        if config.load_checkpoint is not None:
            print(f"Loading TRM checkpoint: {config.load_checkpoint}")
            state_dict = torch.load(config.load_checkpoint, map_location=device, weights_only=False)
            
            # Handle compiled model state dict (remove _orig_mod prefix if present)
            new_state_dict = {}
            for k, v in state_dict.items():
                # Remove _orig_mod prefix if present
                new_key = k.replace("_orig_mod.", "")
                new_state_dict[new_key] = v
            
            # Handle puzzle embedding resize
            puzzle_emb_name = "model.inner.puzzle_emb.weights"
            try:
                expected_shape = model.model.puzzle_emb.weights.shape
                if puzzle_emb_name in new_state_dict:
                    puzzle_emb = new_state_dict[puzzle_emb_name]
                    if puzzle_emb.shape != expected_shape:
                        print(f"Resizing puzzle embedding: {puzzle_emb.shape} -> {expected_shape}")
                        new_state_dict[puzzle_emb_name] = torch.mean(puzzle_emb, dim=0, keepdim=True).expand(expected_shape).contiguous()
            except Exception as e:
                print(f"Warning: Could not check puzzle embedding: {e}")
            
            model.load_state_dict(new_state_dict, assign=True)
            print(f"  Loaded checkpoint successfully")

    return model


def load_sae(config: VisConfig):
    """Load SAE model based on sae_model_type"""
    model_type = config.sae_model_type.lower()
    
    # Determine checkpoint path if not explicitly set
    sae_checkpoint = config.sae_checkpoint
    if sae_checkpoint is None or sae_checkpoint == "":
        # Default path based on model type
        sae_checkpoint = f"weights/{model_type}/best_val.pt"
    
    print(f"SAE Model Type: {model_type}")
    print(f"SAE Checkpoint: {sae_checkpoint}")
    
    # Create appropriate model based on type
    if model_type in ['sae', 'sae_fix']:
        sae = SAE(
            d_model=config.sae_d_model,
            depth=config.sae_depth,
            n_features=config.sae_n_features,
            topk=config.sae_topk,
        ).to(device=device, dtype=DTYPE)
    elif model_type in ['tsae', 'tsae_fix']:
        sae = TSAE(
            d_model=config.sae_d_model,
            depth=config.sae_depth,
            n_features=config.sae_n_features,
            topk=config.sae_topk,
        ).to(device=device, dtype=DTYPE)
    else:
        raise ValueError(f"Unknown SAE model type: {model_type}. Choose from: sae, sae_fix, tsae, tsae_fix")
    
    # Load checkpoint
    if sae_checkpoint is not None and os.path.exists(sae_checkpoint):
        print(f"Loading checkpoint from: {sae_checkpoint}")
        ckpt = torch.load(sae_checkpoint, map_location=device, weights_only=False)
        if 'sae_state_dict' in ckpt:
            sae.load_state_dict(ckpt['sae_state_dict'])
        else:
            sae.load_state_dict(ckpt)
        print(f"  Loaded successfully!")
    else:
        print(f"  Warning: Checkpoint not found at {sae_checkpoint}, using random weights")
    
    sae.eval()
    return sae


def load_visualization_context(config: VisConfig) -> Dict:
    """Load identifiers and test_puzzles for visualization context."""
    data_paths = config.data_paths_test if len(config.data_paths_test) > 0 else config.data_paths
    
    context = {
        'identifiers_map': [],
        'test_puzzles': {}
    }
    
    for data_path in data_paths:
        # Load identifiers
        identifiers_path = os.path.join(data_path, "identifiers.json")
        if os.path.exists(identifiers_path):
            with open(identifiers_path, 'r') as f:
                identifiers = json.load(f)
                if not context['identifiers_map']:
                    context['identifiers_map'] = identifiers
                else:
                    context['identifiers_map'].extend(identifiers)
        
        # Load test puzzles
        test_puzzles_path = os.path.join(data_path, "test_puzzles.json")
        if os.path.exists(test_puzzles_path):
            with open(test_puzzles_path, 'r') as f:
                puzzles = json.load(f)
                context['test_puzzles'].update(puzzles)
    
    return context


def collect_trajectories(model, batch, max_steps: int = 16):
    """
    Run inference and collect z_L trajectories at each step.
    
    Returns:
        z_L_trajectory: [B, D, L, H] where D = number of iterations
        predictions: final model predictions
    """
    with torch.no_grad():
        with device:
            carry = model.initial_carry(batch)
        
        trajectories = []
        preds = None
        
        def get_z_L(carry_obj):
            """Extract z_L from carry, handling different carry structures."""
            # Try inner_carry.z_L (ACTLossHead wrapper)
            if hasattr(carry_obj, 'inner_carry'):
                inner = carry_obj.inner_carry
                if hasattr(inner, 'z_L'):
                    return inner.z_L
            # Try direct z_L (raw model)
            if hasattr(carry_obj, 'z_L'):
                return carry_obj.z_L
            return None
        
        for step in range(max_steps):
            # Save z_L before each step
            z_L_state = get_z_L(carry)
            if z_L_state is not None:
                trajectories.append(z_L_state.detach().clone())
            else:
                if step == 0:
                    print(f"Warning: Could not find z_L in carry. Carry type: {type(carry)}")
                    if hasattr(carry, '__dict__'):
                        print(f"  Carry attrs: {list(carry.__dict__.keys())}")
                    if hasattr(carry, 'inner_carry') and hasattr(carry.inner_carry, '__dict__'):
                        print(f"  Inner carry attrs: {list(carry.inner_carry.__dict__.keys())}")
            
            carry, loss, metrics, preds, all_finish = model(
                carry=carry, batch=batch, return_keys={'preds'}
            )
            
            if all_finish:
                # Save final state
                z_L_state = get_z_L(carry)
                if z_L_state is not None:
                    trajectories.append(z_L_state.detach().clone())
                break
        
        if len(trajectories) == 0:
            raise RuntimeError("No z_L trajectories collected! Check model carry structure.")
        
        # Stack trajectories: [B, D, L, H]
        # Skip first (initial) state, keep D iterations
        if len(trajectories) > 1:
            z_L = torch.stack(trajectories[1:], dim=1)
        else:
            z_L = trajectories[0].unsqueeze(1)
        
        print(f"  Collected {z_L.shape[1]} iterations (trajectory shape: {z_L.shape})")
        
        # Pad to exactly 16 iterations if needed
        if z_L.shape[1] < 16:
            pad_size = 16 - z_L.shape[1]
            z_L = F.pad(z_L, (0, 0, 0, 0, 0, pad_size), mode='replicate')
        elif z_L.shape[1] > 16:
            z_L = z_L[:, :16]
        
        return z_L, preds


IGNORE_LABEL_ID = -100

def visualize_sae_features(
    test_input: np.ndarray,
    test_label: np.ndarray,
    prediction: np.ndarray,
    sae_features: np.ndarray,  # [D=16, L, M=4096]
    puzzle_name: str,
    puzzle_id: int,
    output_path: str,
    top_m: int = 50,
    sort_by_iteration: int = 1,  # 1-16 for specific iteration, 0 for total across all
    answer_only: bool = True,  # If True, only consider answer tokens (where label != -100)
):
    """
    Create visualization with test/prediction on top and SAE feature spatial heatmaps below.

    Each row shows one feature's 30×30 spatial activation pattern across 16 iterations.
    Features are sorted by activation at the specified iteration (1-16) or total (0).
    If answer_only=True, only considers tokens where the label is not IGNORE_LABEL_ID (-100).
    
    Args:
        test_input: tokenized test input [seq_len]
        test_label: tokenized test label [seq_len]
        prediction: model prediction [seq_len]
        sae_features: SAE feature activations [D=16, L, M=4096]
        puzzle_name: name of the puzzle
        puzzle_id: puzzle identifier
        output_path: path to save figure
        top_m: number of top features to show
        aggregation: how to aggregate for selecting top features
    """
    D, L, M = sae_features.shape
    assert M == 4096, f"Expected 4096 features, got {M}"
    
    # Decode grids
    input_grid = decode_arc_grid(test_input)
    label_grid = decode_arc_grid(test_label)
    pred_grid = decode_arc_grid(prediction)
    
    is_correct = np.array_equal(label_grid, pred_grid)
    diff_mask, num_diff = compute_diff(label_grid, pred_grid)
    
    # Create answer mask: True for tokens where label != IGNORE_LABEL_ID
    answer_mask = (test_label != IGNORE_LABEL_ID)  # [L] or [900]
    if len(answer_mask) > L:
        answer_mask = answer_mask[:L]
    elif len(answer_mask) < L:
        answer_mask = np.pad(answer_mask, (0, L - len(answer_mask)), constant_values=False)
    
    num_answer_tokens = answer_mask.sum()
    print(f"    Answer tokens: {num_answer_tokens}/{L} ({100*num_answer_tokens/L:.1f}%)")
    
    # Reshape to spatial: [D, 30, 30, M]
    if L >= 900:
        spatial_features = sae_features[:, :900, :].reshape(D, 30, 30, M)
        answer_mask_spatial = answer_mask[:900].reshape(30, 30)
    else:
        padded = np.zeros((D, 900, M))
        padded[:, :L, :] = sae_features
        spatial_features = padded.reshape(D, 30, 30, M)
        mask_padded = np.zeros(900, dtype=bool)
        mask_padded[:L] = answer_mask
        answer_mask_spatial = mask_padded.reshape(30, 30)
    
    # Create masked features for feature selection (only answer tokens)
    if answer_only and num_answer_tokens > 0:
        # Mask out non-answer tokens for feature selection
        sae_features_masked = sae_features * answer_mask[None, :, None]  # [D, L, M]
        feature_total = sae_features_masked.sum(axis=(0, 1))  # [M]
        selection_desc = "answer tokens only"
    else:
        feature_total = sae_features.sum(axis=(0, 1))  # [M]
        selection_desc = "all tokens"

    # Get indices of features that are actually active (non-zero)
    active_feature_mask = feature_total > 0
    active_feature_indices = np.where(active_feature_mask)[0]

    # Sort active features based on sort_by_iteration
    if sort_by_iteration == 0:
        # Sort by total activation across all iterations
        sort_values = feature_total[active_feature_indices]
        sort_desc = f"total activation ({selection_desc})"
    else:
        # Sort by activation at specific iteration (1-indexed -> 0-indexed)
        iter_idx = min(sort_by_iteration - 1, D - 1)  # Clamp to valid range
        # Sum over spatial dimension for the specific iteration
        if answer_only and num_answer_tokens > 0:
            feature_at_iter = (sae_features[iter_idx, :, :] * answer_mask[:, None]).sum(axis=0)  # [M]
        else:
            feature_at_iter = sae_features[iter_idx, :, :].sum(axis=0)  # [M]
        sort_values = feature_at_iter[active_feature_indices]
        sort_desc = f"iteration {sort_by_iteration} ({selection_desc})"
    
    sorted_order = np.argsort(sort_values)[::-1]
    active_feature_indices = active_feature_indices[sorted_order]

    # Limit to top_m (or fewer if not enough active features)
    num_to_show = min(top_m, len(active_feature_indices))
    top_m_indices = active_feature_indices[:num_to_show]

    print(f"    Active features: {len(active_feature_indices)} (TopK sparse), showing top {num_to_show} by {sort_desc}")
    
    # Handle edge case: no active features
    if num_to_show == 0:
        print(f"    Warning: No active features found!")
        return {
            'puzzle_name': puzzle_name,
            'puzzle_id': puzzle_id,
            'is_correct': is_correct,
            'num_diff': num_diff,
            'total_features_activated': 0,
            'features_shown': 0,
            'top_m_features': [],
        }
    
    # Create figure
    # Top row: 3 panels (input, label, prediction)
    # Below: num_to_show rows × 16 columns of 30×30 heatmaps
    n_cols = D  # 16 iterations
    n_rows = num_to_show  # Only active features
    
    # Calculate figure size
    cell_size = 1.0  # size per heatmap cell
    fig_width = n_cols * cell_size + 4  # extra for labels
    fig_height = n_rows * cell_size + 4  # extra for top row and titles
    
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # Use GridSpec: top row for grids, remaining rows for feature heatmaps
    gs = gridspec.GridSpec(n_rows + 1, n_cols + 1, figure=fig, 
                           height_ratios=[2] + [1]*n_rows,
                           width_ratios=[0.8] + [1]*n_cols,
                           hspace=0.1, wspace=0.1)
    
    # Top row: Test input, Ground truth, Prediction (spanning columns)
    cols_per_grid = max(1, n_cols // 3)
    ax_input = fig.add_subplot(gs[0, 1:1+cols_per_grid])
    ax_label = fig.add_subplot(gs[0, 1+cols_per_grid:1+2*cols_per_grid])
    ax_pred = fig.add_subplot(gs[0, 1+2*cols_per_grid:])
    
    visualize_grid(input_grid, ax_input, "Test Input")
    visualize_grid(label_grid, ax_label, "Ground Truth", highlight_color='#0074D9')
    
    status = "✓ CORRECT" if is_correct else "✗ INCORRECT"
    status_color = '#2ECC40' if is_correct else '#FF4136'
    visualize_grid(pred_grid, ax_pred, f"Prediction ({status})", 
                   highlight_color=status_color,
                   show_diff=diff_mask if not is_correct else None)
    
    # Find global min/max for consistent color scale
    all_activations = []
    for feat_rank, feat_idx in enumerate(top_m_indices):
        for iter_idx in range(D):
            all_activations.append(spatial_features[iter_idx, :, :, feat_idx])
    all_activations = np.array(all_activations)
    vmin = 0
    vmax = np.percentile(all_activations[all_activations > 0], 99) if (all_activations > 0).any() else 1.0
    
    # Draw feature rows
    for feat_rank, feat_idx in enumerate(top_m_indices):
        row = feat_rank + 1  # +1 because row 0 is the top grids
        
        # Feature label on the left
        ax_label_feat = fig.add_subplot(gs[row, 0])
        ax_label_feat.axis('off')
        ax_label_feat.text(0.9, 0.5, f'F{feat_idx}', ha='right', va='center',
                          fontsize=8, fontweight='bold', transform=ax_label_feat.transAxes)
        
        # 16 iteration heatmaps
        for iter_idx in range(D):
            col = iter_idx + 1  # +1 because col 0 is labels
            ax = fig.add_subplot(gs[row, col])

            # Get 30×30 spatial activation for this feature at this iteration
            feat_spatial = spatial_features[iter_idx, :, :, feat_idx].copy()  # [30, 30]
            
            # Mask non-answer tokens if answer_only is enabled
            if answer_only and num_answer_tokens > 0:
                # Set non-answer tokens to NaN (will appear as gray/transparent)
                feat_spatial_masked = np.where(answer_mask_spatial, feat_spatial, np.nan)
            else:
                feat_spatial_masked = feat_spatial

            im = ax.imshow(feat_spatial_masked, cmap='hot', vmin=vmin, vmax=vmax, aspect='equal')
            ax.set_xticks([])
            ax.set_yticks([])

            # Add border if feature is active at this iteration (in answer region)
            if np.nansum(feat_spatial_masked) > 0:
                for spine in ax.spines.values():
                    spine.set_edgecolor('orange')
                    spine.set_linewidth(0.5)
            else:
                for spine in ax.spines.values():
                    spine.set_edgecolor('gray')
                    spine.set_linewidth(0.2)
            
            # Add iteration labels on top row of features
            if feat_rank == 0:
                ax.set_title(f'{iter_idx+1}', fontsize=8)
    
    # Title
    title = f"SAE Features (30×30 spatial): {puzzle_name} (ID: {puzzle_id})"
    if not is_correct:
        title += f" [INCORRECT]"
    answer_info = f" | Answer tokens: {num_answer_tokens}" if answer_only else ""
    title += f"\nShowing {num_to_show} active features (TopK sparse) | Rows=Features | Cols=Iterations 1-16{answer_info}"
    plt.suptitle(title, fontsize=11, fontweight='bold', y=0.99)
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.3])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Activation', fontsize=9)
    
    # Save
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return {
        'puzzle_name': puzzle_name,
        'puzzle_id': puzzle_id,
        'is_correct': is_correct,
        'num_diff': num_diff,
        'total_features_activated': len(active_feature_indices),
        'features_shown': num_to_show,
        'top_m_features': top_m_indices.tolist(),
    }


def visualize_sae_features_spatial(
    sae_features: np.ndarray,  # [D=16, L, M=4096]
    puzzle_name: str,
    puzzle_id: int,
    output_dir: str,
    selected_iters: List[int] = None,  # Which iterations to visualize (default: all)
):
    """
    Create detailed visualization showing spatial activation for each feature.
    This creates a large (64×64) × (30×30) image per iteration.
    
    WARNING: This creates very large images (~40MB each)!
    """
    D, L, M = sae_features.shape
    grid_size = 30  # ARC grid max size
    
    if selected_iters is None:
        selected_iters = list(range(D))
    
    os.makedirs(output_dir, exist_ok=True)
    
    # For each iteration, create a montage of all 4096 features
    for iter_idx in selected_iters:
        if iter_idx >= D:
            continue
            
        # sae_features[iter_idx]: [L, 4096]
        # Reshape L to spatial grid (assume 30x30 max, pad if needed)
        iter_features = sae_features[iter_idx]  # [L, 4096]
        
        # Reshape to spatial: [30, 30, 4096]
        if L >= 900:
            spatial_features = iter_features[:900].reshape(30, 30, 4096)
        else:
            # Pad
            padded = np.zeros((900, 4096))
            padded[:L] = iter_features
            spatial_features = padded.reshape(30, 30, 4096)
        
        # Create montage: 64x64 grid of 30x30 images
        # Each small image shows where feature i is activated
        cell_size = 30  # Use actual grid size
        montage = np.zeros((64 * cell_size, 64 * cell_size))
        
        for feat_idx in range(4096):
            row = feat_idx // 64
            col = feat_idx % 64
            
            # Get this feature's spatial activation
            feat_map = spatial_features[:, :, feat_idx]  # [30, 30]
            
            montage[row*cell_size:(row+1)*cell_size, col*cell_size:(col+1)*cell_size] = feat_map
        
        # Save
        iter_path = os.path.join(output_dir, f"{puzzle_name}_iter{iter_idx+1:02d}_spatial.png")
        
        fig, ax = plt.subplots(figsize=(48, 48))
        vmax = np.percentile(montage[montage > 0], 99) if (montage > 0).any() else 1.0
        im = ax.imshow(montage, cmap='hot', aspect='equal', vmin=0, vmax=vmax)
        ax.set_title(f"SAE Features - Iter {iter_idx+1} - Spatial Activation\n{puzzle_name} (ID: {puzzle_id})\n"
                    f"Each cell = 1 feature (64×64 = 4096 features), Each pixel within cell = grid position (30×30)", 
                    fontsize=14)
        
        # Add grid lines every cell_size pixels
        for i in range(65):
            ax.axhline(i * cell_size - 0.5, color='white', linewidth=0.3, alpha=0.4)
            ax.axvline(i * cell_size - 0.5, color='white', linewidth=0.3, alpha=0.4)
        
        # Add feature index labels on edges
        for i in range(0, 64, 8):
            ax.text(-10, i * cell_size + cell_size/2, str(i*64), fontsize=6, ha='right', va='center')
            ax.text(i * cell_size + cell_size/2, -10, str(i), fontsize=6, ha='center', va='bottom')
        
        ax.set_xlim(-0.5, 64*cell_size - 0.5)
        ax.set_ylim(64*cell_size - 0.5, -0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.3, pad=0.01)
        cbar.set_label('Activation', fontsize=10)
        
        plt.savefig(iter_path, dpi=80, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"    Saved spatial visualization: {iter_path}")


def visualize_top_features(
    sae_features: np.ndarray,  # [D=16, L, M=4096]
    puzzle_name: str,
    output_path: str,
    top_k: int = 20,
):
    """
    Visualize the top-K most active features across all iterations.
    Shows which features have the highest total activation and their spatial patterns.
    """
    D, L, M = sae_features.shape
    
    # Total activation per feature across all iterations and positions
    total_activation = sae_features.sum(axis=(0, 1))  # [M]
    
    # Get top-K feature indices
    top_indices = np.argsort(total_activation)[::-1][:top_k]
    
    # Create figure
    n_cols = 5
    n_rows = (top_k + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3.5))
    axes = axes.flatten()
    
    for i, feat_idx in enumerate(top_indices):
        ax = axes[i]
        
        # Get this feature's activation across iterations
        # Shape: [D, L] for this feature
        feat_over_time = sae_features[:, :, feat_idx]  # [D, L]
        
        # Reshape to [D, 30, 30] if possible
        if L >= 900:
            feat_spatial = feat_over_time[:, :900].reshape(D, 30, 30)
        else:
            padded = np.zeros((D, 900))
            padded[:, :L] = feat_over_time
            feat_spatial = padded.reshape(D, 30, 30)
        
        # Show average activation across iterations
        avg_spatial = feat_spatial.mean(axis=0)
        
        im = ax.imshow(avg_spatial, cmap='hot', aspect='equal')
        ax.set_title(f"Feature {feat_idx}\nTotal: {total_activation[feat_idx]:.2f}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add border color based on activation
        for spine in ax.spines.values():
            spine.set_edgecolor('orange')
            spine.set_linewidth(2)
    
    # Hide unused axes
    for i in range(top_k, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f"Top {top_k} Most Active Features - {puzzle_name}\n(Averaged across {D} iterations)", 
                 fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    output_dir = os.path.dirname(output_path)
    if output_dir:  # Only create directory if path has a directory component
        os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return top_indices


def visualize_feature_heatmap(
    test_input: np.ndarray,
    test_label: np.ndarray,
    prediction: np.ndarray,
    sae_features: np.ndarray,  # [D=16, L, M=4096]
    puzzle_name: str,
    puzzle_id: int,
    output_path: str,
    top_m: int = 50,
    aggregation: str = 'sum',  # 'sum', 'max', or 'mean'
):
    """
    Visualize top-M most frequently activated features as a heatmap.
    
    Args:
        sae_features: [D=16, L, M=4096] feature activations
        top_m: number of top features to show
        aggregation: how to aggregate spatial positions ('sum', 'max', 'mean')
    
    Returns:
        Heatmap where:
        - Rows = top M features (sorted by frequency)
        - Columns = 16 iterations
        - Cell values = aggregated activation strength
    """
    D, L, M = sae_features.shape
    
    # Aggregate across spatial positions for each feature and iteration
    if aggregation == 'sum':
        feature_agg = sae_features.sum(axis=1)  # [D, M]
    elif aggregation == 'max':
        feature_agg = sae_features.max(axis=1)  # [D, M]
    else:  # mean
        feature_agg = sae_features.mean(axis=1)  # [D, M]
    
    # Calculate frequency: how many iterations each feature is active (> 0)
    feature_frequency = (feature_agg > 0).sum(axis=0)  # [M] - count of iterations where feature is active
    
    # Get top-M features by frequency
    top_m_indices = np.argsort(feature_frequency)[::-1][:top_m]
    
    # Extract activation matrix for top-M features: [M, D]
    activation_matrix = feature_agg[:, top_m_indices].T  # [top_m, D]
    
    # Decode grids for display
    input_grid = decode_arc_grid(test_input)
    label_grid = decode_arc_grid(test_label)
    pred_grid = decode_arc_grid(prediction)
    is_correct = np.array_equal(label_grid, pred_grid)
    
    # Create figure
    fig = plt.figure(figsize=(20, max(8, top_m * 0.15)))
    
    # Layout: Top row (3 panels) + Bottom heatmap
    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1, 3], hspace=0.3)
    
    # Top row: Test input, Ground truth, Prediction
    gs_top = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0], wspace=0.2)
    ax_input = fig.add_subplot(gs_top[0, 0])
    ax_label = fig.add_subplot(gs_top[0, 1])
    ax_pred = fig.add_subplot(gs_top[0, 2])
    
    visualize_grid(input_grid, ax_input, "Test Input")
    visualize_grid(label_grid, ax_label, "Ground Truth", highlight_color='#0074D9')
    
    status = "✓ CORRECT" if is_correct else "✗ INCORRECT"
    status_color = '#2ECC40' if is_correct else '#FF4136'
    visualize_grid(pred_grid, ax_pred, f"Prediction ({status})", highlight_color=status_color)
    
    # Bottom: Heatmap
    ax_heatmap = fig.add_subplot(gs[1])
    
    # Create heatmap
    im = ax_heatmap.imshow(activation_matrix, cmap='hot', aspect='auto', interpolation='nearest')
    
    # Set labels
    ax_heatmap.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax_heatmap.set_ylabel('Feature Index (sorted by frequency)', fontsize=12, fontweight='bold')
    
    # X-axis: iterations 1-16
    ax_heatmap.set_xticks(range(D))
    ax_heatmap.set_xticklabels([f'{i+1}' for i in range(D)], fontsize=9)
    
    # Y-axis: feature indices (show every 5th or so)
    y_step = max(1, top_m // 20)
    y_ticks = list(range(0, top_m, y_step)) + [top_m - 1] if top_m > 0 else []
    y_ticks = sorted(set(y_ticks))
    ax_heatmap.set_yticks(y_ticks)
    ax_heatmap.set_yticklabels([f'{top_m_indices[i]}' for i in y_ticks], fontsize=8)
    
    # Add grid for better readability
    ax_heatmap.set_xticks(np.arange(-0.5, D, 1), minor=True)
    ax_heatmap.set_yticks(np.arange(-0.5, top_m, 1), minor=True)
    ax_heatmap.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax_heatmap, pad=0.02)
    cbar.set_label(f'Activation ({aggregation})', fontsize=11)
    
    # Title
    title = f"Top {top_m} Features by Activation Frequency - {puzzle_name} (ID: {puzzle_id})"
    if not is_correct:
        title += f" [INCORRECT]"
    plt.suptitle(title, fontsize=13, fontweight='bold', y=0.98)
    
    # Add statistics text
    stats_text = f"Total features: {M} | Active features: {(feature_frequency > 0).sum()} | "
    stats_text += f"Top {top_m} shown | Aggregation: {aggregation}"
    ax_heatmap.text(0.5, -0.08, stats_text, transform=ax_heatmap.transAxes,
                    ha='center', fontsize=9, style='italic')
    
    # Save
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return {
        'puzzle_name': puzzle_name,
        'puzzle_id': puzzle_id,
        'is_correct': is_correct,
        'top_m_features': top_m_indices.tolist(),
        'feature_frequencies': feature_frequency[top_m_indices].tolist(),
    }


@hydra.main(config_path="config", config_name="cfg_eval", version_base=None)
def main(hydra_config: DictConfig):
    """Main visualization function"""
    # Build config - handle both nested arch config and flat config
    arch_dict = dict(hydra_config.arch) if 'arch' in hydra_config else {}
    
    # Handle loss config
    if 'loss' in arch_dict:
        loss_dict = dict(arch_dict.pop('loss'))
        arch_dict['loss'] = LossConfig(**loss_dict)
    else:
        arch_dict['loss'] = LossConfig(name='losses@ACTLossHead')
    
    config = VisConfig(
        arch=ArchConfig(**arch_dict),
        data_paths=list(hydra_config.data_paths) if 'data_paths' in hydra_config else ['data/arc1concept-aug-1000'],
        data_paths_test=list(hydra_config.get('data_paths_test', [])) if hydra_config.get('data_paths_test') else [],
        global_batch_size=hydra_config.get('global_batch_size', 1),
        load_checkpoint=hydra_config.get('load_checkpoint'),
        sae_checkpoint=hydra_config.get('sae_checkpoint', None),
        sae_model_type=hydra_config.get('sae_model_type', 'sae_fix'),
        puzzle_ids=list(hydra_config.get('puzzle_ids', [])) if hydra_config.get('puzzle_ids') else None,
        num_visualize=hydra_config.get('num_visualize', 10),
        output_dir=hydra_config.get('output_dir', 'sae_visualizations'),
        top_m_features=hydra_config.get('top_m_features', 50),
        sort_by_iteration=hydra_config.get('sort_by_iteration', 1),
        answer_only=hydra_config.get('answer_only', True),
        seed=hydra_config.get('seed', 0),
    )

    print(f"{'='*60}")
    print("SAE Feature Visualization")
    print(f"{'='*60}")
    print(f"SAE Model Type: {config.sae_model_type}")
    print(f"Output directory: {config.output_dir}")
    print(f"SAE checkpoint: {config.sae_checkpoint}")
    print(f"Puzzle IDs: {config.puzzle_ids}")
    print(f"Num visualize: {config.num_visualize}")
    print(f"Top M features (heatmap): {config.top_m_features}")
    print(f"{'='*60}\n")
    
    # Set seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Load data
    print("Loading dataset...")
    try:
        dataloader, metadata = create_dataloader(config, "test")
    except:
        print("No test split found, using train split")
        dataloader, metadata = create_dataloader(config, "train")
    
    # Load models
    print("Loading TRM model...")
    model = create_model(config, metadata)
    model.eval()
    
    print("Loading SAE model...")
    sae = load_sae(config)
    
    # Load visualization context
    print("Loading visualization context...")
    viz_context = load_visualization_context(config)
    identifiers_map = viz_context['identifiers_map']
    print(f"  Loaded {len(identifiers_map)} identifiers")
    
    # Process puzzles
    results = []
    viz_count = 0
    global_idx = 0
    
    puzzle_ids_set = set(config.puzzle_ids) if config.puzzle_ids else None
    
    print(f"\nProcessing puzzles...")
    
    with torch.inference_mode():
        for set_name, batch, global_batch_size in dataloader:
            batch_size = batch["inputs"].shape[0]
            
            # Filter by puzzle_ids if specified
            if puzzle_ids_set is not None:
                batch_indices = range(global_idx, global_idx + batch_size)
                has_match = any(idx in puzzle_ids_set for idx in batch_indices)
                if not has_match:
                    global_idx += batch_size
                    continue
            
            # Check visualization limit
            if viz_count >= config.num_visualize:
                break
            
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            batch_float = {k: v.to(dtype=DTYPE) if v.dtype.is_floating_point else v for k, v in batch.items()}
            
            # Collect trajectories
            print(f"  Processing batch at idx {global_idx}...")
            z_L, preds = collect_trajectories(model, batch_float)
            
            # z_L: [B, D, L, H]
            z_L = z_L.to(dtype=DTYPE)
            
            # Get SAE features
            sae_out = sae(z_L)
            z_n = sae_out['z_n'].float().cpu().numpy()  # [B, D, L, M] - convert to float32 for numpy
            
            # Visualize each example in batch
            batch_inputs = batch["inputs"].cpu().numpy()
            batch_labels = batch["labels"].cpu().numpy()
            # Predictions are token IDs (integers), no need to convert to float
            if preds is not None and "preds" in preds:
                batch_preds = preds["preds"].cpu().numpy()
            else:
                print("  Warning: No predictions available, using labels as predictions")
                batch_preds = batch_labels
            batch_puzzle_ids = batch["puzzle_identifiers"].cpu().numpy()
            
            for i in range(batch_size):
                current_idx = global_idx + i
                
                # Check filters
                if puzzle_ids_set is not None and current_idx not in puzzle_ids_set:
                    continue
                
                if viz_count >= config.num_visualize:
                    break
                
                # Get puzzle name
                pid = int(batch_puzzle_ids[i])
                if pid < len(identifiers_map):
                    full_name = identifiers_map[pid]
                else:
                    full_name = f"puzzle_{pid}"
                base_name = full_name.split("|||")[0] if "|||" in full_name else full_name
                
                # Output path
                save_path = os.path.join(config.output_dir, f"sae_viz_{viz_count:04d}_{base_name}.png")
                
                # Visualize
                print(f"  [{viz_count+1}/{config.num_visualize}] Visualizing {base_name} (idx={current_idx})...")
                
                # 1. Main visualization (heatmap: top M features × 16 iterations)
                result = visualize_sae_features(
                    test_input=batch_inputs[i],
                    test_label=batch_labels[i],
                    prediction=batch_preds[i],
                    sae_features=z_n[i],  # [D, L, M]
                    puzzle_name=base_name,
                    puzzle_id=pid,
                    output_path=save_path,
                    top_m=config.top_m_features,
                    sort_by_iteration=config.sort_by_iteration,
                    answer_only=config.answer_only,
                )
                
                # 2. Top features visualization
                top_features_path = os.path.join(config.output_dir, f"sae_top_{viz_count:04d}_{base_name}.png")
                top_indices = visualize_top_features(
                    sae_features=z_n[i],
                    puzzle_name=base_name,
                    output_path=top_features_path,
                    top_k=20,
                )
                
                # 3. Spatial visualization for key iterations (first, middle, last)
                # This creates large images, so only do for a few iterations
                spatial_dir = os.path.join(config.output_dir, "spatial", base_name)
                visualize_sae_features_spatial(
                    sae_features=z_n[i],
                    puzzle_name=base_name,
                    puzzle_id=pid,
                    output_dir=spatial_dir,
                    selected_iters=[0, 7, 15],  # First, middle, last iterations
                )
                
                results.append(result)
                viz_count += 1
                
                status = "✓" if result['is_correct'] else f"✗ ({result['num_diff']} diff)"
                print(f"       {status} | {result['total_features_activated']} unique features activated")
                print(f"       Top features: {list(top_indices[:5])}")
            
            global_idx += batch_size
            
            # Early exit if all targets processed
            if puzzle_ids_set is not None and global_idx > max(puzzle_ids_set):
                break
    
    # Save results summary
    summary_path = os.path.join(config.output_dir, "results_summary.json")
    with open(summary_path, 'w') as f:
        json.dump({
            'total_visualized': len(results),
            'correct': sum(1 for r in results if r['is_correct']),
            'results': results
        }, f, indent=2)
    
    print(f"\n{'='*60}")
    print("VISUALIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total visualized: {len(results)}")
    print(f"Correct: {sum(1 for r in results if r['is_correct'])}")
    print(f"Output directory: {config.output_dir}")
    print(f"Results saved to: {summary_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
