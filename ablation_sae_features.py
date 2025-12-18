"""
SAE Feature Ablation Study

This script performs ablation studies on SAE features to understand their causal role:
1. Select features to ablate (top-k, random, or specific indices)
2. Ablate features by removing their contribution from z_L
3. Compare predictions before/after ablation (accuracy and visualization)

Usage:
    python ablation_sae_features.py \
        --config-path=ckpt/arc_v1_public \
        --config-name=all_config \
        load_checkpoint=ckpt/arc_v1_public/step_518071 \
        data_paths="[data/arc1concept-aug-1000]" \
        ++sae_model_type=sae_fix \
        ++output_dir=ablation_results \
        ++ablation_mode=top_k \
        ++num_features_to_ablate=10
"""

from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
import os
import json
import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
IGNORE_LABEL_ID = -100


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
    """SAE with ablation capability"""
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

        # SAE dictionary
        self.dictionary_enc = nn.Parameter(torch.randn(n_features, H, dtype=DTYPE) * (2.0 / (H + n_features)) ** 0.5)
        self.dictionary_dec = nn.Parameter(torch.randn(H, n_features, dtype=DTYPE) * (2.0 / (n_features + H)) ** 0.5)
        self.bias_pre = nn.Parameter(torch.zeros(H, dtype=DTYPE))
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

    def encode(self, zL):
        """Encode to sparse features"""
        original_shape = zL.shape
        H = self.d_model
        x = zL.reshape(-1, H)
        logits = F.linear(x - self.bias_pre[None, :], self.dictionary_enc, self.bias_enc)
        z_n_dense = F.relu(logits)
        z_n = self.topk_activation(z_n_dense)
        new_shape = original_shape[:-1] + (self.n_features,)
        return z_n.view(*new_shape)

    def decode(self, z_n):
        """Decode sparse features back to hidden space"""
        original_shape = z_n.shape
        z_n_flat = z_n.reshape(-1, self.n_features)
        x_tgt_flat = F.linear(z_n_flat, self.dictionary_dec) + self.bias_pre[None, :]
        new_shape = original_shape[:-1] + (self.d_model,)
        return x_tgt_flat.view(*new_shape)

    def forward(self, zL, mask=None):
        """Full forward pass"""
        B, D, L, H = zL.shape
        z_n = self.encode(zL)
        x_tgt = self.decode(z_n)
        return {"z_n": z_n, "x_tgt": x_tgt}

    def compute_feature_contribution(self, z_n, feature_indices: List[int]):
        """
        Compute the contribution of specific features to the reconstruction.
        
        Args:
            z_n: [B, D, L, M] sparse feature activations
            feature_indices: list of feature indices to compute contribution for
        
        Returns:
            contribution: [B, D, L, H] - the contribution of specified features
        """
        # Create mask for selected features
        mask = torch.zeros(self.n_features, device=z_n.device, dtype=z_n.dtype)
        mask[feature_indices] = 1.0
        
        # Zero out non-selected features
        z_n_selected = z_n * mask[None, None, None, :]
        
        # Decode only selected features
        contribution = self.decode(z_n_selected)
        
        return contribution

    def ablate_features(self, zL, feature_indices: List[int]):
        """
        Ablate specific features by subtracting their contribution from zL.
        
        Args:
            zL: [B, D, L, H] hidden states
            feature_indices: list of feature indices to ablate
        
        Returns:
            zL_ablated: [B, D, L, H] hidden states with features ablated
            z_n: [B, D, L, M] original feature activations
            contribution: [B, D, L, H] the removed contribution
        """
        # Get sparse features
        z_n = self.encode(zL)
        
        # Compute contribution of features to ablate
        contribution = self.compute_feature_contribution(z_n, feature_indices)
        
        # Ablate by subtracting contribution
        zL_ablated = zL - contribution
        
        return zL_ablated, z_n, contribution


class TSAE(nn.Module):
    """TSAE with ablation capability"""
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

        self.attn_l = Attn(H, n_heads, is_causal=False)
        self.norm_l = nn.LayerNorm(H)
        self.attn_d = Attn(H, n_heads, is_causal=True)
        self.norm_d = nn.LayerNorm(H)

        self.dictionary_enc = nn.Parameter(torch.randn(n_features, H, dtype=DTYPE) * (2.0 / (H + n_features)) ** 0.5)
        self.dictionary_dec = nn.Parameter(torch.randn(H, n_features, dtype=DTYPE) * (2.0 / (n_features + H)) ** 0.5)
        self.bias_pre = nn.Parameter(torch.zeros(H, dtype=DTYPE))
        self.bias_enc = nn.Parameter(torch.zeros(n_features, dtype=DTYPE))
        self.query_token = nn.Parameter(torch.zeros(H, dtype=DTYPE))

        self.register_buffer("num_tokens_seen", torch.zeros((), dtype=torch.long))
        self.register_buffer("last_active_token", torch.zeros(n_features, dtype=torch.long))

    def topk_activation(self, x):
        if self.topk is None or self.topk >= x.size(-1):
            return x
        values, indices = torch.topk(x, self.topk, dim=-1)
        out = torch.zeros_like(x)
        out.scatter_(dim=-1, index=indices, src=values)
        return out

    def encode(self, zL):
        """Encode with attention preprocessing"""
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
        logits = F.linear(x_src - self.bias_pre[None, :], self.dictionary_enc, self.bias_enc)
        z_n_dense = F.relu(logits)
        z_n_flat = self.topk_activation(z_n_dense)
        z_n = z_n_flat.view(B, D, L, self.n_features)
        
        return z_n, x_prior

    def decode(self, z_n):
        """Decode sparse features"""
        original_shape = z_n.shape
        z_n_flat = z_n.reshape(-1, self.n_features)
        x_tgt_flat = F.linear(z_n_flat, self.dictionary_dec) + self.bias_pre[None, :]
        new_shape = original_shape[:-1] + (self.d_model,)
        return x_tgt_flat.view(*new_shape)

    def forward(self, zL, mask=None):
        z_n, x_prior = self.encode(zL)
        return {"z_n": z_n, "x_prior": x_prior}

    def compute_feature_contribution(self, z_n, feature_indices: List[int]):
        """Compute the contribution of specific features"""
        mask = torch.zeros(self.n_features, device=z_n.device, dtype=z_n.dtype)
        mask[feature_indices] = 1.0
        z_n_selected = z_n * mask[None, None, None, :]
        contribution = self.decode(z_n_selected)
        return contribution

    def ablate_features(self, zL, feature_indices: List[int]):
        """Ablate specific features"""
        z_n, x_prior = self.encode(zL)
        contribution = self.compute_feature_contribution(z_n, feature_indices)
        # For TSAE, the features encode the residual, so we subtract from the residual
        zL_ablated = zL - contribution
        return zL_ablated, z_n, contribution


class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str


class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str
    loss: LossConfig


class AblationConfig(pydantic.BaseModel):
    # Model config
    arch: ArchConfig
    data_paths: List[str]
    data_paths_test: List[str] = []
    global_batch_size: int = 1
    
    # Checkpoints
    load_checkpoint: Optional[str] = None
    sae_checkpoint: Optional[str] = None
    
    # Ablation options
    ablation_mode: str = "top_k"  # "top_k", "random", "specific", "all_individually", "progressive"
    num_features_to_ablate: int = 10
    specific_features: Optional[List[int]] = None  # For "specific" mode
    ablation_iterations: Optional[List[int]] = None  # Which iterations to ablate (None = all)
    
    # For progressive mode - ranking metric
    ranking_metric: str = "avg_activation"  # "avg_activation" or "activation_freq"
    max_k_features: int = 20  # Maximum K to test in progressive mode
    max_error_changes: int = 0  # Stop after N error changes (0 = no limit, run all K)
    
    # For all_individually mode
    only_active_features: bool = True  # Only test features that are active for this sample
    save_only_on_change: bool = True  # Only save visualization when accuracy changes
    
    # Visualization options
    puzzle_ids: Optional[List[int]] = None
    num_visualize: int = 10
    output_dir: str = "ablation_results"
    top_m_features: int = 50
    sort_by_iteration: int = 1
    answer_only: bool = True

    # SAE model type
    sae_model_type: str = "sae_fix"
    
    # SAE config
    sae_depth: int = 16
    sae_d_model: int = 512
    sae_n_features: int = 4096
    sae_topk: int = 64
    
    seed: int = 0


def create_dataloader(config: AblationConfig, split: str):
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


def create_model(config: AblationConfig, metadata: PuzzleDatasetMetadata):
    """Create TRM model"""
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

    os.environ["DISABLE_COMPILE"] = "1"
    
    with device:
        model = model_cls(model_cfg)
        model = loss_head_cls(model, **loss_extra)
        
        if config.load_checkpoint is not None:
            print(f"Loading TRM checkpoint: {config.load_checkpoint}")
            state_dict = torch.load(config.load_checkpoint, map_location=device, weights_only=False)
            
            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = k.replace("_orig_mod.", "")
                new_state_dict[new_key] = v
            
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


def load_sae(config: AblationConfig):
    """Load SAE model"""
    model_type = config.sae_model_type.lower()
    
    sae_checkpoint = config.sae_checkpoint
    if sae_checkpoint is None or sae_checkpoint == "":
        sae_checkpoint = f"weights/{model_type}/best_val.pt"
    
    print(f"SAE Model Type: {model_type}")
    print(f"SAE Checkpoint: {sae_checkpoint}")
    
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
        raise ValueError(f"Unknown SAE model type: {model_type}")
    
    if sae_checkpoint is not None and os.path.exists(sae_checkpoint):
        print(f"Loading checkpoint from: {sae_checkpoint}")
        ckpt = torch.load(sae_checkpoint, map_location=device, weights_only=False)
        if 'sae_state_dict' in ckpt:
            sae.load_state_dict(ckpt['sae_state_dict'])
        else:
            sae.load_state_dict(ckpt)
        print(f"  Loaded successfully!")
    else:
        print(f"  Warning: Checkpoint not found at {sae_checkpoint}")
    
    sae.eval()
    return sae


def load_visualization_context(config: AblationConfig) -> Dict:
    """Load identifiers for visualization"""
    data_paths = config.data_paths_test if len(config.data_paths_test) > 0 else config.data_paths
    
    context = {'identifiers_map': [], 'test_puzzles': {}}
    
    for data_path in data_paths:
        identifiers_path = os.path.join(data_path, "identifiers.json")
        if os.path.exists(identifiers_path):
            with open(identifiers_path, 'r') as f:
                identifiers = json.load(f)
                if not context['identifiers_map']:
                    context['identifiers_map'] = identifiers
                else:
                    context['identifiers_map'].extend(identifiers)
    
    return context


def collect_trajectories_with_intervention(
    model, 
    batch, 
    sae,
    ablate_features: Optional[List[int]] = None,
    ablate_iterations: Optional[List[int]] = None,
    max_steps: int = 16
) -> Tuple[torch.Tensor, Dict, List[torch.Tensor]]:
    """
    Run inference and optionally ablate SAE features during specific iterations.
    
    Args:
        model: TRM model
        batch: input batch
        sae: SAE model
        ablate_features: list of feature indices to ablate (None = no ablation)
        ablate_iterations: which iterations to apply ablation (None = all, 0-indexed)
        max_steps: maximum inference steps
    
    Returns:
        z_L_trajectory: [B, D, L, H] collected trajectories
        preds: final predictions
        ablation_info: list of ablation info per step
    """
    with torch.no_grad():
        with device:
            carry = model.initial_carry(batch)
        
        trajectories = []
        ablation_info = []
        preds = None
        
        def get_z_L(carry_obj):
            if hasattr(carry_obj, 'inner_carry'):
                inner = carry_obj.inner_carry
                if hasattr(inner, 'z_L'):
                    return inner.z_L
            if hasattr(carry_obj, 'z_L'):
                return carry_obj.z_L
            return None
        
        def set_z_L(carry_obj, new_z_L):
            """Create a new carry with modified z_L"""
            if hasattr(carry_obj, 'inner_carry'):
                inner = carry_obj.inner_carry
                if hasattr(inner, 'z_L'):
                    # Create new inner carry with modified z_L
                    new_inner = type(inner)(
                        z_H=inner.z_H,
                        z_L=new_z_L
                    )
                    # Create new outer carry
                    new_carry = type(carry_obj)(
                        inner_carry=new_inner,
                        steps=carry_obj.steps,
                        halted=carry_obj.halted,
                        current_data=carry_obj.current_data
                    )
                    return new_carry
            return carry_obj
        
        for step in range(max_steps):
            z_L_state = get_z_L(carry)
            if z_L_state is not None:
                trajectories.append(z_L_state.detach().clone())
                
                # Apply ablation if specified
                if ablate_features is not None and len(ablate_features) > 0:
                    should_ablate = (ablate_iterations is None) or (step in ablate_iterations)
                    if should_ablate:
                        # Reshape z_L for SAE: [B, L, H] -> [B, 1, L, H]
                        z_L_for_sae = z_L_state.unsqueeze(1)
                        
                        # Ablate features
                        z_L_ablated, z_n, contribution = sae.ablate_features(z_L_for_sae, ablate_features)
                        z_L_ablated = z_L_ablated.squeeze(1)
                        
                        # Update carry with ablated z_L
                        carry = set_z_L(carry, z_L_ablated)
                        
                        ablation_info.append({
                            'step': step,
                            'contribution_norm': contribution.norm().item(),
                            'z_n_sum': z_n[..., ablate_features].sum().item()
                        })
            
            carry, loss, metrics, preds, all_finish = model(
                carry=carry, batch=batch, return_keys={'preds'}
            )
            
            if all_finish:
                z_L_state = get_z_L(carry)
                if z_L_state is not None:
                    trajectories.append(z_L_state.detach().clone())
                break
        
        if len(trajectories) == 0:
            raise RuntimeError("No z_L trajectories collected!")
        
        if len(trajectories) > 1:
            z_L = torch.stack(trajectories[1:], dim=1)
        else:
            z_L = trajectories[0].unsqueeze(1)
        
        if z_L.shape[1] < 16:
            pad_size = 16 - z_L.shape[1]
            z_L = F.pad(z_L, (0, 0, 0, 0, 0, pad_size), mode='replicate')
        elif z_L.shape[1] > 16:
            z_L = z_L[:, :16]
        
        return z_L, preds, ablation_info


def rank_features_by_metric(
    sae_features: np.ndarray,  # [D, L, M]
    labels: np.ndarray,  # [L]
    config: AblationConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rank features by the specified metric.
    
    Args:
        sae_features: [D, L, M] feature activations
        labels: [L] labels for answer mask
        config: ablation configuration
    
    Returns:
        ranked_indices: feature indices sorted by importance (descending)
        metric_values: metric values for each feature (in original order)
    """
    D, L, M = sae_features.shape
    topk = config.sae_topk  # typically 64
    
    # Create answer mask
    answer_mask = (labels != IGNORE_LABEL_ID)
    if len(answer_mask) > L:
        answer_mask = answer_mask[:L]
    elif len(answer_mask) < L:
        answer_mask = np.pad(answer_mask, (0, L - len(answer_mask)), constant_values=False)
    
    # Apply answer mask if configured
    if config.answer_only:
        sae_features_masked = sae_features * answer_mask[None, :, None]
    else:
        sae_features_masked = sae_features
    
    if config.ranking_metric == "avg_activation":
        # Metric 1: Sum of activation values
        # Features with higher total activation are considered more important
        if config.sort_by_iteration == 0:
            # Sum across all iterations and positions
            metric_values = sae_features_masked.sum(axis=(0, 1))  # [M]
        else:
            # Sum only for specific iteration
            iter_idx = min(config.sort_by_iteration - 1, D - 1)
            metric_values = sae_features_masked[iter_idx].sum(axis=0)  # [M]
        
    elif config.ranking_metric == "activation_freq":
        # Metric 2: Activation frequency
        # Count how often each feature appears in the top-K (is non-zero after top-k selection)
        # Since SAE already applies top-k, non-zero activations mean the feature was selected
        
        if config.sort_by_iteration == 0:
            # Count non-zero activations across all iterations and positions
            # [D, L, M] -> count non-zeros for each feature
            active_count = (sae_features_masked > 0).sum(axis=(0, 1))  # [M]
        else:
            # Count for specific iteration only
            iter_idx = min(config.sort_by_iteration - 1, D - 1)
            active_count = (sae_features_masked[iter_idx] > 0).sum(axis=0)  # [M]
        
        metric_values = active_count.astype(np.float32)
    else:
        raise ValueError(f"Unknown ranking metric: {config.ranking_metric}")
    
    # Rank by metric (descending order - highest importance first)
    ranked_indices = np.argsort(metric_values)[::-1]
    
    return ranked_indices, metric_values


def select_features_to_ablate(
    sae_features: np.ndarray,  # [D, L, M]
    labels: np.ndarray,  # [L]
    config: AblationConfig,
) -> List[int]:
    """
    Select features to ablate based on the specified mode.
    
    Args:
        sae_features: [D, L, M] feature activations
        labels: [L] labels for answer mask
        config: ablation configuration
    
    Returns:
        list of feature indices to ablate
    """
    D, L, M = sae_features.shape
    
    # Create answer mask
    answer_mask = (labels != IGNORE_LABEL_ID)
    if len(answer_mask) > L:
        answer_mask = answer_mask[:L]
    elif len(answer_mask) < L:
        answer_mask = np.pad(answer_mask, (0, L - len(answer_mask)), constant_values=False)
    
    if config.ablation_mode == "specific":
        # Use user-specified features
        if config.specific_features is None:
            raise ValueError("specific_features must be provided for 'specific' mode")
        return config.specific_features
    
    elif config.ablation_mode == "random":
        # Randomly select features
        np.random.seed(config.seed)
        return list(np.random.choice(M, config.num_features_to_ablate, replace=False))
    
    elif config.ablation_mode == "top_k":
        # Select top-k most active features (considering answer tokens if answer_only)
        if config.answer_only:
            sae_features_masked = sae_features * answer_mask[None, :, None]
        else:
            sae_features_masked = sae_features
        
        # Sort by iteration if specified
        if config.sort_by_iteration == 0:
            feature_total = sae_features_masked.sum(axis=(0, 1))
        else:
            iter_idx = min(config.sort_by_iteration - 1, D - 1)
            feature_total = sae_features_masked[iter_idx].sum(axis=0)
        
        top_indices = np.argsort(feature_total)[::-1][:config.num_features_to_ablate]
        return list(top_indices)
    
    elif config.ablation_mode == "bottom_k":
        # Select bottom-k (least active but non-zero) features
        if config.answer_only:
            sae_features_masked = sae_features * answer_mask[None, :, None]
        else:
            sae_features_masked = sae_features
        
        feature_total = sae_features_masked.sum(axis=(0, 1))
        active_mask = feature_total > 0
        active_indices = np.where(active_mask)[0]
        
        if len(active_indices) == 0:
            return []
        
        sorted_active = active_indices[np.argsort(feature_total[active_indices])]
        return list(sorted_active[:config.num_features_to_ablate])
    
    elif config.ablation_mode == "progressive":
        # For progressive mode, return all ranked features (up to max_k)
        # The actual progressive ablation is handled separately
        ranked_indices, _ = rank_features_by_metric(sae_features, labels, config)
        return list(ranked_indices[:config.max_k_features])
    
    else:
        raise ValueError(f"Unknown ablation mode: {config.ablation_mode}")


def compute_accuracy(preds: np.ndarray, labels: np.ndarray) -> Tuple[float, int, int]:
    """Compute token accuracy and exact match"""
    mask = (labels != IGNORE_LABEL_ID)
    if not mask.any():
        return 0.0, 0, 0
    
    correct = ((preds == labels) & mask).sum()
    total = mask.sum()
    token_acc = correct / total
    
    exact_match = 1 if np.array_equal(preds[mask], labels[mask]) else 0
    
    return float(token_acc), int(correct), int(total)


def visualize_ablation_comparison(
    test_input: np.ndarray,
    test_label: np.ndarray,
    pred_original: np.ndarray,
    pred_ablated: np.ndarray,
    ablated_features: List[int],
    puzzle_name: str,
    puzzle_id: int,
    output_path: str,
):
    """
    Create visualization comparing predictions before and after ablation.
    """
    # Decode grids
    input_grid = decode_arc_grid(test_input)
    label_grid = decode_arc_grid(test_label)
    pred_orig_grid = decode_arc_grid(pred_original)
    pred_abl_grid = decode_arc_grid(pred_ablated)
    
    is_correct_orig = np.array_equal(label_grid, pred_orig_grid)
    is_correct_abl = np.array_equal(label_grid, pred_abl_grid)
    
    diff_orig, num_diff_orig = compute_diff(label_grid, pred_orig_grid)
    diff_abl, num_diff_abl = compute_diff(label_grid, pred_abl_grid)
    diff_between, num_diff_between = compute_diff(pred_orig_grid, pred_abl_grid)
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Top row: Input, Label, Original Prediction
    visualize_grid(input_grid, axes[0, 0], "Test Input")
    visualize_grid(label_grid, axes[0, 1], "Ground Truth", highlight_color='#0074D9')
    
    status_orig = "✓ CORRECT" if is_correct_orig else f"✗ WRONG ({num_diff_orig} diff)"
    color_orig = '#2ECC40' if is_correct_orig else '#FF4136'
    visualize_grid(pred_orig_grid, axes[0, 2], f"Original Pred ({status_orig})", 
                   highlight_color=color_orig,
                   show_diff=diff_orig if not is_correct_orig else None)
    
    # Bottom row: Ablated Prediction, Difference, Feature info
    status_abl = "✓ CORRECT" if is_correct_abl else f"✗ WRONG ({num_diff_abl} diff)"
    color_abl = '#2ECC40' if is_correct_abl else '#FF4136'
    visualize_grid(pred_abl_grid, axes[1, 0], f"Ablated Pred ({status_abl})",
                   highlight_color=color_abl,
                   show_diff=diff_abl if not is_correct_abl else None)
    
    # Show difference between original and ablated
    axes[1, 1].set_title(f"Difference (Orig vs Ablated)\n{num_diff_between} positions changed", fontsize=11)
    if num_diff_between > 0:
        diff_vis = np.zeros((*pred_orig_grid.shape, 3))
        for i in range(pred_orig_grid.shape[0]):
            for j in range(pred_orig_grid.shape[1]):
                if diff_between is not None and diff_between[i, j]:
                    diff_vis[i, j] = [1, 0, 0]  # Red for changed
                else:
                    # ARC_COLORS is a list of hex strings, convert to RGB
                    color_idx = int(pred_orig_grid[i, j]) % len(ARC_COLORS)
                    hex_color = ARC_COLORS[color_idx]
                    # Convert hex to RGB (e.g., '#FF4136' -> (255, 65, 54))
                    r = int(hex_color[1:3], 16) / 255
                    g = int(hex_color[3:5], 16) / 255
                    b = int(hex_color[5:7], 16) / 255
                    diff_vis[i, j] = [r, g, b]
        axes[1, 1].imshow(diff_vis, interpolation='nearest')
    else:
        axes[1, 1].text(0.5, 0.5, "No Change", ha='center', va='center', 
                       fontsize=14, transform=axes[1, 1].transAxes)
    axes[1, 1].set_xticks([])
    axes[1, 1].set_yticks([])
    
    # Feature info
    axes[1, 2].axis('off')
    info_text = f"Ablated Features:\n"
    info_text += f"  Count: {len(ablated_features)}\n"
    info_text += f"  Indices: {ablated_features[:10]}"
    if len(ablated_features) > 10:
        info_text += f"\n  ... and {len(ablated_features) - 10} more"
    info_text += f"\n\nEffect:"
    info_text += f"\n  Original: {'✓' if is_correct_orig else '✗'} ({100 - num_diff_orig/max(1, np.sum(test_label != IGNORE_LABEL_ID))*100:.1f}% correct)"
    info_text += f"\n  Ablated:  {'✓' if is_correct_abl else '✗'} ({100 - num_diff_abl/max(1, np.sum(test_label != IGNORE_LABEL_ID))*100:.1f}% correct)"
    info_text += f"\n  Changed:  {num_diff_between} positions"
    
    axes[1, 2].text(0.1, 0.7, info_text, fontsize=10, va='top', family='monospace',
                   transform=axes[1, 2].transAxes)
    
    plt.suptitle(f"Ablation Study: {puzzle_name} (ID: {puzzle_id})", fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return {
        'puzzle_name': puzzle_name,
        'puzzle_id': puzzle_id,
        'is_correct_original': is_correct_orig,
        'is_correct_ablated': is_correct_abl,
        'num_diff_original': num_diff_orig,
        'num_diff_ablated': num_diff_abl,
        'num_changed_positions': num_diff_between,
        'ablated_features': ablated_features,
    }


def run_individual_feature_ablation(
    model,
    sae,
    batch: Dict,
    batch_float: Dict,
    sae_features: np.ndarray,  # [D, L, M]
    original_preds: np.ndarray,  # [L]
    labels: np.ndarray,  # [L]
    inputs: np.ndarray,  # [L]
    puzzle_name: str,
    puzzle_id: int,
    output_dir: str,
    config: 'AblationConfig',
) -> Dict:
    """
    Ablate each active feature individually and record accuracy changes.
    
    Args:
        model: TRM model
        sae: SAE model
        batch: single-sample batch dict
        batch_float: batch with float dtype
        sae_features: [D, L, M] SAE feature activations
        original_preds: original model predictions
        labels: ground truth labels
        inputs: input tokens
        puzzle_name: name of puzzle
        puzzle_id: puzzle identifier
        output_dir: directory to save results
        config: ablation configuration
    
    Returns:
        dict with results for this puzzle
    """
    D, L, M = sae_features.shape
    
    # Create puzzle-specific directory
    puzzle_dir = os.path.join(output_dir, puzzle_name)
    os.makedirs(puzzle_dir, exist_ok=True)
    
    # Compute original accuracy
    label_grid = decode_arc_grid(labels)
    orig_pred_grid = decode_arc_grid(original_preds)
    is_correct_orig = np.array_equal(label_grid, orig_pred_grid)
    _, orig_num_diff = compute_diff(label_grid, orig_pred_grid)
    
    # Create answer mask for feature selection
    answer_mask = (labels != IGNORE_LABEL_ID)
    if len(answer_mask) > L:
        answer_mask = answer_mask[:L]
    elif len(answer_mask) < L:
        answer_mask = np.pad(answer_mask, (0, L - len(answer_mask)), constant_values=False)
    
    # Get active features (only test features that are actually active)
    if config.only_active_features:
        if config.answer_only:
            sae_features_masked = sae_features * answer_mask[None, :, None]
        else:
            sae_features_masked = sae_features
        feature_total = sae_features_masked.sum(axis=(0, 1))
        active_features = np.where(feature_total > 0)[0]
    else:
        active_features = np.arange(M)
    
    print(f"    Testing {len(active_features)} active features individually...")
    
    # Results for this puzzle
    feature_results = []
    improved_count = 0
    degraded_count = 0
    unchanged_count = 0
    
    # Progress bar
    pbar = tqdm.tqdm(active_features, desc=f"    Ablating features", leave=False)
    
    for feat_idx in pbar:
        feat_idx = int(feat_idx)
        
        # Run ablation with single feature
        _, preds_abl, _ = collect_trajectories_with_intervention(
            model, batch_float, sae,
            ablate_features=[feat_idx],
            ablate_iterations=config.ablation_iterations,
            max_steps=16
        )
        
        ablated_preds = preds_abl["preds"].cpu().numpy()[0] if preds_abl and "preds" in preds_abl else labels
        
        # Compute ablated accuracy
        abl_pred_grid = decode_arc_grid(ablated_preds)
        is_correct_abl = np.array_equal(label_grid, abl_pred_grid)
        _, abl_num_diff = compute_diff(label_grid, abl_pred_grid)
        
        # Compute accuracy difference (positive = got worse, negative = got better)
        # Use number of differing cells as accuracy metric
        acc_diff = abl_num_diff - orig_num_diff  # positive = more wrong after ablation
        
        # Determine change type
        if is_correct_orig and not is_correct_abl:
            change_type = "degraded"
            degraded_count += 1
        elif not is_correct_orig and is_correct_abl:
            change_type = "improved"
            improved_count += 1
        elif orig_num_diff != abl_num_diff:
            change_type = "changed"
            if acc_diff > 0:
                degraded_count += 1
            else:
                improved_count += 1
        else:
            change_type = "unchanged"
            unchanged_count += 1
        
        # Record result
        result = {
            'feature_idx': feat_idx,
            'original_correct': is_correct_orig,
            'ablated_correct': is_correct_abl,
            'original_diff': orig_num_diff,
            'ablated_diff': abl_num_diff,
            'acc_diff': acc_diff,
            'change_type': change_type,
        }
        feature_results.append(result)
        
        # Save visualization only if accuracy changed
        if config.save_only_on_change and change_type == "unchanged":
            continue
        
        # Create filename with feature index and acc diff
        if acc_diff > 0:
            acc_str = f"+{acc_diff}"  # Got worse
        elif acc_diff < 0:
            acc_str = f"{acc_diff}"  # Got better (negative)
        else:
            acc_str = "0"
        
        filename = f"{puzzle_name}_feature_{feat_idx:04d}_acc_diff_{acc_str}.png"
        output_path = os.path.join(puzzle_dir, filename)
        
        # Create visualization
        visualize_ablation_comparison(
            test_input=inputs,
            test_label=labels,
            pred_original=original_preds,
            pred_ablated=ablated_preds,
            ablated_features=[feat_idx],
            puzzle_name=puzzle_name,
            puzzle_id=puzzle_id,
            output_path=output_path,
        )
        
        pbar.set_postfix({
            'improved': improved_count, 
            'degraded': degraded_count,
            'last_feat': feat_idx
        })
    
    pbar.close()
    
    # Save puzzle-level summary
    puzzle_summary = {
        'puzzle_name': puzzle_name,
        'puzzle_id': puzzle_id,
        'original_correct': is_correct_orig,
        'original_diff': orig_num_diff,
        'total_features_tested': len(active_features),
        'improved_count': improved_count,
        'degraded_count': degraded_count,
        'unchanged_count': unchanged_count,
        'feature_results': feature_results,
    }
    
    summary_path = os.path.join(puzzle_dir, "feature_ablation_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(puzzle_summary, f, indent=2)
    
    print(f"    Completed: {improved_count} improved, {degraded_count} degraded, {unchanged_count} unchanged")
    print(f"    Results saved to: {puzzle_dir}")
    
    return puzzle_summary


def visualize_progressive_ablation(
    test_input: np.ndarray,
    test_label: np.ndarray,
    original_preds: np.ndarray,
    pred_prev: np.ndarray,
    pred_curr: np.ndarray,
    k_prev: int,
    k_curr: int,
    new_feature: int,
    puzzle_name: str,
    puzzle_id: int,
    output_path: str,
):
    """
    Create simple visualization: Test Input | Ground Truth | Original (K=0) | K Pred | K+1 Pred
    """
    # Decode grids
    input_grid = decode_arc_grid(test_input)
    label_grid = decode_arc_grid(test_label)
    orig_pred_grid = decode_arc_grid(original_preds)
    pred_prev_grid = decode_arc_grid(pred_prev)
    pred_curr_grid = decode_arc_grid(pred_curr)
    
    # Compute correctness and errors
    is_correct_orig = np.array_equal(label_grid, orig_pred_grid)
    is_correct_prev = np.array_equal(label_grid, pred_prev_grid)
    is_correct_curr = np.array_equal(label_grid, pred_curr_grid)
    
    _, num_diff_orig = compute_diff(label_grid, orig_pred_grid)
    _, num_diff_prev = compute_diff(label_grid, pred_prev_grid)
    _, num_diff_curr = compute_diff(label_grid, pred_curr_grid)
    
    # Create figure with 1 row, 5 columns
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    
    # 1. Test Input
    visualize_grid(input_grid, axes[0], "Test Input")
    
    # 2. Ground Truth
    visualize_grid(label_grid, axes[1], "Ground Truth", highlight_color='#0074D9')
    
    # 3. Original (K=0) Prediction
    status_orig = "✓" if is_correct_orig else f"✗ ({num_diff_orig} err)"
    color_orig = '#2ECC40' if is_correct_orig else '#FF4136'
    visualize_grid(orig_pred_grid, axes[2], f"Original (K=0)\n{status_orig}", highlight_color=color_orig)
    
    # 4. K Pred (previous)
    status_prev = "✓" if is_correct_prev else f"✗ ({num_diff_prev} err)"
    color_prev = '#2ECC40' if is_correct_prev else '#FF4136'
    visualize_grid(pred_prev_grid, axes[3], f"K={k_prev} Pred\n{status_prev}", highlight_color=color_prev)
    
    # 5. K+1 Pred (current)
    status_curr = "✓" if is_correct_curr else f"✗ ({num_diff_curr} err)"
    color_curr = '#2ECC40' if is_correct_curr else '#FF4136'
    visualize_grid(pred_curr_grid, axes[4], f"K={k_curr} Pred\n{status_curr}", highlight_color=color_curr)
    
    plt.suptitle(f"{puzzle_name} | Errors: {num_diff_prev} → {num_diff_curr} | +Feature {new_feature}", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return {
        'k_prev': int(k_prev),
        'k_curr': int(k_curr),
        'new_feature': int(new_feature),
        'num_diff_prev': int(num_diff_prev),
        'num_diff_curr': int(num_diff_curr),
    }


def run_progressive_ablation(
    model,
    sae,
    batch: Dict,
    batch_float: Dict,
    sae_features: np.ndarray,  # [D, L, M]
    original_preds: np.ndarray,  # [L]
    labels: np.ndarray,  # [L]
    inputs: np.ndarray,  # [L]
    puzzle_name: str,
    puzzle_id: int,
    output_dir: str,
    config: 'AblationConfig',
) -> Dict:
    """
    Run progressive ablation: K=1,2,3,...max_k
    Only save visualization when prediction changes from K to K+1.
    
    Args:
        model: TRM model
        sae: SAE model
        batch: single-sample batch dict
        batch_float: batch with float dtype
        sae_features: [D, L, M] SAE feature activations
        original_preds: original model predictions
        labels: ground truth labels
        inputs: input tokens
        puzzle_name: name of puzzle
        puzzle_id: puzzle identifier
        output_dir: directory to save results
        config: ablation configuration
    
    Returns:
        dict with progressive ablation results for this puzzle
    """
    D, L, M = sae_features.shape
    
    # Create puzzle-specific directory
    puzzle_dir = os.path.join(output_dir, puzzle_name)
    os.makedirs(puzzle_dir, exist_ok=True)
    
    # Get ranked features by the specified metric
    ranked_indices, metric_values = rank_features_by_metric(sae_features, labels, config)
    max_k = min(config.max_k_features, len(ranked_indices))
    
    print(f"    Ranking features by: {config.ranking_metric}")
    print(f"    Top 10 features: {list(ranked_indices[:10])}")
    print(f"    Top 10 metric values: {[f'{metric_values[i]:.2f}' for i in ranked_indices[:10]]}")
    
    # Compute original accuracy
    label_grid = decode_arc_grid(labels)
    orig_pred_grid = decode_arc_grid(original_preds)
    is_correct_orig = np.array_equal(label_grid, orig_pred_grid)
    _, orig_num_diff = compute_diff(label_grid, orig_pred_grid)
    
    # Results storage
    all_k_results = []
    change_visualizations = []
    
    # Track previous prediction for comparison
    prev_preds = original_preds.copy()
    prev_k = 0
    prev_features = []
    prev_num_diff = orig_num_diff  # Track previous number of errors
    
    # Record K=0 (original)
    all_k_results.append({
        'k': 0,
        'ablated_features': [],
        'is_correct': bool(is_correct_orig),
        'num_diff': int(orig_num_diff),
        'prediction_changed': False,
    })
    
    print(f"    K=0 (original): {'✓' if is_correct_orig else '✗'} ({orig_num_diff} errors)")

    # Generate visualization for original prediction (K=0)
    orig_filename = f"k00_original.png"
    orig_output_path = os.path.join(puzzle_dir, orig_filename)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    input_grid = decode_arc_grid(inputs)
    visualize_grid(input_grid, axes[0], "Test Input")
    visualize_grid(label_grid, axes[1], "Ground Truth", highlight_color='#0074D9')

    status_orig = "✓" if is_correct_orig else f"✗ ({orig_num_diff} err)"
    color_orig = '#2ECC40' if is_correct_orig else '#FF4136'
    visualize_grid(orig_pred_grid, axes[2], f"Original (K=0)\n{status_orig}", highlight_color=color_orig)

    plt.suptitle(f"{puzzle_name} | Original Prediction", fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(orig_output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    Saved original prediction visualization: {orig_filename}")
    
    # Progressive ablation: K=1, 2, 3, ..., max_k
    for k in range(1, max_k + 1):
        # Get top-k features to ablate
        features_to_ablate = list(ranked_indices[:k])
        new_feature = int(ranked_indices[k-1])  # The newly added feature
        
        # Run ablation
        _, preds_abl, _ = collect_trajectories_with_intervention(
            model, batch_float, sae,
            ablate_features=features_to_ablate,
            ablate_iterations=config.ablation_iterations,
            max_steps=16
        )
        
        curr_preds = preds_abl["preds"].cpu().numpy()[0] if preds_abl and "preds" in preds_abl else labels
        
        # Compute accuracy
        curr_pred_grid = decode_arc_grid(curr_preds)
        is_correct_curr = np.array_equal(label_grid, curr_pred_grid)
        _, curr_num_diff = compute_diff(label_grid, curr_pred_grid)
        
        # Check if NUMBER OF ERRORS changed (not just grid difference)
        error_count_changed = (curr_num_diff != prev_num_diff)
        
        # Record result (convert numpy types to Python types for JSON)
        k_result = {
            'k': int(k),
            'ablated_features': [int(f) for f in features_to_ablate],
            'new_feature': int(new_feature),
            'new_feature_metric': float(metric_values[new_feature]),
            'is_correct': bool(is_correct_curr),
            'num_diff': int(curr_num_diff),
            'error_count_changed': bool(error_count_changed),
        }
        all_k_results.append(k_result)
        
        # Status indicator
        change_marker = f" *ERRORS CHANGED: {prev_num_diff}→{curr_num_diff}*" if error_count_changed else ""
        print(f"    K={k}: {'✓' if is_correct_curr else '✗'} ({curr_num_diff} errors) [+feat {new_feature}]{change_marker}")
        
        # Save visualization ONLY if number of errors changed
        if error_count_changed:
            filename = f"k{prev_k:02d}_to_k{k:02d}_feat{new_feature:04d}.png"
            output_path = os.path.join(puzzle_dir, filename)
            
            viz_result = visualize_progressive_ablation(
                test_input=inputs,
                test_label=labels,
                original_preds=original_preds,
                pred_prev=prev_preds,
                pred_curr=curr_preds,
                k_prev=prev_k,
                k_curr=k,
                new_feature=new_feature,
                puzzle_name=puzzle_name,
                puzzle_id=puzzle_id,
                output_path=output_path,
            )
            change_visualizations.append(viz_result)
            
            # Stop if we've reached max_error_changes
            if config.max_error_changes > 0 and len(change_visualizations) >= config.max_error_changes:
                print(f"    Reached max_error_changes={config.max_error_changes}, stopping.")
                break
        
        # ALWAYS update previous state to compare K with K-1
        prev_preds = curr_preds.copy()
        prev_k = k
        prev_features = [int(f) for f in features_to_ablate]
        prev_num_diff = curr_num_diff  # Track previous error count
    
    # Save puzzle-level summary (convert all numpy types to Python types)
    puzzle_summary = {
        'puzzle_name': puzzle_name,
        'puzzle_id': int(puzzle_id),
        'ranking_metric': config.ranking_metric,
        'max_k_tested': int(max_k),
        'original_correct': bool(is_correct_orig),
        'original_diff': int(orig_num_diff),
        'ranked_features': [int(i) for i in ranked_indices[:max_k]],
        'metric_values_top_k': [float(metric_values[i]) for i in ranked_indices[:max_k]],
        'all_k_results': all_k_results,  # Already properly typed
        'num_changes': len(change_visualizations),
        'change_visualizations': change_visualizations,
    }
    
    summary_path = os.path.join(puzzle_dir, "progressive_ablation_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(puzzle_summary, f, indent=2)
    
    print(f"    Completed: {len(change_visualizations)} prediction changes visualized (out of {max_k} ablations)")
    print(f"    Results saved to: {puzzle_dir}")
    
    return puzzle_summary


@hydra.main(config_path="config", config_name="cfg_eval", version_base=None)
def main(hydra_config: DictConfig):
    """Main ablation study function"""
    # Build config
    arch_dict = dict(hydra_config.arch) if 'arch' in hydra_config else {}
    
    if 'loss' in arch_dict:
        loss_dict = dict(arch_dict.pop('loss'))
        arch_dict['loss'] = LossConfig(**loss_dict)
    else:
        arch_dict['loss'] = LossConfig(name='losses@ACTLossHead')
    
    config = AblationConfig(
        arch=ArchConfig(**arch_dict),
        data_paths=list(hydra_config.data_paths) if 'data_paths' in hydra_config else ['data/arc1concept-aug-1000'],
        data_paths_test=list(hydra_config.get('data_paths_test', [])) if hydra_config.get('data_paths_test') else [],
        global_batch_size=hydra_config.get('global_batch_size', 1),
        load_checkpoint=hydra_config.get('load_checkpoint'),
        sae_checkpoint=hydra_config.get('sae_checkpoint', None),
        sae_model_type=hydra_config.get('sae_model_type', 'sae_fix'),
        ablation_mode=hydra_config.get('ablation_mode', 'top_k'),
        num_features_to_ablate=hydra_config.get('num_features_to_ablate', 10),
        specific_features=list(hydra_config.get('specific_features', [])) if hydra_config.get('specific_features') else None,
        ablation_iterations=list(hydra_config.get('ablation_iterations', [])) if hydra_config.get('ablation_iterations') else None,
        ranking_metric=hydra_config.get('ranking_metric', 'avg_activation'),
        max_k_features=hydra_config.get('max_k_features', 20),
        max_error_changes=hydra_config.get('max_error_changes', 0),
        only_active_features=hydra_config.get('only_active_features', True),
        save_only_on_change=hydra_config.get('save_only_on_change', True),
        puzzle_ids=list(hydra_config.get('puzzle_ids', [])) if hydra_config.get('puzzle_ids') else None,
        num_visualize=hydra_config.get('num_visualize', 10),
        output_dir=hydra_config.get('output_dir', 'ablation_results'),
        top_m_features=hydra_config.get('top_m_features', 50),
        sort_by_iteration=hydra_config.get('sort_by_iteration', 1),
        answer_only=hydra_config.get('answer_only', True),
        seed=hydra_config.get('seed', 0),
    )

    print(f"{'='*60}")
    print("SAE Feature Ablation Study")
    print(f"{'='*60}")
    print(f"SAE Model Type: {config.sae_model_type}")
    print(f"Ablation Mode: {config.ablation_mode}")
    if config.ablation_mode == "all_individually":
        print(f"  - Only active features: {config.only_active_features}")
        print(f"  - Save only on change: {config.save_only_on_change}")
    elif config.ablation_mode == "progressive":
        print(f"  - Ranking Metric: {config.ranking_metric}")
        print(f"  - Max K Features: {config.max_k_features}")
        print(f"  - Max Error Changes: {config.max_error_changes} (0=no limit)")
        print(f"  - Sort by Iteration: {config.sort_by_iteration}")
        print(f"  - Answer Only: {config.answer_only}")
    else:
        print(f"Num Features to Ablate: {config.num_features_to_ablate}")
    print(f"Output directory: {config.output_dir}")
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
    
    # Aggregate statistics
    total_original_correct = 0
    total_ablated_correct = 0
    total_changed = 0
    total_samples = 0
    
    # For all_individually mode
    all_puzzle_summaries = []
    
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
            
            if viz_count >= config.num_visualize:
                break
            
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            batch_float = {k: v.to(dtype=DTYPE) if v.dtype.is_floating_point else v for k, v in batch.items()}
            
            print(f"\n  Processing batch at idx {global_idx}...")
            
            # Step 1: Run ORIGINAL inference (no ablation)
            print(f"    Running original inference...")
            z_L_orig, preds_orig, _ = collect_trajectories_with_intervention(
                model, batch_float, sae,
                ablate_features=None,
                max_steps=16
            )
            z_L_orig = z_L_orig.to(dtype=DTYPE)
            
            # Get SAE features for feature selection
            sae_out = sae(z_L_orig)
            z_n_orig = sae_out['z_n'].float().cpu().numpy()  # [B, D, L, M]
            
            # Process each example in batch
            batch_inputs = batch["inputs"].cpu().numpy()
            batch_labels = batch["labels"].cpu().numpy()
            batch_preds_orig = preds_orig["preds"].cpu().numpy() if preds_orig and "preds" in preds_orig else batch_labels
            batch_puzzle_ids = batch["puzzle_identifiers"].cpu().numpy()
            
            for i in range(batch_size):
                current_idx = global_idx + i
                
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
                
                print(f"\n  [{viz_count+1}/{config.num_visualize}] Processing {base_name} (idx={current_idx})...")
                
                # ========== ALL_INDIVIDUALLY MODE ==========
                if config.ablation_mode == "all_individually":
                    # Create single-sample batch
                    single_batch = {k: v[i:i+1] for k, v in batch.items()}
                    single_batch_float = {k: v.to(dtype=DTYPE) if v.dtype.is_floating_point else v for k, v in single_batch.items()}
                    
                    # Run individual feature ablation
                    puzzle_summary = run_individual_feature_ablation(
                        model=model,
                        sae=sae,
                        batch=single_batch,
                        batch_float=single_batch_float,
                        sae_features=z_n_orig[i],  # [D, L, M]
                        original_preds=batch_preds_orig[i],
                        labels=batch_labels[i],
                        inputs=batch_inputs[i],
                        puzzle_name=base_name,
                        puzzle_id=pid,
                        output_dir=config.output_dir,
                        config=config,
                    )
                    
                    all_puzzle_summaries.append(puzzle_summary)
                    total_samples += 1
                    if puzzle_summary['original_correct']:
                        total_original_correct += 1
                
                # ========== PROGRESSIVE MODE ==========
                elif config.ablation_mode == "progressive":
                    # Create single-sample batch
                    single_batch = {k: v[i:i+1] for k, v in batch.items()}
                    single_batch_float = {k: v.to(dtype=DTYPE) if v.dtype.is_floating_point else v for k, v in single_batch.items()}
                    
                    # Run progressive ablation
                    puzzle_summary = run_progressive_ablation(
                        model=model,
                        sae=sae,
                        batch=single_batch,
                        batch_float=single_batch_float,
                        sae_features=z_n_orig[i],  # [D, L, M]
                        original_preds=batch_preds_orig[i],
                        labels=batch_labels[i],
                        inputs=batch_inputs[i],
                        puzzle_name=base_name,
                        puzzle_id=pid,
                        output_dir=config.output_dir,
                        config=config,
                    )
                    
                    all_puzzle_summaries.append(puzzle_summary)
                    total_samples += 1
                    if puzzle_summary['original_correct']:
                        total_original_correct += 1
                    
                # ========== OTHER MODES (top_k, random, specific, bottom_k) ==========
                else:
                    # Step 2: Select features to ablate
                    features_to_ablate = select_features_to_ablate(
                        z_n_orig[i],  # [D, L, M]
                        batch_labels[i],
                        config
                    )
                    print(f"    Selected {len(features_to_ablate)} features to ablate: {features_to_ablate[:10]}...")
                    
                    # Step 3: Run ABLATED inference
                    print(f"    Running ablated inference...")
                    
                    # Create single-sample batch for ablation
                    single_batch = {k: v[i:i+1] for k, v in batch.items()}
                    single_batch_float = {k: v.to(dtype=DTYPE) if v.dtype.is_floating_point else v for k, v in single_batch.items()}
                    
                    _, preds_abl, ablation_info = collect_trajectories_with_intervention(
                        model, single_batch_float, sae,
                        ablate_features=features_to_ablate,
                        ablate_iterations=config.ablation_iterations,
                        max_steps=16
                    )
                    
                    batch_preds_abl = preds_abl["preds"].cpu().numpy()[0] if preds_abl and "preds" in preds_abl else batch_labels[i]
                    
                    # Step 4: Compare and visualize
                    output_path = os.path.join(config.output_dir, f"ablation_{viz_count:04d}_{base_name}.png")
                    
                    result = visualize_ablation_comparison(
                        test_input=batch_inputs[i],
                        test_label=batch_labels[i],
                        pred_original=batch_preds_orig[i],
                        pred_ablated=batch_preds_abl,
                        ablated_features=features_to_ablate,
                        puzzle_name=base_name,
                        puzzle_id=pid,
                        output_path=output_path,
                    )
                    
                    # Add ablation info to result
                    result['ablation_info'] = ablation_info
                    results.append(result)
                    
                    # Update statistics
                    total_samples += 1
                    if result['is_correct_original']:
                        total_original_correct += 1
                    if result['is_correct_ablated']:
                        total_ablated_correct += 1
                    if result['num_changed_positions'] > 0:
                        total_changed += 1
                    
                    # Print result
                    orig_status = "✓" if result['is_correct_original'] else f"✗ ({result['num_diff_original']} diff)"
                    abl_status = "✓" if result['is_correct_ablated'] else f"✗ ({result['num_diff_ablated']} diff)"
                    print(f"    Original: {orig_status}")
                    print(f"    Ablated:  {abl_status}")
                    print(f"    Changed:  {result['num_changed_positions']} positions")
                
                viz_count += 1
            
            global_idx += batch_size
            
            if puzzle_ids_set is not None and global_idx > max(puzzle_ids_set):
                break
    
    # Save results summary
    if config.ablation_mode == "all_individually":
        # Aggregate stats from all puzzles
        total_improved = sum(p['improved_count'] for p in all_puzzle_summaries)
        total_degraded = sum(p['degraded_count'] for p in all_puzzle_summaries)
        total_unchanged = sum(p['unchanged_count'] for p in all_puzzle_summaries)
        total_features_tested = sum(p['total_features_tested'] for p in all_puzzle_summaries)
        
        summary = {
            'config': {
                'ablation_mode': config.ablation_mode,
                'sae_model_type': config.sae_model_type,
                'only_active_features': config.only_active_features,
                'save_only_on_change': config.save_only_on_change,
                'answer_only': config.answer_only,
            },
            'statistics': {
                'total_puzzles': len(all_puzzle_summaries),
                'original_correct': total_original_correct,
                'total_features_tested': total_features_tested,
                'total_improved': total_improved,
                'total_degraded': total_degraded,
                'total_unchanged': total_unchanged,
            },
            'puzzle_summaries': [
                {k: v for k, v in p.items() if k != 'feature_results'}  # Exclude detailed results
                for p in all_puzzle_summaries
            ]
        }
    elif config.ablation_mode == "progressive":
        # Aggregate stats from progressive ablation
        total_changes = sum(p['num_changes'] for p in all_puzzle_summaries)
        total_k_tested = sum(p['max_k_tested'] for p in all_puzzle_summaries)
        
        summary = {
            'config': {
                'ablation_mode': config.ablation_mode,
                'sae_model_type': config.sae_model_type,
                'ranking_metric': config.ranking_metric,
                'max_k_features': config.max_k_features,
                'sort_by_iteration': config.sort_by_iteration,
                'answer_only': config.answer_only,
            },
            'statistics': {
                'total_puzzles': len(all_puzzle_summaries),
                'original_correct': total_original_correct,
                'total_k_tested': total_k_tested,
                'total_prediction_changes': total_changes,
                'avg_changes_per_puzzle': total_changes / max(1, len(all_puzzle_summaries)),
            },
            'puzzle_summaries': [
                {k: v for k, v in p.items() if k != 'all_k_results'}  # Exclude detailed k results
                for p in all_puzzle_summaries
            ]
        }
    else:
        summary = {
            'config': {
                'ablation_mode': config.ablation_mode,
                'num_features_to_ablate': config.num_features_to_ablate,
                'sae_model_type': config.sae_model_type,
                'answer_only': config.answer_only,
                'sort_by_iteration': config.sort_by_iteration,
            },
            'statistics': {
                'total_samples': total_samples,
                'original_correct': total_original_correct,
                'ablated_correct': total_ablated_correct,
                'samples_changed': total_changed,
                'original_accuracy': total_original_correct / max(1, total_samples),
                'ablated_accuracy': total_ablated_correct / max(1, total_samples),
                'change_rate': total_changed / max(1, total_samples),
            },
            'results': results
        }
    
    summary_path = os.path.join(config.output_dir, "ablation_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print("ABLATION STUDY COMPLETE")
    print(f"{'='*60}")
    
    if config.ablation_mode == "all_individually":
        print(f"Total puzzles: {len(all_puzzle_summaries)}")
        print(f"Original correct: {total_original_correct}/{len(all_puzzle_summaries)}")
        print(f"Total features tested: {total_features_tested}")
        print(f"  Improved predictions: {total_improved}")
        print(f"  Degraded predictions: {total_degraded}")
        print(f"  Unchanged predictions: {total_unchanged}")
    elif config.ablation_mode == "progressive":
        print(f"Total puzzles: {len(all_puzzle_summaries)}")
        print(f"Original correct: {total_original_correct}/{len(all_puzzle_summaries)}")
        print(f"Ranking metric: {config.ranking_metric}")
        print(f"Max K tested: {config.max_k_features}")
        print(f"Total K ablations: {total_k_tested}")
        print(f"Total prediction changes: {total_changes}")
        print(f"Avg changes per puzzle: {total_changes / max(1, len(all_puzzle_summaries)):.2f}")
    else:
        print(f"Total samples: {total_samples}")
        print(f"Original accuracy: {total_original_correct}/{total_samples} ({100*total_original_correct/max(1,total_samples):.1f}%)")
        print(f"Ablated accuracy:  {total_ablated_correct}/{total_samples} ({100*total_ablated_correct/max(1,total_samples):.1f}%)")
        print(f"Samples with changes: {total_changed}/{total_samples} ({100*total_changed/max(1,total_samples):.1f}%)")
    
    print(f"\nOutput directory: {config.output_dir}")
    print(f"Summary saved to: {summary_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
