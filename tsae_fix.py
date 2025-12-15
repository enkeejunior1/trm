from typing import Optional, List, Any
from dataclasses import dataclass

import os
import yaml
import torch
from datetime import datetime
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import tqdm
import hydra
import pydantic
import wandb
from omegaconf import DictConfig
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from utils_sae import (
    LossConfig, ArchConfig, EvalConfig, EvalState, 
    create_dataloader, create_model, init_eval_state, load_checkpoint, evaluate
)

from einops import rearrange

# Global dtype configuration
DTYPE = torch.bfloat16

class Attn(nn.Module):
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

    def forward(self, zL, attn_mask=None):
        """
        zL: [batch_size, seq_len, d_model]
        attn_mask: [batch_size, seq_len] boolean mask (True = valid, False = PAD)
        """
        batch_size, seq_len, d_model = zL.shape
        
        # Project to Q, K, V
        q = self.q_proj(zL)  # [batch, seq_len, d_model]
        k = self.k_proj(zL)  # [batch, seq_len, d_model]
        v = self.v_proj(zL)  # [batch, seq_len, d_model]
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Create attention mask for Flash Attention
        # attn_mask needs to be [batch, 1, 1, seq_len] or [batch, 1, seq_len, seq_len]
        # Where True = attend, False = mask out
        mask_for_attn = None
        if attn_mask is not None and not self.is_causal:
            # For non-causal: mask out PAD tokens in key/value
            # Shape: [batch, 1, 1, seq_len] - broadcast to all heads and queries
            mask_for_attn = attn_mask[:, None, None, :]  # [B, 1, 1, L]
        
        # Use Flash Attention (memory efficient)
        # F.scaled_dot_product_attention automatically uses Flash Attention if available
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask_for_attn,
            dropout_p=0.0,
            is_causal=self.is_causal
        )  # [B, H, T, head_dim]
        
        # Reshape back
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        # Final projection
        out = self.o_proj(out)
        
        return out


class TSAE(nn.Module):
    """
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
        # --- AuxK-related parameters ---
        auxk_topk: int = 512,            # k_aux from the paper
        aux_alpha: float = 1.0 / 32.0,   # L + alpha * L_aux
        dead_token_threshold: int = 200_000,  # How many tokens of inactivity before considered dead
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

        # Store AuxK configuration
        self.auxk_topk = auxk_topk
        self.aux_alpha = aux_alpha
        self.dead_token_threshold = dead_token_threshold

        # 1) Spatial attention (along L dimension)
        self.attn_l = Attn(H, n_heads, is_causal=False)
        self.norm_l = nn.LayerNorm(H)

        # 2) Depth attention (along D dimension, causal)
        self.attn_d = Attn(H, n_heads, is_causal=True)
        self.norm_d = nn.LayerNorm(H)

        # SAE dictionary: H -> M (encoder), M -> H (decoder)
        # F.linear expects weight of shape (out_features, in_features)
        # Encoder: (*, H) -> (*, M), so weight should be (M, H)
        # Decoder: (*, M) -> (*, H), so weight should be (H, M)
        self.dictionary_enc = nn.Parameter(torch.randn(n_features, H, dtype=DTYPE) * (2.0 / (H + n_features)) ** 0.5)
        self.dictionary_dec = nn.Parameter(torch.randn(H, n_features, dtype=DTYPE) * (2.0 / (n_features + H)) ** 0.5)
        
        # Initialize decoder as transpose of encoder
        self.dictionary_dec.data.copy_(self.dictionary_enc.data.T)
        self.bias_pre = nn.Parameter(torch.zeros(H, dtype=DTYPE))
        self.bias_enc = nn.Parameter(torch.zeros(n_features, dtype=DTYPE))

        # 3) Query token
        self.query_token = nn.Parameter(torch.zeros(H, dtype=DTYPE))

        # ---- AuxK statistics buffers ----
        # Total number of tokens processed so far
        self.register_buffer("num_tokens_seen", torch.zeros((), dtype=torch.long))
        # Last activation time (token index) for each latent
        self.register_buffer("last_active_token", torch.zeros(n_features, dtype=torch.long))


    @torch.no_grad()
    def normalize_dic(self):
        self.dictionary_dec.data = self.dictionary_dec.data / self.dictionary_dec.data.norm(dim=0, keepdim=True)

    def topk_activation(self, x):
        """
        x: (..., M)
        Keep only TopK activations, set the rest to 0
        """
        if self.topk is None or self.topk >= x.size(-1):
            return x
        values, indices = torch.topk(x, self.topk, dim=-1)
        out = torch.zeros_like(x)
        out.scatter_(dim=-1, index=indices, src=values)
        return out

    def forward(self, zL, mask=None):
        """
        zL: [B, D, L, H]
        mask: [B, L] boolean mask where True = valid token, False = ignore (PAD)
        return: dict (includes loss)
        """
        B, D, L, H = zL.shape
        assert D == self.depth, f"depth mismatch: got {D}, expected {self.depth}"
        
        # If no mask provided, use all tokens
        if mask is None:
            mask = torch.ones(B, L, dtype=torch.bool, device=zL.device)
        else:
            # Adjust mask to match zL's sequence length (handle register tokens)
            L_data = mask.shape[1]
            if L != L_data:
                # Pad mask with True (valid) for additional tokens (e.g., register tokens)
                mask_padded = torch.ones(B, L, dtype=torch.bool, device=mask.device)
                mask_padded[:, :L_data] = mask
                mask = mask_padded

        # Check for NaN/Inf in input
        if torch.isnan(zL).any() or torch.isinf(zL).any():
            for d in range(D):
                layer_data = zL[:, d, :, :]  # [B, L, H]
                print(f"Layer {d}: min={layer_data.min().item():.4f}, max={layer_data.max().item():.4f}, "
                    f"mean={layer_data.mean().item():.4f}, std={layer_data.std().item():.4f}, "
                    f"NaN count: {torch.isnan(layer_data).sum().item()}, "
                    f"Inf count: {torch.isinf(layer_data).sum().item()}")
            raise ValueError(
                f"NaN or Inf detected in input zL!\n"
                f"  zL stats: min={zL.min().item():.4f}, max={zL.max().item():.4f}, mean={zL.mean().item():.4f}\n"
                f"  NaN count: {torch.isnan(zL).sum().item()}, Inf count: {torch.isinf(zL).sum().item()}"
            )

        # -------- 1) L-axis self-attention (for each depth) --------
        x = torch.cat([self.query_token[None, None, None, :].expand(B, 1, L, H), zL[:, :-1, :, :]], dim=1)
        x = x.view(B * D, L, H)
        # Repeat mask for all depths
        mask_repeated = mask.unsqueeze(1).expand(B, D, L).reshape(B * D, L)  # [B*D, L]
        x = x + self.attn_l(self.norm_l(x), attn_mask=mask_repeated)  # [B*D, L, H]
        x = x.view(B, D, L, H)   # [B, D, L, H]

        # -------- 2) D-axis causal self-attention (for each token position) --------
        # [B, D, L, H] -> [B*L, D, H]
        # No need to mask here as we process all (B*L) positions
        x = rearrange(x, 'b d l h -> (b l) d h')
        x = x + self.attn_d(self.norm_d(x))  # attn_d is causal, no extra mask needed
        x_prior = rearrange(x, '(b l) d h -> b d l h', b=B, l=L)  # [B, D, L, H]

        # prediction from attn - apply mask
        mask_expanded = mask.unsqueeze(1).unsqueeze(-1).expand(B, D, L, H)  # [B, D, L, H]
        squared_error_attn = (x_prior - zL) ** 2
        masked_squared_error_attn = squared_error_attn * mask_expanded
        num_valid = mask_expanded.sum()
        recon_loss_attn = masked_squared_error_attn.sum() / num_valid.clamp(min=1)

        # Process in chunks to avoid OOM (chunk size: 256 samples at a time)
        chunk_size = 256
        N = B * D * L
        x_src = (zL - x_prior).detach().reshape(N, H)    # [N, H]
        
        # encoder: x -> logits -> ReLU -> TopK
        logits = F.linear(x_src - self.bias_pre[None, :], self.dictionary_enc, self.bias_enc)  # [N, M]
        z_n_dense = F.relu(logits)
        z_n_flat = self.topk_activation(z_n_dense) 

        # decoder: z_n -> x_novel
        x_tgt_flat = F.linear(z_n_flat, self.dictionary_dec) + self.bias_pre[None, :]
        x_tgt = x_tgt_flat.view(B, D, L, H)
        z_n = z_n_flat.view(B, D, L, self.n_features)

        # ---- Update activation tracking for AuxK ----
        with torch.no_grad():
            # Count tokens processed in this batch (N = B * D * L)
            tokens_this_batch = z_n_flat.size(0)  # N
            self.num_tokens_seen += tokens_this_batch
            
            # Track which latents were active (> 0) in this batch
            # z_n_flat: [N, M] -> bool[M]
            active_mask = (z_n_flat > 0).any(dim=0)
            
            # Update last_active_token for active latents to current token count
            self.last_active_token[active_mask] = self.num_tokens_seen

        # Check for NaN/Inf in intermediate outputs
        if torch.isnan(x_tgt).any() or torch.isinf(x_tgt).any():
            raise ValueError(
                f"NaN or Inf detected in x_tgt (decoder output)!\n"
                f"  x_tgt stats: min={x_tgt.min().item():.4f}, max={x_tgt.max().item():.4f}, mean={x_tgt.mean().item():.4f}\n"
                f"  z_n stats: min={z_n.min().item():.4f}, max={z_n.max().item():.4f}, mean={z_n.mean().item():.4f}\n"
                f"  dictionary_dec stats: min={self.dictionary_dec.min().item():.4f}, max={self.dictionary_dec.max().item():.4f}"
            )

        # -------- 5) Main Loss Calculation --------
        # recon_loss: how well SAE reconstructs the residual
        # Apply mask to only compute loss on valid (non-PAD) tokens
        x_src_reshaped = x_src.view(B, D, L, H)
        
        # Compute MSE only on valid tokens
        squared_error = (x_tgt - x_src_reshaped) ** 2
        masked_squared_error = squared_error * mask_expanded
        recon_loss = masked_squared_error.sum() / num_valid.clamp(min=1)
        
        loss_main = recon_loss_attn + recon_loss
        
        # -------- 6) AuxK Loss Calculation --------
        aux_loss = torch.zeros((), device=zL.device, dtype=zL.dtype)
        if self.auxk_topk is not None and self.auxk_topk > 0:
            # Find dead latents (those that haven't been active for a long time)
            with torch.no_grad():
                # How long has each latent been inactive (in tokens)?
                ages = self.num_tokens_seen - self.last_active_token  # [M]
                dead_mask = ages >= self.dead_token_threshold
                dead_indices = dead_mask.nonzero(as_tuple=False).squeeze(-1)  # [Nd] or []
            
            if dead_indices.numel() > 0:
                # Among dead latents, sort by age (oldest first)
                dead_ages = ages[dead_indices].float()
                k_aux = min(self.auxk_topk, dead_indices.numel())
                _, sort_idx = torch.sort(dead_ages, descending=True)
                selected = dead_indices[sort_idx[:k_aux]]  # [k_aux]
                
                # Main model reconstruction error e = x_src - x_tgt
                # Detach e so gradients only flow through aux dictionary
                e = (x_src_reshaped - x_tgt).detach()  # [B, D, L, H]
                e_flat = e.view(-1, H)  # [N, H]
                
                # Use only the dead latent subset for encoding/decoding
                W_enc_dead = self.dictionary_enc[selected]          # [k_aux, H]
                b_enc_dead = self.bias_enc[selected]                # [k_aux]
                W_dec_dead = self.dictionary_dec[:, selected]       # [H, k_aux]
                
                # Encoder: e -> logits_aux -> ReLU
                logits_aux = F.linear(e_flat - self.bias_pre[None, :],
                                      W_enc_dead,
                                      b_enc_dead)                  # [N, k_aux]
                z_aux = F.relu(logits_aux)
                
                # Decoder: z_aux -> e_hat
                e_hat_flat = F.linear(z_aux, W_dec_dead)            # [N, H]
                e_hat = e_hat_flat.view_as(e)                       # [B, D, L, H]
                
                # Apply mask to aux loss as well
                squared_error_aux = (e_hat - e) ** 2
                masked_squared_error_aux = squared_error_aux * mask_expanded
                aux_loss = masked_squared_error_aux.sum() / num_valid.clamp(min=1)
        
        # Replace NaN/Inf in aux_loss with 0 (as per paper)
        if torch.isnan(aux_loss) or torch.isinf(aux_loss):
            aux_loss = torch.zeros_like(aux_loss)
        
        # -------- 7) Total Loss --------
        loss = loss_main + self.aux_alpha * aux_loss
        
        # NaN/Inf check - raise error to stop training if main loss has issues
        if torch.isnan(loss) or torch.isinf(loss):
            raise ValueError(
                f"NaN or Inf detected in loss!\n"
                f"  recon_loss: {recon_loss.item()}\n"
                f"  aux_loss: {aux_loss.item()}\n"
                f"  total_loss: {loss.item()}\n"
                f"  x_tgt stats: min={x_tgt.min().item():.4f}, max={x_tgt.max().item():.4f}, mean={x_tgt.mean().item():.4f}\n"
                f"  x_src stats: min={x_src_reshaped.min().item():.4f}, max={x_src_reshaped.max().item():.4f}, mean={x_src_reshaped.mean().item():.4f}\n"
            )

        return {
            "loss": loss,
            "recon_loss": recon_loss.detach(),
            "recon_loss_attn": recon_loss_attn.detach(),
            "aux_loss": aux_loss.detach(),
            "x_prior": x_prior,
            "x_tgt": x_tgt,
            "x_src": x_src_reshaped,
            "z_n": z_n,
        }

def validate_sae(
    sae: TSAE,
    eval_state: EvalState,
    val_loader: torch.utils.data.DataLoader,
    config: DictConfig,
) -> dict:
    """Run validation on a single batch"""
    sae.eval()
    
    with torch.no_grad():
        # Get first batch only
        for set_name, batch, global_batch_size in val_loader:
            # To device and dtype
            batch = {k: v.to(device='cuda', dtype=DTYPE if v.dtype.is_floating_point else v.dtype) for k, v in batch.items()}
            
            # Create mask from labels only
            # We want to process all tokens (including PAD in inputs) since model generates hidden states for them
            # Only exclude tokens that shouldn't be predicted (IGNORE_LABEL_ID in labels)
            IGNORE_LABEL_ID = -100
            
            mask = (batch['labels'] != IGNORE_LABEL_ID)
            
            with torch.device("cuda"):
                carry = eval_state.model.initial_carry(batch)  # type: ignore

            # Store trajectories for this batch
            batch_trajectories_L = []
            
            # Forward pass - collect z_L at each step
            # Force max iterations for SAE training
            max_steps = config.arch.halt_max_steps
            for step in range(max_steps):
                assert hasattr(carry, 'inner_carry') and hasattr(carry.inner_carry, 'z_L')
                batch_trajectories_L.append(carry.inner_carry.z_L.detach().clone())
                
                carry, loss, metrics, preds, all_finish = eval_state.model(
                    carry=carry, batch=batch, return_keys=set()
                )
            
            # Collect final state
            assert hasattr(carry, 'inner_carry') and hasattr(carry.inner_carry, 'z_L')
            batch_trajectories_L.append(carry.inner_carry.z_L.detach().clone())

            z_L = torch.stack(batch_trajectories_L, dim=1)[:, 1:, ...]
            del batch_trajectories_L
            
            # Convert to DTYPE
            z_L = z_L.to(dtype=DTYPE)
            
            # Adjust mask to match z_L's sequence length (handle register tokens)
            B_full, D_full, L_model, H_full = z_L.shape
            L_data = mask.shape[1]
            if L_model != L_data:
                # Pad mask with True (valid) for additional tokens (e.g., register tokens)
                mask_padded = torch.ones(B_full, L_model, dtype=torch.bool, device=mask.device)
                mask_padded[:, :L_data] = mask
                mask = mask_padded
            
            # Process in mini-batches to avoid OOM
            mini_batch_size = 4  # Smaller for validation to be safe
            num_mini_batches = (B_full + mini_batch_size - 1) // mini_batch_size
            
            total_loss = 0.0
            total_recon_loss = 0.0
            total_recon_loss_attn = 0.0
            total_aux_loss = 0.0
            
            for mini_idx in range(num_mini_batches):
                start_idx = mini_idx * mini_batch_size
                end_idx = min(start_idx + mini_batch_size, B_full)
                
                z_L_mini = z_L[start_idx:end_idx]
                mask_mini = mask[start_idx:end_idx]  # [mini_batch, L]
                
                # Forward pass with mask
                out = sae(z_L_mini, mask=mask_mini)
                
                # Accumulate metrics
                total_loss += out['loss'].item() / num_mini_batches
                total_recon_loss += out['recon_loss'].item() / num_mini_batches
                total_recon_loss_attn += out['recon_loss_attn'].item() / num_mini_batches
                total_aux_loss += out['aux_loss'].item() / num_mini_batches
                
                del z_L_mini, out
            
            val_metrics = {
                'val_loss': total_loss,
                'val_recon_loss': total_recon_loss,
                'val_recon_loss_attn': total_recon_loss_attn,
                'val_aux_loss': total_aux_loss,
            }
            
            del z_L
            
            # Only process first batch
            break
    
    sae.train()
    return val_metrics


def train(
    config: DictConfig,
    eval_state: EvalState,
    eval_loader: torch.utils.data.DataLoader,
    val_loader_factory,  # Function to create a new val_loader each epoch
    num_epochs: int = 10,
):
    # Model hyperparameters
    depth, length, d_model = 16, 916, 512
    n_heads = 16
    n_features = 4096
    topk = 64
    n_registers = 4
    auxk_topk = 512
    aux_alpha = 1.0 / 32.0
    dead_token_threshold = 200_000
    
    # Create experiment directory with timestamp and hyperparameters
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"tsae_d{depth}_h{n_heads}_f{n_features}_k{topk}_auxk{auxk_topk}_{timestamp}"
    
    # Update checkpoint path with experiment name
    base_path = config.checkpoint_path.replace('ckpt/', '')
    results_path = os.path.join(base_path, 'tsae_runs', exp_name, 'weights')
    os.makedirs(results_path, exist_ok=True)
    print(f"Experiment directory: {results_path}")
    
    # Initialize wandb with experiment name
    wandb.init(
        project="trm_tsae",
        name=exp_name,
        config={
            "depth": depth,
            "d_model": d_model,
            "n_heads": n_heads,
            "n_features": n_features,
            "topk": topk,
            "n_registers": n_registers,
            "auxk_topk": auxk_topk,
            "aux_alpha": aux_alpha,
            "dead_token_threshold": dead_token_threshold,
        }
    )
    
    # Freeze TRM model to save memory and prevent gradient computation
    eval_state.model.eval()
    for param in eval_state.model.parameters():
        param.requires_grad = False
    print("TRM model frozen - all parameters set to requires_grad=False")
    
    sae = TSAE(
        d_model=d_model,
        depth=depth,
        n_heads=n_heads,
        n_features=n_features,
        topk=topk,
        lambda_sparse=1e-3,
        n_registers=n_registers,
        auxk_topk=auxk_topk,
        aux_alpha=aux_alpha,
        dead_token_threshold=dead_token_threshold,
    ).to(device='cuda', dtype=DTYPE)

    print(f"SAE initialized with {n_registers} register tokens for spatial attention")
    print(f"Model parameters: {sum(p.numel() for p in sae.parameters()) / 1e6:.2f}M")
    
    # Use AdamW with weight decay for stability
    sae_optim = torch.optim.AdamW(sae.parameters(), lr=3e-5, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)
    sae.train()

    """Run multi-epoch training"""
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*60}\n")
        
        epoch_losses = []
        epoch_recon_losses = []
        epoch_recon_losses_attn = []
        epoch_aux_losses = []
        processed_batches = 0
        
        try:
            for set_name, batch, global_batch_size in eval_loader:
                processed_batches += 1
                try:
                    with torch.no_grad():
                        # To device and dtype
                        batch = {k: v.to(device='cuda', dtype=DTYPE if v.dtype.is_floating_point else v.dtype) for k, v in batch.items()}
                        
                        # Create mask from labels only
                        # We want to process all tokens (including PAD in inputs) since model generates hidden states for them
                        # Only exclude tokens that shouldn't be predicted (IGNORE_LABEL_ID in labels)
                        IGNORE_LABEL_ID = -100
                        
                        mask = (batch['labels'] != IGNORE_LABEL_ID)
                        
                        with torch.device("cuda"):
                            carry = eval_state.model.initial_carry(batch)  # type: ignore

                        # Store trajectories for this batch
                        batch_trajectories_L = []
                        batch_trajectories_H = []
                        
                        # Forward pass - collect z_L and z_H at each step
                        # Force max iterations for SAE training
                        max_steps = config.arch.halt_max_steps
                        pbar = tqdm.tqdm(total=max_steps, desc=f"Epoch {epoch+1} - Batch {processed_batches}")
                        for step in range(max_steps):
                            # Save z_L and z_H at each inference step (BEFORE forward pass)
                            # Detach and clone to prevent memory leak
                            if not (hasattr(carry, 'inner_carry') and hasattr(carry.inner_carry, 'z_L') and hasattr(carry.inner_carry, 'z_H')):
                                raise AttributeError(f"carry.inner_carry missing z_L or z_H at step {step}, epoch {epoch+1}, batch {processed_batches}")
                            batch_trajectories_L.append(carry.inner_carry.z_L.detach().clone())
                            batch_trajectories_H.append(carry.inner_carry.z_H.detach().clone())
                            
                            carry, loss, metrics, preds, all_finish = eval_state.model(
                                carry=carry, batch=batch, return_keys=set()
                            )
                            pbar.update(1)
                        
                        # Save final z_L and z_H after last step
                        # Detach and clone to prevent memory leak
                        if not (hasattr(carry, 'inner_carry') and hasattr(carry.inner_carry, 'z_L') and hasattr(carry.inner_carry, 'z_H')):
                            raise AttributeError(f"carry.inner_carry missing z_L or z_H at final step, epoch {epoch+1}, batch {processed_batches}")
                        batch_trajectories_L.append(carry.inner_carry.z_L.detach().clone())
                        batch_trajectories_H.append(carry.inner_carry.z_H.detach().clone())

                        pbar.close()

                        z_L = torch.stack(batch_trajectories_L, dim=1)[:, 1:, ...]
                        z_H = torch.stack(batch_trajectories_H, dim=1)[:, 1:, ...]
                        
                        # Clear trajectory lists to free memory
                        del batch_trajectories_L, batch_trajectories_H

                    # Convert to DTYPE (already detached and cloned)
                    z_L = z_L.to(dtype=DTYPE)
                    
                    # Check for NaN/Inf before processing
                    if torch.isnan(z_L).any() or torch.isinf(z_L).any():
                        print(f"  Warning: NaN/Inf detected in z_L at epoch {epoch+1}, batch {processed_batches}")
                        print(f"    NaN count: {torch.isnan(z_L).sum().item()}, Inf count: {torch.isinf(z_L).sum().item()}")
                        # Skip this batch
                        del z_L, z_H
                        continue
                    
                    # Adjust mask to match z_L's sequence length (handle register tokens)
                    # z_L shape: [B, D, L_model, H], mask shape: [B, L_data]
                    B_full, D_full, L_model, H_full = z_L.shape
                    L_data = mask.shape[1]
                    if L_model != L_data:
                        # Pad mask with True (valid) for additional tokens (e.g., register tokens)
                        mask_padded = torch.ones(B_full, L_model, dtype=torch.bool, device=mask.device)
                        mask_padded[:, :L_data] = mask
                        mask = mask_padded
                    
                    # Process in mini-batches to save memory (gradient accumulation)
                    mini_batch_size = 8  # Process 8 samples at a time (reduced for stability)
                    num_mini_batches = (B_full + mini_batch_size - 1) // mini_batch_size
                    
                    sae_optim.zero_grad()
                    total_loss = 0.0
                    total_recon_loss = 0.0
                    total_recon_loss_attn = 0.0
                    total_aux_loss = 0.0
                    
                    for mini_idx in range(num_mini_batches):
                        start_idx = mini_idx * mini_batch_size
                        end_idx = min(start_idx + mini_batch_size, B_full)
                        
                        z_L_mini = z_L[start_idx:end_idx].requires_grad_(True)
                        mask_mini = mask[start_idx:end_idx]  # [mini_batch, L]
                        
                        # Forward pass with mask
                        try:
                            out = sae(z_L_mini, mask=mask_mini)
                            sae_loss = out['loss'] / num_mini_batches  # Scale loss for accumulation
                            
                            # Check for NaN/Inf in loss
                            if torch.isnan(sae_loss) or torch.isinf(sae_loss):
                                print(f"  Warning: NaN/Inf in loss at epoch {epoch+1}, batch {processed_batches}, mini_batch {mini_idx}")
                                del z_L_mini, out, sae_loss
                                break
                            
                            # Backward pass
                            sae_loss.backward()
                            
                            # Accumulate metrics
                            batch_loss = out['loss'].item()
                            batch_recon_loss = out['recon_loss'].item()
                            batch_recon_loss_attn = out['recon_loss_attn'].item()
                            batch_aux_loss = out['aux_loss'].item()
                            
                            total_loss += batch_loss / num_mini_batches
                            total_recon_loss += batch_recon_loss / num_mini_batches
                            total_recon_loss_attn += batch_recon_loss_attn / num_mini_batches
                            total_aux_loss += batch_aux_loss / num_mini_batches
                            
                            # Free memory
                            del z_L_mini, out, sae_loss
                        except Exception as e:
                            print(f"  Error in SAE forward pass at epoch {epoch+1}, batch {processed_batches}, mini_batch {mini_idx}: {e}")
                            import traceback
                            traceback.print_exc()
                            del z_L_mini
                            # Skip this mini-batch
                            continue
                    
                    # Only update if we processed at least one mini-batch successfully
                    if num_mini_batches > 0 and total_loss > 0:
                        # Gradient clipping for stability
                        torch.nn.utils.clip_grad_norm_(sae.parameters(), max_norm=1.0)
                        sae_optim.step()
                        sae.normalize_dic()
                        
                        # Log to wandb
                        wandb.log({
                            "train/loss": total_loss,
                            "train/recon_loss": total_recon_loss,
                            "train/recon_loss_attn": total_recon_loss_attn,
                            "train/aux_loss": total_aux_loss,
                            "train/epoch": epoch,
                            "train/batch": processed_batches,
                        }, step=global_step)
                        
                        global_step += 1

                        print(
                            f"  Batch {processed_batches} | Total: {total_loss:.4f} | "
                            f"recon: {total_recon_loss:.4f}, recon_attn: {total_recon_loss_attn:.4f}, "
                            f"aux: {total_aux_loss:.6f}"
                        )
                        
                        # Accumulate epoch metrics
                        epoch_losses.append(total_loss)
                        epoch_recon_losses.append(total_recon_loss)
                        epoch_recon_losses_attn.append(total_recon_loss_attn)
                        epoch_aux_losses.append(total_aux_loss)
                    
                    # Free z_L and z_H to save memory
                    del z_L, z_H
                    
                    # Save the trained SAE periodically
                    if global_step % 100 == 0:
                        sae_save_path = os.path.join(results_path, f"tsae_step_{global_step:06d}.pt")
                        torch.save({
                            'sae_state_dict': sae.state_dict(),
                            'sae_optim_state_dict': sae_optim.state_dict(),
                            'global_step': global_step,
                            'epoch': epoch,
                        }, sae_save_path)
                        print(f"  Saved TSAE to {sae_save_path}")
                    
                except Exception as e:
                    print(f"  Error processing batch {processed_batches} in epoch {epoch+1}: {e}")
                    import traceback
                    traceback.print_exc()
                    # Continue to next batch
                    continue
        
        except Exception as e:
            print(f"\n  Error during epoch {epoch+1}, batch {processed_batches}: {e}")
            import traceback
            traceback.print_exc()
            # Continue to next epoch instead of crashing
            if len(epoch_losses) == 0:
                print(f"  Warning: No batches processed in epoch {epoch+1}, skipping epoch summary")
                continue
        
        # Epoch summary (after all batches in this epoch)
        if len(epoch_losses) == 0:
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1} Summary: No batches processed")
            print(f"{'='*60}\n")
            continue
            
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Processed batches: {len(epoch_losses)}")
        print(f"  Avg Loss: {sum(epoch_losses)/len(epoch_losses):.4f}")
        print(f"  Avg Recon: {sum(epoch_recon_losses)/len(epoch_recon_losses):.4f}")
        print(f"  Avg Recon Attn: {sum(epoch_recon_losses_attn)/len(epoch_recon_losses_attn):.4f}")
        print(f"  Avg Aux: {sum(epoch_aux_losses)/len(epoch_aux_losses):.6f}")
        print(f"{'='*60}\n")
        
        # *** VALIDATION: Re-create val_loader each epoch for consistent first batch ***
        print(f"\n{'*'*60}")
        print(f"Running Validation at end of Epoch {epoch+1}")
        print(f"{'*'*60}")
        val_metrics = None
        try:
            # Create a fresh val_loader each epoch to ensure consistent first batch
            print(f"  [1/3] Creating fresh validation loader (test split)...")
            val_loader = val_loader_factory()
            
            # Run validation
            print(f"  [2/3] Computing validation metrics on test set...")
            val_metrics = validate_sae(sae, eval_state, val_loader, config)
            
            print(f"  [3/3] ✓ Validation completed!")
            print(f"        Val Loss: {val_metrics['val_loss']:.4f}")
            print(f"        Val Recon Loss: {val_metrics['val_recon_loss']:.4f}")
            print(f"        Val Recon Loss Attn: {val_metrics['val_recon_loss_attn']:.4f}")
            print(f"        Val Aux Loss: {val_metrics['val_aux_loss']:.6f}")
            print(f"{'*'*60}\n")
            
            # Clear val_loader to free memory
            del val_loader
            
            # Save best model if validation loss improved
            current_val_loss = val_metrics['val_loss']
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                best_model_path = os.path.join(results_path, "best_val.pt")
                torch.save({
                    'sae_state_dict': sae.state_dict(),
                    'sae_optim_state_dict': sae_optim.state_dict(),
                    'global_step': global_step,
                    'epoch': epoch + 1,
                    'val_loss': best_val_loss,
                }, best_model_path)
                print(f"  🌟 New best validation loss: {best_val_loss:.4f} - saved to {best_model_path}")
            
        except Exception as e:
            print(f"  ✗ Warning: Validation failed at end of epoch {epoch+1}: {e}")
            import traceback
            traceback.print_exc()
            print(f"{'*'*60}\n")
            val_metrics = None
        
        # Log epoch summary to wandb
        log_dict = {
            "epoch/number": epoch + 1,
            "epoch/avg_loss": sum(epoch_losses)/len(epoch_losses),
            "epoch/avg_recon_loss": sum(epoch_recon_losses)/len(epoch_recon_losses),
            "epoch/avg_recon_loss_attn": sum(epoch_recon_losses_attn)/len(epoch_recon_losses_attn),
            "epoch/avg_aux_loss": sum(epoch_aux_losses)/len(epoch_aux_losses),
            "epoch/processed_batches": len(epoch_losses),
        }
        if val_metrics is not None:
            log_dict.update({
                "epoch/val_loss": val_metrics['val_loss'],
                "epoch/val_recon_loss": val_metrics['val_recon_loss'],
                "epoch/val_recon_loss_attn": val_metrics['val_recon_loss_attn'],
                "epoch/val_aux_loss": val_metrics['val_aux_loss'],
                "epoch/best_val_loss": best_val_loss,
            })
            print(f"  ✓ Logged validation metrics to wandb")
        else:
            print(f"  ⚠ Skipping validation metrics logging for epoch {epoch+1}")
            
        wandb.log(log_dict, step=global_step)
    
    wandb.finish()
    return sae, results_path


@hydra.main(config_path="config", config_name="cfg_eval", version_base=None)
def launch(hydra_config: DictConfig):
    """Main evaluation launch function"""
    # Load config
    config = EvalConfig(**hydra_config)  # type: ignore

    # Seed RNGs to ensure consistency
    torch.random.manual_seed(config.seed)

    # Dataset - Use split from config for training
    assert config.split == "train", f"Training must use train split, but got config.split={config.split}"
    eval_loader, eval_metadata = create_dataloader(
        config, 
        config.split,  # "train" or "test"
        test_set_mode=False, 
        epochs_per_iter=1, 
        global_batch_size=config.global_batch_size
    )
    
    # Create validation loader factory (creates a new loader each epoch for consistent first batch)
    print("Setting up validation loader factory (test split, batch_size=128)...")
    def create_val_loader():
        """Factory function to create a fresh validation loader each epoch"""
        val_loader, _ = create_dataloader(
            config,
            "test",  # Always use test split for validation
            test_set_mode=False,
            epochs_per_iter=1,
            global_batch_size=128  # Fixed batch size for validation
        )
        return val_loader

    # Evaluation state
    eval_state = init_eval_state(config, eval_metadata)

    # Save config
    if config.checkpoint_path is not None:
        os.makedirs(config.checkpoint_path.replace('ckpt/', 'results/'), exist_ok=True)
        config_file = os.path.join(config.checkpoint_path.replace('ckpt/', 'results/'), "eval_config.yaml")
        with open(config_file, "wt") as f:
            yaml.dump(config.model_dump(), f)

    # Run training
    print("Starting TSAE training...")
    eval_state.model.eval()
    tsae, results_path = train(config, eval_state, eval_loader, create_val_loader, num_epochs=config.num_epochs)
    
    # Save final weights
    tsae_save_path = os.path.join(results_path, f"tsae_final.pt")
    torch.save({
        'tsae_state_dict': tsae.state_dict(),
    }, tsae_save_path)
    print(f"Saved final TSAE weights to {tsae_save_path}")

if __name__ == "__main__":
    launch()
