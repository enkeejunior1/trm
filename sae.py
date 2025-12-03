from typing import Optional, List, Any
from dataclasses import dataclass

import os
import yaml
import torch
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

    def forward(self, zL):
        batch_size, seq_len, d_model = zL.shape
        
        # Project to Q, K, V
        q = self.q_proj(zL)  # [batch, seq_len, d_model]
        k = self.k_proj(zL)  # [batch, seq_len, d_model]
        v = self.v_proj(zL)  # [batch, seq_len, d_model]
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Use Flash Attention (memory efficient)
        # F.scaled_dot_product_attention automatically uses Flash Attention if available
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=self.is_causal
        )  # [B, H, T, head_dim]
        
        # Reshape back
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        # Final projection
        out = self.o_proj(out)
        
        return out


import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class SAE(nn.Module):
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

    @torch.no_grad()
    def normalize_dic(self):
        self.dictionary_dec.data = self.dictionary_dec.data / self.dictionary_dec.data.norm(dim=0, keepdim=True)

    def topk_activation(self, x):
        """
        x: (..., M)
        TopK만 남기고 나머지 0
        """
        if self.topk is None or self.topk >= x.size(-1):
            return x
        values, indices = torch.topk(x, self.topk, dim=-1)
        out = torch.zeros_like(x)
        out.scatter_(dim=-1, index=indices, src=values)
        return out

    def forward(self, zL):
        """
        zL: [B, D, L, H]
        return: dict (loss 포함된 버전)
        """
        B, D, L, H = zL.shape
        assert D == self.depth, f"depth mismatch: got {D}, expected {self.depth}"

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

        # Process without chunking for better performance
        N = B * D * L
        x_src = zL.reshape(N, H)    # [N, H]
        
        # encoder: x -> logits -> ReLU -> TopK
        logits = F.linear(x_src - self.bias_pre[None, :], self.dictionary_enc, self.bias_enc)  # [N, M]
        z_n_dense = F.relu(logits)
        z_n_flat = self.topk_activation(z_n_dense)  # [N, M]

        # decoder: z_n -> x_tgt
        x_tgt_flat = F.linear(z_n_flat, self.dictionary_dec)  # [N, H]
        
        x_tgt = x_tgt_flat.view(B, D, L, H)                        # [B, D, L, H]
        z_n = z_n_flat.view(B, D, L, self.n_features)              # [B, D, L, M]

        # Check for NaN/Inf in intermediate outputs
        if torch.isnan(x_tgt).any() or torch.isinf(x_tgt).any():
            raise ValueError(
                f"NaN or Inf detected in x_tgt (decoder output)!\n"
                f"  x_tgt stats: min={x_tgt.min().item():.4f}, max={x_tgt.max().item():.4f}, mean={x_tgt.mean().item():.4f}\n"
                f"  z_n stats: min={z_n.min().item():.4f}, max={z_n.max().item():.4f}, mean={z_n.mean().item():.4f}\n"
                f"  dictionary_dec stats: min={self.dictionary_dec.min().item():.4f}, max={self.dictionary_dec.max().item():.4f}"
            )

        # -------- 5) Loss 계산 --------
        # recon_loss: SAE가 residual을 잘 설명하도록
        x_src_reshaped = x_src.view(B, D, L, H)
        recon_loss = F.mse_loss(x_tgt, x_src_reshaped)
        sparse_loss = z_n.abs().mean()
        
        loss = recon_loss + self.lambda_sparse * sparse_loss
        
        # NaN/Inf check - raise error to stop training
        if torch.isnan(loss) or torch.isinf(loss):
            raise ValueError(
                f"NaN or Inf detected in loss!\n"
                f"  recon_loss: {recon_loss.item()}\n"
                f"  sparse_loss: {sparse_loss.item()}\n"
                f"  total_loss: {loss.item()}\n"
                f"  x_tgt stats: min={x_tgt.min().item():.4f}, max={x_tgt.max().item():.4f}, mean={x_tgt.mean().item():.4f}\n"
                f"  x_src stats: min={x_src_reshaped.min().item():.4f}, max={x_src_reshaped.max().item():.4f}, mean={x_src_reshaped.mean().item():.4f}\n"
            )

        return {
            "loss": loss,
            "recon_loss": recon_loss.detach(),
            "sparse_loss": sparse_loss.detach(),
            "x_tgt": x_tgt,
            "x_src": x_src_reshaped,
            "z_n": z_n,
        }

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

        # 1) spatial attention (L 방향)
        self.attn_l = Attn(H, n_heads, is_causal=False)
        self.norm_l = nn.LayerNorm(H)

        # 2) depth attention (D 방향, causal)
        self.attn_d = Attn(H, n_heads, is_causal=True)
        self.norm_d = nn.LayerNorm(H)

        # 3) SAE dictionary: H -> M
        # Initialize dictionary with Xavier/Glorot initialization for stability
        self.dictionary = nn.Parameter(torch.randn(H, n_features, dtype=DTYPE) * (2.0 / (H + n_features)) ** 0.5)
        self.bias_novel = nn.Parameter(torch.zeros(n_features, dtype=DTYPE))

    def topk_activation(self, x):
        """
        x: (..., M)
        TopK만 남기고 나머지 0
        """
        if self.topk is None or self.topk >= x.size(-1):
            return x
        values, indices = torch.topk(x, self.topk, dim=-1)
        out = torch.zeros_like(x)
        out.scatter_(dim=-1, index=indices, src=values)
        return out

    def forward(self, zL):
        """
        zL: [B, D, L, H]
        return: dict (loss 포함된 버전)
        """
        B, D, L, H = zL.shape
        assert D == self.depth, f"depth mismatch: got {D}, expected {self.depth}"

        # 원본 보관 (optional)
        # orig_zL = zL

        # -------- 1) L-axis self-attention (각 depth별로) with register tokens --------
        # [B, D, L, H] -> [B*D, L, H]
        x = zL.view(B * D, L, H)
        x = x + self.attn_l(self.norm_l(x))  # [B*D, L, H]
        x = x.view(B, D, L, H)   # [B, D, L, H]

        # -------- 2) D-axis causal self-attention (각 토큰 위치별로) --------
        # [B, D, L, H] -> [B*L, D, H]
        # No register tokens for causal attention (keeps it simple and efficient)
        x = rearrange(x, 'b d l h -> (b l) d h')
        x = x + self.attn_d(self.norm_d(x))
        x = rearrange(x, '(b l) d h -> b d l h', b=B, l=L)  # [B, D, L, H]

        # -------- 3) depth 방향 예측: x_{d+1} ~ x_{<=d} --------
        # target: depth 1..D-1
        x_target = x[:, 1:, :, :]      # [B, D-1, L, H]
        # prediction: depth 0..D-2
        x_pred_base = x[:, :-1, :, :]  # [B, D-1, L, H]
        
        # Loss 1: x_pred_base를 직접 학습 (attention이 target 맞추도록)
        pred_loss = F.mse_loss(x_pred_base, x_target)

        # -------- 4) residual에 SAE 적용 (TopK Sparse Code) with chunking --------
        # x_pred_base.detach()로 이미 설명된 부분은 고정
        residual = x_target - x_pred_base.detach()    # [B, D-1, L, H]

        # Process in chunks to avoid OOM (chunk size: 256 samples at a time)
        chunk_size = 256
        N = B * (D) * L
        res_flat = residual.reshape(N, H)    # [N, H]
        
        # Process in chunks
        z_n_chunks = []
        x_novel_chunks = []
        
        for i in range(0, N, chunk_size):
            end_i = min(i + chunk_size, N)
            res_chunk = res_flat[i:end_i]  # [chunk, H]
            
            # encoder: residual -> logits -> ReLU -> TopK
            logits_chunk = F.linear(res_chunk, self.dictionary.t(), self.bias_novel)  # [chunk, M]
            z_n_dense_chunk = F.relu(logits_chunk)
            z_n_chunk = self.topk_activation(z_n_dense_chunk)  # [chunk, M]
            
            # decoder: z_n -> x_novel
            x_novel_chunk = F.linear(z_n_chunk, self.dictionary)  # [chunk, H]
            
            z_n_chunks.append(z_n_chunk)
            x_novel_chunks.append(x_novel_chunk)
        
        # Concatenate chunks
        z_n_flat = torch.cat(z_n_chunks, dim=0)  # [N, M]
        x_novel_flat = torch.cat(x_novel_chunks, dim=0)  # [N, H]
        
        x_novel = x_novel_flat.view(B, D, L, H)                        # [B, D-1, L, H]
        z_n = z_n_flat.view(B, D, L, self.n_features)                  # [B, D-1, L, M]

        # 최종 재구성 (x_pred_base.detach()로 고정된 부분 + SAE의 novel 부분)
        x_hat = x_pred_base.detach() + x_novel                             # [B, D-1, L, H]

        # -------- 5) Loss 계산 --------
        # pred_loss: attention이 target을 직접 맞추도록
        # recon_loss: SAE가 residual을 잘 설명하도록 (pred_loss + recon_loss = 전체 reconstruction)
        recon_loss = F.mse_loss(x_hat, x_target)
        sparse_loss = z_n.abs().mean()
        
        loss = pred_loss + recon_loss + self.lambda_sparse * sparse_loss
        
        # NaN/Inf check - raise error to stop training
        if torch.isnan(loss) or torch.isinf(loss):
            raise ValueError(
                f"NaN or Inf detected in loss!\n"
                f"  pred_loss: {pred_loss.item()}\n"
                f"  recon_loss: {recon_loss.item()}\n"
                f"  sparse_loss: {sparse_loss.item()}\n"
                f"  total_loss: {loss.item()}\n"
                f"  x_pred_base stats: min={x_pred_base.min().item():.4f}, max={x_pred_base.max().item():.4f}, mean={x_pred_base.mean().item():.4f}\n"
                f"  x_novel stats: min={x_novel.min().item():.4f}, max={x_novel.max().item():.4f}, mean={x_novel.mean().item():.4f}\n"
                f"  residual stats: min={residual.min().item():.4f}, max={residual.max().item():.4f}, mean={residual.mean().item():.4f}"
            )

        return {
            "loss": loss,
            "pred_loss": pred_loss.detach(),
            "recon_loss": recon_loss.detach(),
            "sparse_loss": sparse_loss.detach(),
            "x_hat": x_hat,
            "x_target": x_target,
            "x_pred_base": x_pred_base,
            "x_novel": x_novel,
            "z_n": z_n,
        }


def train(
    config: DictConfig,
    eval_state: EvalState,
    eval_loader: torch.utils.data.DataLoader,
    num_epochs: int = 10,
):
    # Initialize wandb
    wandb.init(
        project="trm_sae",
    )
    
    # Freeze TRM model to save memory and prevent gradient computation
    eval_state.model.eval()
    for param in eval_state.model.parameters():
        param.requires_grad = False
    print("TRM model frozen - all parameters set to requires_grad=False")
    
    depth, length, d_model = 16, 916, 512
    n_registers = 4
    sae = SAE(
        d_model=d_model,
        depth=depth,
        n_heads=8,
        n_features=4096,
        topk=64,
        lambda_sparse=1e-3,
        n_registers=n_registers,
    ).to(device='cuda', dtype=DTYPE)

    print(f"SAE initialized with {n_registers} register tokens for spatial attention")
    print(f"Model parameters: {sum(p.numel() for p in sae.parameters()) / 1e6:.2f}M")
    
    # Use AdamW with weight decay for stability
    sae_optim = torch.optim.AdamW(sae.parameters(), lr=3e-5, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)
    sae.train()

    """Run multi-epoch training"""
    global_step = 0
    
    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*60}\n")
        
        epoch_losses = []
        epoch_recon_losses = []
        epoch_sparse_losses = []
        processed_batches = 0
        
        for set_name, batch, global_batch_size in eval_loader:
            processed_batches += 1
            with torch.no_grad():
                # To device and dtype
                batch = {k: v.to(device='cuda', dtype=DTYPE if v.dtype.is_floating_point else v.dtype) for k, v in batch.items()}
                with torch.device("cuda"):
                    carry = eval_state.model.initial_carry(batch)  # type: ignore

                # Store trajectories for this batch
                batch_trajectories_L = []
                batch_trajectories_H = []
                
                # Forward pass - collect z_L and z_H at each step
                inference_steps = 0
                pbar = tqdm.tqdm(desc=f"Epoch {epoch+1} - Batch {processed_batches}")
                while True:
                    # Save z_L and z_H at each inference step (BEFORE forward pass)
                    # Detach and clone to prevent memory leak
                    assert hasattr(carry, 'inner_carry') and (hasattr(carry.inner_carry, 'z_L') and hasattr(carry.inner_carry, 'z_H'))
                    batch_trajectories_L.append(carry.inner_carry.z_L.detach().clone())
                    batch_trajectories_H.append(carry.inner_carry.z_H.detach().clone())
                    
                    carry, loss, metrics, preds, all_finish = eval_state.model(
                        carry=carry, batch=batch, return_keys=set()
                    )
                    inference_steps += 1
                    pbar.update(1)

                    if all_finish:
                        # Save final z_L and z_H after last step
                        # Detach and clone to prevent memory leak
                        assert hasattr(carry, 'inner_carry') and (hasattr(carry.inner_carry, 'z_L') and hasattr(carry.inner_carry, 'z_H'))
                        batch_trajectories_L.append(carry.inner_carry.z_L.detach().clone())
                        batch_trajectories_H.append(carry.inner_carry.z_H.detach().clone())
                        break

                pbar.close()

                z_L = torch.stack(batch_trajectories_L, dim=1)[:, 1:, ...]
                z_H = torch.stack(batch_trajectories_H, dim=1)[:, 1:, ...]
                
                # Clear trajectory lists to free memory
                del batch_trajectories_L, batch_trajectories_H

            # Convert to DTYPE (already detached and cloned)
            z_L = z_L.to(dtype=DTYPE)
            
            # Process in mini-batches to save memory (gradient accumulation)
            mini_batch_size = 8  # Process 8 samples at a time (reduced for stability)
            B_full = z_L.shape[0]
            num_mini_batches = (B_full + mini_batch_size - 1) // mini_batch_size
            
            sae_optim.zero_grad()
            total_loss = 0.0
            total_recon_loss = 0.0
            total_sparse_loss = 0.0
            
            for mini_idx in range(num_mini_batches):
                start_idx = mini_idx * mini_batch_size
                end_idx = min(start_idx + mini_batch_size, B_full)
                
                z_L_mini = z_L[start_idx:end_idx].requires_grad_(True)
                
                # Forward pass
                out = sae(z_L_mini)
                sae_loss = out['loss'] / num_mini_batches  # Scale loss for accumulation
                
                # Backward pass
                sae_loss.backward()
                
                # Accumulate metrics
                batch_loss = out['loss'].item()
                batch_recon_loss = out['recon_loss'].item()
                batch_sparse_loss = out['sparse_loss'].item()
                
                total_loss += batch_loss / num_mini_batches
                total_recon_loss += batch_recon_loss / num_mini_batches
                total_sparse_loss += batch_sparse_loss / num_mini_batches
                
                # Free memory
                del z_L_mini, out, sae_loss
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(sae.parameters(), max_norm=1.0)
            sae_optim.step()
            sae.normalize_dic()
            
            # Log to wandb
            wandb.log({
                "train/loss": total_loss,
                "train/recon_loss": total_recon_loss,
                "train/sparse_loss": total_sparse_loss,
                "train/epoch": epoch,
                "train/batch": processed_batches,
            }, step=global_step)
            
            global_step += 1

            print(
                f"  Batch {processed_batches} | Total: {total_loss:.4f} | "
                f"recon: {total_recon_loss:.4f}, sparse: {total_sparse_loss:.6f}"
            )
            
            # Accumulate epoch metrics
            epoch_losses.append(total_loss)
            epoch_recon_losses.append(total_recon_loss)
            epoch_sparse_losses.append(total_sparse_loss)
            
            # Free z_L and z_H to save memory
            del z_L, z_H

            # Save the trained SAE periodically
            if global_step % 100 == 0:
                results_path = config.checkpoint_path.replace('ckpt/', 'weights/')
                os.makedirs(results_path, exist_ok=True)
                sae_save_path = os.path.join(results_path, f"sae_step_{global_step:06d}.pt")
                torch.save({
                    'sae_state_dict': sae.state_dict(),
                    'sae_optim_state_dict': sae_optim.state_dict(),
                    'global_step': global_step,
                    'epoch': epoch,
                }, sae_save_path)
                print(f"  Saved SAE to {sae_save_path}")
        
        # Epoch summary (after all batches in this epoch)
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Avg Loss: {sum(epoch_losses)/len(epoch_losses):.4f}")
        print(f"  Avg Recon: {sum(epoch_recon_losses)/len(epoch_recon_losses):.4f}")
        print(f"  Avg Sparse: {sum(epoch_sparse_losses)/len(epoch_sparse_losses):.6f}")
        print(f"{'='*60}\n")
        
        # Log epoch summary to wandb
        wandb.log({
            "epoch/avg_loss": sum(epoch_losses)/len(epoch_losses),
            "epoch/avg_recon_loss": sum(epoch_recon_losses)/len(epoch_recon_losses),
            "epoch/avg_sparse_loss": sum(epoch_sparse_losses)/len(epoch_sparse_losses),
        }, step=global_step)
        
        # Save checkpoint at end of each epoch
        results_path = config.checkpoint_path.replace('ckpt/', 'weights/')
        os.makedirs(results_path, exist_ok=True)
        sae_save_path = os.path.join(results_path, f"sae_epoch_{epoch+1:03d}.pt")
        torch.save({
            'sae_state_dict': sae.state_dict(),
            'sae_optim_state_dict': sae_optim.state_dict(),
            'global_step': global_step,
            'epoch': epoch + 1,
        }, sae_save_path)
        print(f"Saved epoch checkpoint to {sae_save_path}\n")
    
    wandb.finish()
    return sae


@hydra.main(config_path="config", config_name="cfg_eval", version_base=None)
def launch(hydra_config: DictConfig):
    """Main evaluation launch function"""
    # Load config
    config = EvalConfig(**hydra_config)  # type: ignore

    # Seed RNGs to ensure consistency
    torch.random.manual_seed(config.seed)

    # Dataset - Use split from config
    eval_loader, eval_metadata = create_dataloader(
        config, 
        config.split,  # "train" or "test"
        test_set_mode=False, 
        epochs_per_iter=1, 
        global_batch_size=config.global_batch_size
    )

    # Evaluation state
    eval_state = init_eval_state(config, eval_metadata)

    # Save config
    if config.checkpoint_path is not None:
        os.makedirs(config.checkpoint_path.replace('ckpt/', 'results/'), exist_ok=True)
        config_file = os.path.join(config.checkpoint_path.replace('ckpt/', 'results/'), "eval_config.yaml")
        with open(config_file, "wt") as f:
            yaml.dump(config.model_dump(), f)

    # Run training
    print("Starting SAE training...")
    eval_state.model.eval()
    sae = train(config, eval_state, eval_loader, num_epochs=config.num_epochs)


if __name__ == "__main__":
    launch()
