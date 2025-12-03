from typing import Optional, List, Any
from dataclasses import dataclass

import os
import yaml
import torch
from torch import nn
from torch.utils.data import DataLoader

import tqdm
import hydra
import pydantic
from omegaconf import DictConfig
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from utils.functions import load_model_class


class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str


class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str
    loss: LossConfig


class EvalConfig(pydantic.BaseModel):
    # Config
    arch: ArchConfig
    # Data
    data_paths: List[str]
    data_paths_test: List[str] = []

    # Batch size for evaluation
    global_batch_size: int

    # Names
    load_checkpoint: Optional[str] = None
    checkpoint_path: Optional[str] = None

    # Extras
    seed: int = 0
    num_epochs: int = 10  # Number of training epochs for SAE
    split: str = "train"  # Dataset split to use: "train" or "test"

@dataclass
class EvalState:
    model: nn.Module
    carry: Any


def create_dataloader(config: EvalConfig, split: str, **kwargs):
    """Create dataloader for evaluation"""
    dataset = PuzzleDataset(PuzzleDatasetConfig(
        seed=config.seed,
        dataset_paths=config.data_paths_test if len(config.data_paths_test)>0 and split=="test" else config.data_paths,
        rank=0,
        num_replicas=1,
        **kwargs
    ), split=split)
    dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=None,
        num_workers=0,
        pin_memory=True,
    )
    return dataloader, dataset.metadata


def create_model(config: EvalConfig, eval_metadata: PuzzleDatasetMetadata):
    """Create model for evaluation"""
    model_cfg = dict(
        **config.arch.__pydantic_extra__,  # type: ignore
        batch_size=config.global_batch_size,
        vocab_size=eval_metadata.vocab_size,
        seq_len=eval_metadata.seq_len,
        num_puzzle_identifiers=eval_metadata.num_puzzle_identifiers,
        causal=False  # Non-autoregressive
    )

    # Instantiate model with loss head
    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)

    with torch.device("cuda"):
        model: nn.Module = model_cls(model_cfg)
        print(model)
        model = loss_head_cls(model, **config.arch.loss.__pydantic_extra__)  # type: ignore
        if "DISABLE_COMPILE" not in os.environ:
            model = torch.compile(model)  # type: ignore

        # Load checkpoint
        load_checkpoint(model, config)

    return model


def init_eval_state(config: EvalConfig, eval_metadata: PuzzleDatasetMetadata):
    """Initialize evaluation state"""
    # Model
    model = create_model(config, eval_metadata)

    return EvalState(
        model=model,
        carry=None
    )


def load_checkpoint(model: nn.Module, config: EvalConfig):
    """Load model checkpoint"""
    if config.load_checkpoint is not None:
        print(f"Loading checkpoint {config.load_checkpoint}")

        # Load state dict
        state_dict = torch.load(config.load_checkpoint, map_location="cuda", weights_only=False)

        # Resize and reset puzzle emb if needed
        puzzle_emb_name = "_orig_mod.model.inner.puzzle_emb.weights"
        expected_shape: torch.Size = model.model.puzzle_emb.weights.shape  # type: ignore
        if puzzle_emb_name in state_dict:
            puzzle_emb = state_dict[puzzle_emb_name]
            if puzzle_emb.shape != expected_shape:
                print(f"Resetting puzzle embedding as shape is different. Found {puzzle_emb.shape}, Expected {expected_shape}")
                # Re-initialize using mean
                state_dict[puzzle_emb_name] = (
                    torch.mean(puzzle_emb, dim=0, keepdim=True).expand(expected_shape).contiguous()
                )
        model.load_state_dict(state_dict, assign=True)


def evaluate(
    config: EvalConfig,
    eval_state: EvalState,
    eval_loader: torch.utils.data.DataLoader,
):
    """Run evaluation and save z_L/z_H trajectories"""
    with torch.inference_mode():
        processed_batches = 0
        
        for set_name, batch, global_batch_size in eval_loader:
            processed_batches += 1
            print(f"Processing batch {processed_batches}: {set_name}")
            
            # To device
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.device("cuda"):
                carry = eval_state.model.initial_carry(batch)  # type: ignore

            # Store trajectories for this batch
            batch_trajectories_L = []
            batch_trajectories_H = []
            
            # Forward pass - collect z_L and z_H at each step
            inference_steps = 0
            pbar = tqdm.tqdm(desc=f"Inference steps for batch {processed_batches}")
            while True:
                # Save z_L and z_H at each inference step (BEFORE forward pass)
                assert hasattr(carry, 'inner_carry') and (hasattr(carry.inner_carry, 'z_L') and hasattr(carry.inner_carry, 'z_H'))
                batch_trajectories_L.append(carry.inner_carry.z_L.cpu())
                batch_trajectories_H.append(carry.inner_carry.z_H.cpu())
                
                carry, loss, metrics, preds, all_finish = eval_state.model(
                    carry=carry, batch=batch, return_keys=set()
                )
                inference_steps += 1
                pbar.update(1)

                if all_finish:
                    # Save final z_L and z_H after last step
                    assert hasattr(carry, 'inner_carry') and (hasattr(carry.inner_carry, 'z_L') and hasattr(carry.inner_carry, 'z_H'))
                    batch_trajectories_L.append(carry.inner_carry.z_L.cpu())
                    batch_trajectories_H.append(carry.inner_carry.z_H.cpu())
                    break

            pbar.close()
            print(f"  Completed inference in {inference_steps} steps")
            
            # Save trajectories for this batch
            if config.checkpoint_path is not None:
                stacked_trajectories_L = torch.stack(batch_trajectories_L, dim=0)
                stacked_trajectories_H = torch.stack(batch_trajectories_H, dim=0)
                batch_data = {
                    'loss': loss.cpu(),
                    'trajectories_L': stacked_trajectories_L,
                    'trajectories_H': stacked_trajectories_H,
                    'metrics': {k: v.cpu() for k, v in metrics.items()},
                }
                
                os.makedirs(config.checkpoint_path.replace('ckpt/', 'results/'), exist_ok=True)
                batch_path = os.path.join(
                    config.checkpoint_path.replace('ckpt/', 'results/'),
                    f"batch_data_{processed_batches:04d}.pt"
                )
                torch.save(batch_data, batch_path)
                print(f"  Saved batch data to {batch_path}")
                print(f"    Loss: {loss.item():.4f}")
                print(f"    Trajectories_L: {stacked_trajectories_L.shape}")
                print(f"    Trajectories_H: {stacked_trajectories_H.shape}")
                del batch_data, stacked_trajectories_L, stacked_trajectories_H

            del carry, loss, preds, batch, all_finish, metrics, batch_trajectories_L, batch_trajectories_H

        print("\nProcessing complete!")


@hydra.main(config_path="config", config_name="cfg_eval", version_base=None)
def launch(hydra_config: DictConfig):
    """Main evaluation launch function"""
    # Load config
    config = EvalConfig(**hydra_config)  # type: ignore

    # Seed RNGs to ensure consistency
    torch.random.manual_seed(config.seed)

    # Dataset
    try:
        eval_loader, eval_metadata = create_dataloader(config, "test", test_set_mode=True, epochs_per_iter=1, global_batch_size=config.global_batch_size)
    except:
        print("NO EVAL DATA FOUND, using train data")
        eval_loader, eval_metadata = create_dataloader(config, "train", test_set_mode=False, epochs_per_iter=1, global_batch_size=config.global_batch_size)

    # Evaluation state
    eval_state = init_eval_state(config, eval_metadata)

    # Save config
    if config.checkpoint_path is not None:
        os.makedirs(config.checkpoint_path.replace('ckpt/', 'results/'), exist_ok=True)
        config_file = os.path.join(config.checkpoint_path.replace('ckpt/', 'results/'), "eval_config.yaml")
        with open(config_file, "wt") as f:
            yaml.dump(config.model_dump(), f)

    # Run evaluation
    print("Starting evaluation...")
    eval_state.model.eval()
    evaluate(config, eval_state, eval_loader)


if __name__ == "__main__":
    launch()
