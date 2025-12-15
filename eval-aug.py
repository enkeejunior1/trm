from typing import Optional, Any, Sequence, List
from dataclasses import dataclass
import os
import math
import yaml
import shutil
import copy

import torch
from torch import nn
from torch.utils.data import DataLoader

import tqdm
import wandb
import coolname
import hydra
import pydantic
from omegaconf import DictConfig

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from utils.functions import load_model_class, get_model_source_path
from models.ema import EMAHelper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str


class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str
    loss: LossConfig


class EvaluatorConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")
    name: str


class EvalConfig(pydantic.BaseModel):
    # Config
    arch: ArchConfig
    # Data
    data_paths: List[str]
    data_paths_test: List[str] = []
    # Evaluators
    evaluators: List[EvaluatorConfig] = []

    # Batch size for evaluation
    global_batch_size: int

    # Names
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    load_checkpoint: Optional[str] = None
    checkpoint_path: Optional[str] = None

    # Extras
    seed: int = 0
    eval_save_outputs: List[str] = []

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
        batch_size=None,
        num_workers=1,
        prefetch_factor=8,
        pin_memory=True,
        persistent_workers=True
    )
    return dataloader, dataset.metadata, dataset


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

    with device:
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
        state_dict = torch.load(config.load_checkpoint, map_location=device, weights_only=False)

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


def create_evaluators(config: EvalConfig, eval_metadata: PuzzleDatasetMetadata) -> List[Any]:
    """Create evaluators for evaluation"""
    data_paths = config.data_paths_test if len(config.data_paths_test)>0 else config.data_paths
    # Initialize evaluators
    evaluators = []
    for cfg in config.evaluators:
        for data_path in data_paths:
            cls = load_model_class(cfg.name, "evaluators.")(
                data_path=data_path, eval_metadata=eval_metadata, **cfg.__pydantic_extra__
            )  # type: ignore
            evaluators.append(cls)

    return evaluators


def evaluate(
    config: EvalConfig,
    eval_state: EvalState,
    eval_loader: torch.utils.data.DataLoader,
    eval_metadata: PuzzleDatasetMetadata,
    evaluators: List[Any],
    eval_dataset: PuzzleDataset,
):
    """Run evaluation on the model"""
    reduced_metrics = None

    with torch.inference_mode():
        return_keys = set(config.eval_save_outputs)
        return_keys.add("preds")  # Always need preds for per-puzzle-ID tracking
        for evaluator in evaluators:
            evaluator.begin_eval()
            return_keys.update(evaluator.required_outputs)

        # Run evaluation
        set_ids = {k: idx for idx, k in enumerate(eval_metadata.sets)}

        save_preds = {}
        metric_keys = []
        metric_values = None

        # Build puzzle_id -> group_id mapping from dataset
        # Load dataset to access group_indices
        eval_dataset._lazy_load_dataset()
        puzzle_to_group = {}
        for set_name, dataset in eval_dataset._data.items():
            group_indices = dataset["group_indices"]
            puzzle_ids_array = dataset["puzzle_identifiers"]
            
            # For each group, find which puzzles belong to it
            for group_id in range(len(group_indices) - 1):
                group_start = group_indices[group_id]
                group_end = group_indices[group_id + 1]
                # All puzzles in this range belong to this group
                for puzzle_idx in range(group_start, group_end):
                    puzzle_id = int(puzzle_ids_array[puzzle_idx])
                    puzzle_to_group[puzzle_id] = (set_name, int(group_id))

        # Track per-puzzle-ID and per-group statistics
        puzzle_id_stats = {}  # puzzle_id -> {correct: int, total: int}
        group_stats = {}  # (set_name, group_id) -> {correct: int, total: int, puzzle_ids: set}

        carry = None
        processed_batches = 0
        
        for set_name, batch, global_batch_size in eval_loader:
            processed_batches += 1
            print(f"Processing batch {processed_batches}: {set_name}")
            
            # To device
            batch = {k: v.to(device) for k, v in batch.items()}
            with device:
                carry = eval_state.model.initial_carry(batch)  # type: ignore

            # Store trajectories for this batch
            batch_trajectories_L = []
            batch_trajectories_H = []
            
            # Forward
            inference_steps = 0
            pbar = tqdm.tqdm(desc=f"Inference steps for batch {processed_batches}")
            while True:
                # Save z_L at each inference step (BEFORE forward pass)
                # This captures the reasoning trajectory at each step
                assert hasattr(carry, 'inner_carry') and (hasattr(carry.inner_carry, 'z_L') and hasattr(carry.inner_carry, 'z_H'))
                batch_trajectories_L.append(carry.inner_carry.z_L.cpu())
                batch_trajectories_H.append(carry.inner_carry.z_H.cpu())
                
                carry, loss, metrics, preds, all_finish = eval_state.model(
                    carry=carry, batch=batch, return_keys=return_keys
                )
                inference_steps += 1
                pbar.update(1)

                if all_finish:
                    # Save final z_L after last step
                    assert hasattr(carry, 'inner_carry') and (hasattr(carry.inner_carry, 'z_L') and hasattr(carry.inner_carry, 'z_H'))
                    batch_trajectories_L.append(carry.inner_carry.z_L.cpu())
                    batch_trajectories_H.append(carry.inner_carry.z_H.cpu())
                    break

            pbar.close()
            print(f"  Completed inference in {inference_steps} steps")
            
            # Save predictions, loss, trajectories, and metrics for this batch immediately
            if config.checkpoint_path is not None:
                stacked_trajectories_L = torch.stack(batch_trajectories_L, dim=0)
                stacked_trajectories_H = torch.stack(batch_trajectories_H, dim=0)
                batch_data = {
                    'loss': loss.cpu(),
                    'trajectories_L': stacked_trajectories_L.cpu(),
                    'trajectories_H': stacked_trajectories_H.cpu(),
                    'metrics': {k: v.cpu() for k, v in metrics.items()},
                    'predictions': {},
                    'batch_info': {}
                }
                
                # Save predictions and relevant batch data
                for collection_name, collection in [('preds', preds), ('batch', batch)]:
                    for k, v in collection.items():
                        if k in config.eval_save_outputs:
                            if collection_name == 'preds':
                                batch_data['predictions'][k] = v.cpu()
                            else:
                                batch_data['batch_info'][k] = v.cpu()
                
                os.makedirs(config.checkpoint_path.replace('ckpt/', 'results/'), exist_ok=True)
                batch_path = os.path.join(
                    config.checkpoint_path.replace('ckpt/', 'results/'),
                    f"batch_data_{processed_batches:04d}.pt"
                )
                torch.save(batch_data, batch_path)
                print(f"  Saved batch data to {batch_path}")
                print(f"    Loss: {loss.item():.4f}")
                print(f"    Metrics: {metrics}")
                print(f"    Predictions: {batch_data['predictions']}")
                print(f"    Batch info: {batch_data['batch_info']}")
                print(f"    Trajectories_L: {stacked_trajectories_L.shape}")
                print(f"    Trajectories_H: {stacked_trajectories_H.shape}")
                del batch_data, stacked_trajectories_L, stacked_trajectories_H

            # Track per-puzzle-ID exact accuracy
            if "puzzle_identifiers" in batch:
                puzzle_ids = batch["puzzle_identifiers"].cpu().numpy()
                # Get exact correctness per example from metrics
                # We need to recompute this from the carry since metrics are aggregated
                labels = batch["labels"]
                mask = (labels != -100)  # IGNORE_LABEL_ID
                loss_counts = mask.sum(-1).cpu().numpy()
                is_correct = mask & (preds["preds"] == labels)
                seq_is_correct = (is_correct.sum(-1) == mask.sum(-1)).cpu().numpy()
                halted = carry.halted.cpu().numpy()
                valid = halted & (loss_counts > 0)
                
                # Update puzzle_id_stats and group_stats
                blank_id = eval_metadata.blank_identifier_id
                for pid, is_exact_correct, is_valid in zip(puzzle_ids, seq_is_correct, valid):
                    if is_valid and pid != blank_id:  # Skip padding
                        pid = int(pid)
                        
                        # Update puzzle-level stats
                        if pid not in puzzle_id_stats:
                            puzzle_id_stats[pid] = {"correct": 0, "total": 0}
                        puzzle_id_stats[pid]["total"] += 1
                        if is_exact_correct:
                            puzzle_id_stats[pid]["correct"] += 1
                        
                        # Update group-level stats
                        if pid in puzzle_to_group:
                            group_key = puzzle_to_group[pid]
                            if group_key not in group_stats:
                                group_stats[group_key] = {"correct": 0, "total": 0, "puzzle_ids": set()}
                            group_stats[group_key]["puzzle_ids"].add(pid)
                            group_stats[group_key]["total"] += 1
                            if is_exact_correct:
                                group_stats[group_key]["correct"] += 1

            # Update evaluators
            for evaluator in evaluators:
                evaluator.update_batch(batch, preds)
            del carry, loss, preds, batch, all_finish

            # Aggregate metrics
            set_id = set_ids[set_name]

            if metric_values is None:
                metric_keys = list(
                    sorted(metrics.keys())
                )  # Sort keys to guarantee consistent order
                metric_values = torch.zeros(
                    (len(set_ids), len(metrics.values())), dtype=torch.float32, device=device
                )

            metric_values[set_id] += torch.stack([metrics[k] for k in metric_keys])
            del metrics

        # Concatenate saved predictions
        save_preds = {k: torch.cat(v, dim=0) for k, v in save_preds.items()}

        # Save predictions
        if config.checkpoint_path is not None:
            os.makedirs(config.checkpoint_path, exist_ok=True)
            
            if len(save_preds):
                torch.save(
                    save_preds, os.path.join(config.checkpoint_path, f"all_preds.pt")
                )
        del save_preds

        # Process metrics
        if metric_values is not None:
            reduced_metrics = metric_values.cpu().numpy()
            reduced_metrics = {
                set_name: {
                    metric_name: reduced_metrics[set_id, metric_id]
                    for metric_id, metric_name in enumerate(metric_keys)
                }
                for set_id, set_name in enumerate(set_ids)
            }

            # Postprocess metrics
            for set_name, m in reduced_metrics.items():
                count = m.pop("count")
                reduced_metrics[set_name] = {k: v / count for k, v in m.items()}
            
            # Print average metrics (especially accuracy)
            print("\n" + "="*60)
            print("Average Metrics:")
            print("="*60)
            for set_name, metrics_dict in reduced_metrics.items():
                print(f"\n{set_name}:", flush=True)
                for metric_name, metric_value in metrics_dict.items():
                    print(f"  {metric_name}: {metric_value:.4f}", flush=True)
            print("="*60)
            
            # Print per-puzzle-ID and per-group statistics
            if puzzle_id_stats:
                print("\n" + "="*80)
                print("Per-Puzzle-ID Exact Accuracy Statistics:")
                print("="*80)
                
                # Sort by puzzle_id
                sorted_puzzle_ids = sorted(puzzle_id_stats.keys())
                
                # Calculate overall statistics
                total_puzzles = len(sorted_puzzle_ids)
                total_samples = sum(stats["total"] for stats in puzzle_id_stats.values())
                total_correct = sum(stats["correct"] for stats in puzzle_id_stats.values())
                
                # Count puzzles with different accuracy levels
                perfect_puzzles = sum(1 for stats in puzzle_id_stats.values() if stats["correct"] == stats["total"] and stats["total"] > 0)
                partial_puzzles = sum(1 for stats in puzzle_id_stats.values() if 0 < stats["correct"] < stats["total"])
                failed_puzzles = sum(1 for stats in puzzle_id_stats.values() if stats["correct"] == 0)
                
                print(f"\nPuzzle-Level Summary (treating each puzzle_id separately):")
                print(f"  Total unique puzzle IDs evaluated: {total_puzzles}")
                print(f"  Total samples: {total_samples}")
                print(f"  Average samples per puzzle ID: {total_samples / total_puzzles:.2f}")
                print(f"  Total correct samples: {total_correct}")
                print(f"  Overall sample-level accuracy: {total_correct / total_samples * 100:.2f}%")
                print(f"\n  Puzzle IDs with 100% accuracy: {perfect_puzzles} ({perfect_puzzles / total_puzzles * 100:.2f}%)")
                print(f"  Puzzle IDs with partial accuracy: {partial_puzzles} ({partial_puzzles / total_puzzles * 100:.2f}%)")
                print(f"  Puzzle IDs with 0% accuracy: {failed_puzzles} ({failed_puzzles / total_puzzles * 100:.2f}%)")
                
                # Calculate average exact_acc per puzzle (treating each puzzle equally)
                puzzle_level_accs = [stats["correct"] / stats["total"] for stats in puzzle_id_stats.values() if stats["total"] > 0]
                avg_puzzle_acc = sum(puzzle_level_accs) / len(puzzle_level_accs) if puzzle_level_accs else 0
                print(f"  Average puzzle-level exact accuracy: {avg_puzzle_acc * 100:.2f}%")
                
                print(f"\nDetailed per-puzzle breakdown (showing first 50 puzzle IDs):")
                print(f"{'Puzzle ID':<12} {'Correct':<10} {'Total':<10} {'Accuracy':<12}")
                print("-" * 50)
                
                for i, pid in enumerate(sorted_puzzle_ids[:50]):
                    stats = puzzle_id_stats[pid]
                    acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
                    print(f"{pid:<12} {stats['correct']:<10} {stats['total']:<10} {acc:<11.2f}%")
                
                if len(sorted_puzzle_ids) > 50:
                    print(f"... and {len(sorted_puzzle_ids) - 50} more puzzle IDs")
                
                print("="*80)
            
            # Print group-level statistics (considering augmentation)
            if group_stats:
                print("\n" + "="*80)
                print("Per-Group Exact Accuracy Statistics (Original Puzzle + Augmentations):")
                print("="*80)
                
                # Calculate statistics
                total_groups = len(group_stats)
                total_samples_in_groups = sum(stats["total"] for stats in group_stats.values())
                total_correct_in_groups = sum(stats["correct"] for stats in group_stats.values())
                
                # Group accuracy: a group is correct if ALL its samples are correct
                perfect_groups = sum(1 for stats in group_stats.values() if stats["correct"] == stats["total"] and stats["total"] > 0)
                partial_groups = sum(1 for stats in group_stats.values() if 0 < stats["correct"] < stats["total"])
                failed_groups = sum(1 for stats in group_stats.values() if stats["correct"] == 0)
                
                # Count groups by size (number of augmentations)
                group_sizes = {}
                for stats in group_stats.values():
                    num_puzzles = len(stats["puzzle_ids"])
                    group_sizes[num_puzzles] = group_sizes.get(num_puzzles, 0) + 1
                
                print(f"\nGroup-Level Summary (original + augmented versions treated as one group):")
                print(f"  Total groups: {total_groups}")
                print(f"  Total samples across all groups: {total_samples_in_groups}")
                print(f"  Average samples per group: {total_samples_in_groups / total_groups:.2f}")
                print(f"  Total correct samples: {total_correct_in_groups}")
                print(f"  Overall sample-level accuracy: {total_correct_in_groups / total_samples_in_groups * 100:.2f}%")
                
                print(f"\n  Groups with 100% accuracy (all versions correct): {perfect_groups} ({perfect_groups / total_groups * 100:.2f}%)")
                print(f"  Groups with partial accuracy: {partial_groups} ({partial_groups / total_groups * 100:.2f}%)")
                print(f"  Groups with 0% accuracy: {failed_groups} ({failed_groups / total_groups * 100:.2f}%)")
                
                # Average group-level accuracy
                group_level_accs = [stats["correct"] / stats["total"] for stats in group_stats.values() if stats["total"] > 0]
                avg_group_acc = sum(group_level_accs) / len(group_level_accs) if group_level_accs else 0
                print(f"  Average group-level exact accuracy: {avg_group_acc * 100:.2f}%")
                
                print(f"\nGroup size distribution (number of puzzle versions per group):")
                for size in sorted(group_sizes.keys()):
                    count = group_sizes[size]
                    print(f"  {size} version(s): {count} groups ({count / total_groups * 100:.2f}%)")
                
                print("\nAugmentation Impact Analysis:")
                # Find groups where some versions are correct but not all
                inconsistent_groups = []
                for group_key, stats in group_stats.items():
                    if 0 < stats["correct"] < stats["total"]:
                        inconsistent_groups.append((group_key, stats))
                
                if inconsistent_groups:
                    print(f"  Found {len(inconsistent_groups)} groups with inconsistent results (some versions correct, others wrong)")
                    print(f"  This is {len(inconsistent_groups) / total_groups * 100:.2f}% of all groups")
                    print(f"\n  Examples of inconsistent groups (showing first 10):")
                    print(f"  {'Group ID':<15} {'Set':<10} {'Correct':<10} {'Total':<10} {'Accuracy':<12} {'# Puzzles'}")
                    print("-" * 75)
                    for i, (group_key, stats) in enumerate(inconsistent_groups[:10]):
                        set_name, group_id = group_key
                        acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
                        print(f"  {group_id:<15} {set_name:<10} {stats['correct']:<10} {stats['total']:<10} {acc:<11.2f}% {len(stats['puzzle_ids'])}")
                else:
                    print(f"  All groups have consistent results (all versions correct or all wrong)")
                    print(f"  This suggests augmentations preserve the puzzle difficulty")
                
                print("="*80)
        else:
            raise ValueError("No metrics found")

        # Run evaluators
        print(f"\nRunning {len(evaluators)} evaluator(s)...")
        
        for i, evaluator in enumerate(evaluators):
            print(f"Running evaluator {i+1}/{len(evaluators)}: {evaluator.__class__.__name__}")
                
            # Path for saving
            evaluator_save_path = None
            if config.checkpoint_path is not None:
                evaluator_save_path = os.path.join(
                    config.checkpoint_path,
                    f"evaluator_{evaluator.__class__.__name__}",
                )
                os.makedirs(evaluator_save_path, exist_ok=True)

            # Run and log
            metrics = evaluator.result(evaluator_save_path, rank=0, world_size=1, group=None)
            if metrics is not None:
                if reduced_metrics is None:
                    reduced_metrics = {}

                reduced_metrics.update(metrics)
                print(f"  Completed {evaluator.__class__.__name__}")
                
        print("All evaluators completed!")
        
        # Save puzzle_id_stats and group_stats
        if puzzle_id_stats and config.checkpoint_path is not None:
            # Prepare puzzle-level stats
            puzzle_stats_output = {
                "puzzle_level_summary": {
                    "total_puzzles": len(puzzle_id_stats),
                    "total_samples": sum(stats["total"] for stats in puzzle_id_stats.values()),
                    "total_correct": sum(stats["correct"] for stats in puzzle_id_stats.values()),
                    "avg_samples_per_puzzle": sum(stats["total"] for stats in puzzle_id_stats.values()) / len(puzzle_id_stats),
                    "overall_sample_accuracy": sum(stats["correct"] for stats in puzzle_id_stats.values()) / sum(stats["total"] for stats in puzzle_id_stats.values()),
                    "avg_puzzle_accuracy": sum(stats["correct"] / stats["total"] for stats in puzzle_id_stats.values() if stats["total"] > 0) / len([s for s in puzzle_id_stats.values() if s["total"] > 0]),
                    "perfect_puzzles": sum(1 for stats in puzzle_id_stats.values() if stats["correct"] == stats["total"] and stats["total"] > 0),
                    "partial_puzzles": sum(1 for stats in puzzle_id_stats.values() if 0 < stats["correct"] < stats["total"]),
                    "failed_puzzles": sum(1 for stats in puzzle_id_stats.values() if stats["correct"] == 0),
                },
                "per_puzzle": {
                    str(pid): {
                        "correct": stats["correct"],
                        "total": stats["total"],
                        "accuracy": stats["correct"] / stats["total"] if stats["total"] > 0 else 0
                    }
                    for pid, stats in sorted(puzzle_id_stats.items())
                }
            }
            
            # Add group-level stats if available
            if group_stats:
                group_sizes = {}
                for stats in group_stats.values():
                    num_puzzles = len(stats["puzzle_ids"])
                    group_sizes[num_puzzles] = group_sizes.get(num_puzzles, 0) + 1
                
                puzzle_stats_output["group_level_summary"] = {
                    "total_groups": len(group_stats),
                    "total_samples": sum(stats["total"] for stats in group_stats.values()),
                    "total_correct": sum(stats["correct"] for stats in group_stats.values()),
                    "avg_samples_per_group": sum(stats["total"] for stats in group_stats.values()) / len(group_stats),
                    "overall_sample_accuracy": sum(stats["correct"] for stats in group_stats.values()) / sum(stats["total"] for stats in group_stats.values()),
                    "avg_group_accuracy": sum(stats["correct"] / stats["total"] for stats in group_stats.values() if stats["total"] > 0) / len([s for s in group_stats.values() if s["total"] > 0]),
                    "perfect_groups": sum(1 for stats in group_stats.values() if stats["correct"] == stats["total"] and stats["total"] > 0),
                    "partial_groups": sum(1 for stats in group_stats.values() if 0 < stats["correct"] < stats["total"]),
                    "failed_groups": sum(1 for stats in group_stats.values() if stats["correct"] == 0),
                    "group_size_distribution": {str(k): v for k, v in sorted(group_sizes.items())}
                }
                
                puzzle_stats_output["per_group"] = {
                    f"{set_name}_group_{group_id}": {
                        "correct": stats["correct"],
                        "total": stats["total"],
                        "accuracy": stats["correct"] / stats["total"] if stats["total"] > 0 else 0,
                        "num_puzzle_versions": len(stats["puzzle_ids"]),
                        "puzzle_ids": sorted(list(stats["puzzle_ids"]))
                    }
                    for (set_name, group_id), stats in sorted(group_stats.items())
                }
            
            puzzle_stats_file = os.path.join(config.checkpoint_path.replace('ckpt/', 'results/'), "puzzle_id_stats.yaml")
            os.makedirs(os.path.dirname(puzzle_stats_file), exist_ok=True)
            with open(puzzle_stats_file, "wt") as f:
                yaml.dump(puzzle_stats_output, f, default_flow_style=False, sort_keys=False)
            print(f"\nPuzzle ID and Group statistics saved to: {puzzle_stats_file}")

    return reduced_metrics


def save_code_and_config(config: EvalConfig):
    """Save code and configuration files"""
    if config.checkpoint_path is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)

    # Copy code
    code_list = [
        get_model_source_path(config.arch.name),
        get_model_source_path(config.arch.loss.name)
    ]
    for code_file in code_list:
        if code_file is not None:
            code_name = os.path.basename(code_file)
            shutil.copy(code_file, os.path.join(config.checkpoint_path, code_name))

    # Dump config as yaml
    config_file = os.path.join(config.checkpoint_path, "eval_config.yaml")
    with open(config_file, "wt") as f:
        yaml.dump(config.model_dump(), f)


@hydra.main(config_path="config", config_name="cfg_eval", version_base=None)
def launch(hydra_config: DictConfig):
    """Main evaluation launch function"""
    # Load config
    config = EvalConfig(**hydra_config)  # type: ignore

    # Naming
    if config.project_name is None:
        config.project_name = f"{os.path.basename(config.data_paths[0]).capitalize()}-ACT-eval"
    if config.run_name is None:
        config.run_name = f"{config.arch.name.split('@')[-1]} eval {coolname.generate_slug(2)}"
    if config.checkpoint_path is None:
        config.checkpoint_path = os.path.join("eval_results", config.project_name, config.run_name)

    # Seed RNGs to ensure consistency
    torch.random.manual_seed(config.seed)

    # Dataset
    try:
        eval_loader, eval_metadata, eval_dataset = create_dataloader(config, "test", test_set_mode=True, epochs_per_iter=1, global_batch_size=config.global_batch_size)
    except:
        print("NO EVAL DATA FOUND, using train data")
        eval_loader, eval_metadata, eval_dataset = create_dataloader(config, "train", test_set_mode=False, epochs_per_iter=1, global_batch_size=config.global_batch_size)

    # Create evaluators
    try:
        evaluators = create_evaluators(config, eval_metadata)
    except:
        print("No evaluator found")
        evaluators = []

    # Evaluation state
    eval_state = init_eval_state(config, eval_metadata)

    # Save code and config
    save_code_and_config(config)

    # Run evaluation
    print("Starting evaluation...")
    eval_state.model.eval()
    metrics = evaluate(config, eval_state, eval_loader, eval_metadata, evaluators, eval_dataset)

    if metrics is not None:
        print("\nEvaluation Results:")
        for key, value in metrics.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for sub_key, sub_value in value.items():
                    print(f"  {sub_key}: {sub_value}")
            else:
                print(f"{key}: {value}")

        # Save metrics
        if config.checkpoint_path is not None:
            metrics_file = os.path.join(config.checkpoint_path, "metrics.yaml")
            with open(metrics_file, "wt") as f:
                yaml.dump(metrics, f)
            print(f"\nMetrics saved to: {metrics_file}")

    print("Evaluation completed!")


if __name__ == "__main__":
    launch()
