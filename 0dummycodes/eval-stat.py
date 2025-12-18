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
    puzzle_ids: Optional[List[int]] = None  # Filter to specific puzzle IDs (e.g., [1, 10, 50, 100])

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

        # Check puzzle embedding shape - MUST match exactly
        puzzle_emb_name = "_orig_mod.model.inner.puzzle_emb.weights"
        expected_shape: torch.Size = model.model.puzzle_emb.weights.shape  # type: ignore
        if puzzle_emb_name in state_dict:
            puzzle_emb = state_dict[puzzle_emb_name]
            if puzzle_emb.shape != expected_shape:
                error_msg = (
                    f"\n{'='*80}\n"
                    f"ERROR: Puzzle embedding shape mismatch!\n"
                    f"{'='*80}\n"
                    f"Checkpoint has: {puzzle_emb.shape[0]} puzzle embeddings\n"
                    f"Current dataset needs: {expected_shape[0]} puzzle embeddings\n"
                    f"\n"
                    f"This means the checkpoint was trained on a DIFFERENT dataset than\n"
                    f"the one you're trying to evaluate on.\n"
                    f"\n"
                    f"Solutions:\n"
                    f"  1. Use the SAME dataset for evaluation as was used for training\n"
                    f"     (Check the checkpoint's all_config.yaml to see training data_paths)\n"
                    f"  2. Retrain the model on the current dataset\n"
                    f"\n"
                    f"Current eval data: {config.data_paths_test if config.data_paths_test else config.data_paths}\n"
                    f"{'='*80}\n"
                )
                raise ValueError(error_msg)
        
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

        carry = None
        processed_batches = 0
        
        # Track per-puzzle-ID statistics
        puzzle_id_stats = {}  # puzzle_id -> {correct: int, total: int}
        puzzle_id_first_seen = set()  # Track which puzzle_ids we've already counted (for first-example-only mode)
        
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

            # Update evaluators
            for evaluator in evaluators:
                evaluator.update_batch(batch, preds)
            
            # Track per-puzzle-ID exact accuracy (FIRST TEST EXAMPLE ONLY - matching visualize_test_data.py)
            if "puzzle_identifiers" in batch:
                puzzle_ids = batch["puzzle_identifiers"].cpu().numpy()
                labels = batch["labels"]
                mask = (labels != -100)  # IGNORE_LABEL_ID
                loss_counts = mask.sum(-1).cpu().numpy()
                is_correct = mask & (preds["preds"] == labels)
                seq_is_correct = (is_correct.sum(-1) == mask.sum(-1)).cpu().numpy()
                halted = carry.halted.cpu().numpy()
                valid = halted & (loss_counts > 0)
                
                blank_id = eval_metadata.blank_identifier_id
                for pid, is_exact_correct, is_valid in zip(puzzle_ids, seq_is_correct, valid):
                    if is_valid and pid != blank_id:  # Skip padding
                        pid = int(pid)
                        # Filter by puzzle_ids if specified
                        if config.puzzle_ids is not None and pid not in config.puzzle_ids:
                            continue
                        # Only count the FIRST test example for each puzzle_id
                        if pid not in puzzle_id_first_seen:
                            puzzle_id_first_seen.add(pid)
                            if pid not in puzzle_id_stats:
                                puzzle_id_stats[pid] = {"correct": 0, "total": 0}
                            puzzle_id_stats[pid]["total"] += 1
                            if is_exact_correct:
                                puzzle_id_stats[pid]["correct"] += 1
            
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
            
            # Print per-puzzle-ID statistics
            if puzzle_id_stats:
                print("\n" + "="*80)
                if config.puzzle_ids is not None:
                    print(f"Per-Puzzle-ID Exact Accuracy Statistics (FIRST TEST EXAMPLE ONLY - Filtered to puzzle_ids: {config.puzzle_ids}):")
                else:
                    print("Per-Puzzle-ID Exact Accuracy Statistics (FIRST TEST EXAMPLE ONLY):")
                print("="*80)
                
                sorted_puzzle_ids = sorted(puzzle_id_stats.keys())
                total_puzzles = len(sorted_puzzle_ids)
                total_samples = sum(stats["total"] for stats in puzzle_id_stats.values())
                total_correct = sum(stats["correct"] for stats in puzzle_id_stats.values())
                
                perfect_puzzles = sum(1 for stats in puzzle_id_stats.values() 
                                     if stats["correct"] == stats["total"] and stats["total"] > 0)
                partial_puzzles = sum(1 for stats in puzzle_id_stats.values() 
                                     if 0 < stats["correct"] < stats["total"])
                failed_puzzles = sum(1 for stats in puzzle_id_stats.values() 
                                    if stats["correct"] == 0)
                
                print(f"\nSummary (matching visualize_test_data.py - 1st example per puzzle):")
                print(f"  Total unique puzzle IDs evaluated: {total_puzzles}")
                print(f"  Total samples (1 per puzzle): {total_samples}")
                print(f"  Overall accuracy: {total_correct / total_samples * 100:.2f}% ({total_correct}/{total_samples})")
                print(f"\n  Puzzles SOLVED (correct): {perfect_puzzles} ({perfect_puzzles / total_puzzles * 100:.2f}%)")
                print(f"  Puzzles FAILED (incorrect): {failed_puzzles} ({failed_puzzles / total_puzzles * 100:.2f}%)")
                
                print(f"\nDetailed breakdown (first 50 puzzle IDs):")
                print(f"{'Puzzle ID':<12} {'Status':<12}")
                print("-" * 30)
                
                for i, pid in enumerate(sorted_puzzle_ids[:50]):
                    stats = puzzle_id_stats[pid]
                    status = "✓ SOLVED" if stats["correct"] == 1 else "✗ FAILED"
                    print(f"{pid:<12} {status:<12}")
                
                if len(sorted_puzzle_ids) > 50:
                    print(f"... and {len(sorted_puzzle_ids) - 50} more puzzle IDs")
                
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
        
        # Save puzzle_id_stats to file
        if puzzle_id_stats and config.checkpoint_path is not None:
            total_correct = sum(stats["correct"] for stats in puzzle_id_stats.values())
            total_puzzles = len(puzzle_id_stats)
            
            note_text = "Statistics for FIRST TEST EXAMPLE ONLY per puzzle (matching visualize_test_data.py)"
            if config.puzzle_ids is not None:
                note_text += f" - Filtered to puzzle_ids: {config.puzzle_ids}"
            
            puzzle_stats_output = {
                "note": note_text,
                "filter": {
                    "puzzle_ids": config.puzzle_ids if config.puzzle_ids is not None else "all"
                },
                "summary": {
                    "total_puzzles_evaluated": total_puzzles,
                    "puzzles_solved": sum(1 for stats in puzzle_id_stats.values() if stats["correct"] == 1),
                    "puzzles_failed": sum(1 for stats in puzzle_id_stats.values() if stats["correct"] == 0),
                    "overall_accuracy": total_correct / total_puzzles if total_puzzles > 0 else 0,
                },
                "per_puzzle": {
                    str(pid): {
                        "status": "solved" if stats["correct"] == 1 else "failed",
                        "correct": stats["correct"],
                    }
                    for pid, stats in sorted(puzzle_id_stats.items())
                }
            }
            
            puzzle_stats_file = os.path.join(config.checkpoint_path.replace('ckpt/', 'results/'), "puzzle_id_stats.yaml")
            os.makedirs(os.path.dirname(puzzle_stats_file), exist_ok=True)
            with open(puzzle_stats_file, "wt") as f:
                yaml.dump(puzzle_stats_output, f, default_flow_style=False, sort_keys=False)
            print(f"\nPuzzle ID statistics saved to: {puzzle_stats_file}")

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
        eval_loader, eval_metadata = create_dataloader(config, "test", test_set_mode=True, epochs_per_iter=1, global_batch_size=config.global_batch_size)
    except:
        print("NO EVAL DATA FOUND, using train data")
        eval_loader, eval_metadata = create_dataloader(config, "train", test_set_mode=False, epochs_per_iter=1, global_batch_size=config.global_batch_size)

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
    metrics = evaluate(config, eval_state, eval_loader, eval_metadata, evaluators)

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
