from typing import Optional, Any, Sequence, List, Dict
from dataclasses import dataclass
import os
import math
import yaml
import shutil
import copy
import json

import numpy as np
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

# Import visualization utilities
from viz_utils import (
    decode_arc_grid,
    visualize_grid,
    compute_diff,
    visualize_batch_results,
    create_summary_figure,
    ARC_COLORS
)
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

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
    
    # Visualization options
    visualize: bool = False
    visualization_output_dir: Optional[str] = None
    num_visualize: Optional[int] = None  # Limit number of visualizations (None = all)
    puzzle_ids: Optional[List[int]] = None  # Specific puzzle IDs to visualize
    random_visualize: bool = False  # Randomly select puzzles for visualization (uses seed)

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


def load_visualization_context(config: EvalConfig) -> Dict:
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


def parse_augmentation(full_identifier: str) -> Dict:
    """
    Parse augmentation info from identifier.
    
    Format: base_name|||transform_type|||color_permutation
    Example: 7c9b52a0|||t3|||0651378924
    
    Transform types:
    - t0: identity
    - t1: rotate 90
    - t2: rotate 180
    - t3: rotate 270
    - t4: flip horizontal
    - t5: flip vertical
    - t6: flip + rotate 90
    - t7: flip + rotate 270
    
    Color permutation: string of digits representing color mapping
    e.g., "0651378924" means color 0->0, 1->6, 2->5, 3->1, etc.
    """
    if "|||" not in full_identifier:
        return {"base_name": full_identifier, "transform": None, "color_perm": None}
    
    parts = full_identifier.split("|||")
    base_name = parts[0]
    transform = parts[1] if len(parts) > 1 else None
    color_perm = parts[2] if len(parts) > 2 else None
    
    return {
        "base_name": base_name,
        "transform": transform,
        "color_perm": color_perm
    }


def apply_augmentation_to_grid(grid: np.ndarray, aug_info: Dict) -> np.ndarray:
    """
    Apply augmentation (transformation + color permutation) to a grid.
    
    Args:
        grid: [H, W] numpy array of color IDs (0-9)
        aug_info: dict with 'transform' and 'color_perm' keys
    
    Returns:
        Transformed grid
    """
    result = grid.copy()
    
    # Apply spatial transformation
    transform = aug_info.get("transform")
    if transform:
        if transform == "t0":
            pass  # identity
        elif transform == "t1":
            result = np.rot90(result, k=1)  # 90 degrees CCW
        elif transform == "t2":
            result = np.rot90(result, k=2)  # 180 degrees
        elif transform == "t3":
            result = np.rot90(result, k=3)  # 270 degrees CCW
        elif transform == "t4":
            result = np.fliplr(result)  # flip horizontal
        elif transform == "t5":
            result = np.flipud(result)  # flip vertical
        elif transform == "t6":
            result = np.rot90(np.fliplr(result), k=1)  # flip + rotate 90
        elif transform == "t7":
            result = np.rot90(np.fliplr(result), k=3)  # flip + rotate 270
    
    # Apply color permutation
    color_perm = aug_info.get("color_perm")
    if color_perm and len(color_perm) == 10:
        # Create color mapping: original color i -> new color int(color_perm[i])
        color_map = np.array([int(c) for c in color_perm], dtype=np.uint8)
        result = color_map[result]
    
    return result


def transform_demo_examples(demo_examples: List[Dict], aug_info: Dict) -> List[Dict]:
    """
    Transform demo examples to match augmented test data.
    
    Args:
        demo_examples: list of demo dicts with 'input' and 'output' keys
        aug_info: augmentation info from parse_augmentation
    
    Returns:
        List of transformed demo dicts
    """
    if not aug_info.get("transform") and not aug_info.get("color_perm"):
        return demo_examples
    
    transformed = []
    for demo in demo_examples:
        in_grid = np.array(demo['input'], dtype=np.uint8)
        out_grid = np.array(demo['output'], dtype=np.uint8)
        
        transformed.append({
            'input': apply_augmentation_to_grid(in_grid, aug_info).tolist(),
            'output': apply_augmentation_to_grid(out_grid, aug_info).tolist()
        })
    
    return transformed


def visualize_single_prediction(
    test_input: np.ndarray,
    test_label: np.ndarray,
    prediction: np.ndarray,
    demo_examples: List[Dict],
    puzzle_name: str,
    puzzle_id: int,
    output_path: str,
    loss: float = None,
    is_augmented: bool = False,
    full_identifier: str = None,
    show_demos: bool = False,  # Disabled by default
):
    """
    Visualize a single prediction with ground truth and model prediction.
    
    Args:
        test_input: tokenized test input [seq_len]
        test_label: tokenized test label [seq_len]
        prediction: model prediction [seq_len]
        demo_examples: list of demo dicts with 'input' and 'output' keys (unused if show_demos=False)
        puzzle_name: base name of the puzzle (without augmentation suffix)
        puzzle_id: puzzle identifier number
        output_path: path to save the figure
        loss: optional loss value
        is_augmented: whether this is an augmented (transformed) puzzle
        full_identifier: full identifier including augmentation info
        show_demos: whether to show demo examples (default: False)
    
    Returns:
        dict with visualization results
    """
    # Decode grids
    input_grid = decode_arc_grid(test_input)
    label_grid = decode_arc_grid(test_label)
    pred_grid = decode_arc_grid(prediction)
    
    is_correct = np.array_equal(label_grid, pred_grid)
    diff_mask, num_diff = compute_diff(label_grid, pred_grid)
    
    # Simple 2-row layout: Test (Input + Target) and Model (Input + Prediction)
    fig_width = 10
    fig_height = 7
    fig, axes = plt.subplots(2, 2, figsize=(fig_width, fig_height))
    
    # Row 0: Test Input and Target (Ground Truth)
    visualize_grid(input_grid, axes[0, 0], "Test - Input")
    visualize_grid(label_grid, axes[0, 1], "Test - Target (Ground Truth)", 
                   highlight_color='#0074D9')
    
    # Row 1: Model Input and Prediction
    status = "✓ CORRECT" if is_correct else "✗ INCORRECT"
    status_color = '#2ECC40' if is_correct else '#FF4136'
    
    visualize_grid(input_grid, axes[1, 0], "Model - Input")
    visualize_grid(pred_grid, axes[1, 1], f"Model - Prediction ({status})",
                   highlight_color=status_color, 
                   show_diff=diff_mask if not is_correct else None)
    
    # Title
    title = f"Puzzle: {puzzle_name} (ID: {puzzle_id})"
    title += f"\nTest: {status}"
    if loss is not None:
        title += f" | Loss: {loss:.4f}"
    if not is_correct:
        title += f" | {num_diff} cells wrong"
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return {
        'puzzle_name': puzzle_name,
        'puzzle_id': puzzle_id,
        'is_correct': is_correct,
        'num_diff': num_diff,
        'loss': loss,
        'is_augmented': is_augmented
    }


def visualize_evaluation_results(
    all_inputs: np.ndarray,
    all_labels: np.ndarray,
    all_predictions: np.ndarray,
    all_puzzle_ids: np.ndarray,
    config: EvalConfig,
    viz_context: Dict,
    all_losses: np.ndarray = None,
):
    """
    Visualize all evaluation results.
    
    Args:
        all_inputs: [N, seq_len] tokenized inputs
        all_labels: [N, seq_len] tokenized labels
        all_predictions: [N, seq_len] model predictions
        all_puzzle_ids: [N] puzzle identifier indices
        config: evaluation config
        viz_context: dict with 'identifiers_map' and 'test_puzzles'
        all_losses: optional [N] array of losses per example
    
    Returns:
        list of result dicts
    """
    output_dir = config.visualization_output_dir or os.path.join(config.checkpoint_path, "visualizations")
    os.makedirs(output_dir, exist_ok=True)
    
    identifiers_map = viz_context['identifiers_map']
    test_puzzles = viz_context['test_puzzles']
    
    # Filter by puzzle_ids if specified
    if config.puzzle_ids is not None:
        mask = np.isin(all_puzzle_ids, config.puzzle_ids)
        indices = np.where(mask)[0]
    else:
        indices = np.arange(len(all_inputs))
    
    # Randomly shuffle if requested
    if config.random_visualize:
        np.random.seed(config.seed)
        np.random.shuffle(indices)
    
    # Limit number of visualizations
    if config.num_visualize is not None:
        indices = indices[:config.num_visualize]
    
    print(f"\n{'='*60}")
    print(f"Visualizing {len(indices)} predictions...")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")
    
    results = []
    correct_count = 0
    
    for viz_idx, idx in enumerate(indices):
        # Get puzzle name
        pid = int(all_puzzle_ids[idx])
        if pid < len(identifiers_map):
            name = identifiers_map[pid]
        else:
            name = f"puzzle_{pid}"
        
        # Get base name (without augmentation suffix)
        base_name = name.split("|||")[0] if "|||" in name else name
        
        # Get demo examples
        demos = test_puzzles.get(base_name, {}).get("train", [])
        
        # Loss for this example
        loss = float(all_losses[idx]) if all_losses is not None else None
        
        # Output path
        save_path = os.path.join(output_dir, f"pred_{viz_idx:04d}_{base_name}.png")
        
        # Visualize
        result = visualize_single_prediction(
            test_input=all_inputs[idx],
            test_label=all_labels[idx],
            prediction=all_predictions[idx],
            demo_examples=demos,
            puzzle_name=base_name,
            puzzle_id=pid,
            output_path=save_path,
            loss=loss,
        )
        
        results.append(result)
        if result['is_correct']:
            correct_count += 1
        
        status = "✓" if result['is_correct'] else f"✗ ({result['num_diff']} diff)"
        print(f"  [{viz_idx+1}/{len(indices)}] {base_name}: {status}")
    
    # Create summary figure
    if len(results) > 0:
        summary_path = os.path.join(output_dir, "summary.png")
        create_evaluation_summary(results, summary_path)
        
        # Save results JSON
        results_path = os.path.join(output_dir, "visualization_results.json")
        with open(results_path, 'w') as f:
            json.dump({
                'total': len(results),
                'correct': correct_count,
                'accuracy': correct_count / len(results) if len(results) > 0 else 0,
                'results': results
            }, f, indent=2)
        print(f"\nResults saved to: {results_path}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("VISUALIZATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total visualized: {len(results)}")
    print(f"Correct: {correct_count} ({correct_count/len(results)*100:.1f}%)")
    print(f"Incorrect: {len(results) - correct_count} ({(len(results)-correct_count)/len(results)*100:.1f}%)")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")
    
    return results


def create_evaluation_summary(results: List[Dict], output_path: str):
    """Create a summary figure for evaluation visualizations."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    n = len(results)
    correct = sum(1 for r in results if r.get('is_correct', False))
    incorrect = n - correct
    
    # 1. Accuracy bar chart
    ax = axes[0]
    x = ['Correct', 'Incorrect']
    counts = [correct, incorrect]
    colors = ['#2ECC40', '#FF4136']
    
    bars = ax.bar(x, counts, color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel('Count')
    ax.set_title('Prediction Results')
    
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(count), ha='center', va='bottom', fontweight='bold')
    
    # 2. Accuracy pie chart
    ax = axes[1]
    if correct > 0 or incorrect > 0:
        sizes = [correct, incorrect]
        labels = [f'Correct\n{correct} ({correct/n*100:.1f}%)', 
                  f'Incorrect\n{incorrect} ({incorrect/n*100:.1f}%)']
        explode = (0.05, 0)
        ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='',
               startangle=90, shadow=True)
    ax.set_title('Accuracy Distribution')
    
    # Title
    acc = correct / n * 100 if n > 0 else 0
    plt.suptitle(f'Evaluation Summary: {correct}/{n} Correct ({acc:.1f}%)',
                fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Summary figure saved to: {output_path}")


def evaluate(
    config: EvalConfig,
    eval_state: EvalState,
    eval_loader: torch.utils.data.DataLoader,
    eval_metadata: PuzzleDatasetMetadata,
    evaluators: List[Any],
    viz_context: Dict = None,
):
    """Run evaluation on the model"""
    reduced_metrics = None
    
    # For visualization: setup output directory and tracking
    viz_results = []
    viz_count = 0
    viz_correct_count = 0
    viz_output_dir = None
    identifiers_map = []
    test_puzzles = {}
    
    if config.visualize and viz_context is not None:
        viz_output_dir = config.visualization_output_dir or os.path.join(config.checkpoint_path, "visualizations")
        os.makedirs(viz_output_dir, exist_ok=True)
        identifiers_map = viz_context.get('identifiers_map', [])
        test_puzzles = viz_context.get('test_puzzles', {})
        print(f"\n{'='*60}")
        print(f"Visualization enabled - saving to: {viz_output_dir}")
        if config.puzzle_ids:
            print(f"Filtering to puzzle IDs: {config.puzzle_ids}")
        if config.num_visualize:
            print(f"Limiting to {config.num_visualize} visualizations")
        print(f"{'='*60}\n")

    with torch.inference_mode():
        return_keys = set(config.eval_save_outputs)
        for evaluator in evaluators:
            evaluator.begin_eval()
            return_keys.update(evaluator.required_outputs)
        
        # Add "preds" to return keys if visualization is enabled
        if config.visualize:
            return_keys.add("preds")

        # Run evaluation
        set_ids = {k: idx for idx, k in enumerate(eval_metadata.sets)}

        metric_keys = []
        metric_values = None

        carry = None
        processed_batches = 0
        skipped_batches = 0
        global_example_idx = 0  # Track global dataset index across all batches
        
        # Convert puzzle_ids to set for O(1) lookup
        puzzle_ids_set = set(config.puzzle_ids) if config.puzzle_ids is not None else None
        
        for set_name, batch, global_batch_size in eval_loader:
            batch_size = batch["inputs"].shape[0]
            
            # Check if ANY example in this batch matches puzzle_ids filter
            # If puzzle_ids is specified and no examples match, skip this batch entirely
            if puzzle_ids_set is not None:
                batch_indices = range(global_example_idx, global_example_idx + batch_size)
                has_match = any(idx in puzzle_ids_set for idx in batch_indices)
                
                if not has_match:
                    # Skip this batch - no matching puzzle_ids
                    global_example_idx += batch_size
                    skipped_batches += 1
                    if skipped_batches % 1000 == 0:
                        print(f"  Skipped {skipped_batches} batches (current idx: {global_example_idx})...")
                    continue
            
            processed_batches += 1
            print(f"Processing batch {processed_batches}: {set_name} (global_idx: {global_example_idx})")
            
            # To device
            batch = {k: v.to(device) for k, v in batch.items()}
            with device:
                carry = eval_state.model.initial_carry(batch)  # type: ignore
            
            # Forward
            inference_steps = 0
            pbar = tqdm.tqdm(desc=f"Inference steps for batch {processed_batches}")
            while True:
                carry, loss, metrics, preds, all_finish = eval_state.model(
                    carry=carry, batch=batch, return_keys=return_keys
                )
                inference_steps += 1
                pbar.update(1)

                if all_finish:
                    break

            pbar.close()
            print(f"  Completed inference in {inference_steps} steps")
            
            # Print batch-level exact accuracy
            if 'accuracy' in metrics:
                batch_acc = metrics['accuracy'].item()
                print(f"  Batch {processed_batches} Exact Accuracy: {batch_acc:.4f} ({batch_acc*100:.2f}%)")
            
            # Print all batch metrics for debugging
            print(f"  Batch {processed_batches} Metrics:")
            for metric_name, metric_value in sorted(metrics.items()):
                if metric_name != 'count':
                    val = metric_value.item() if hasattr(metric_value, 'item') else metric_value
                    print(f"    {metric_name}: {val:.4f}")

            # Update evaluators
            for evaluator in evaluators:
                evaluator.update_batch(batch, preds)
            
            # Incremental visualization - visualize each batch immediately
            if config.visualize and preds is not None and "preds" in preds and viz_output_dir is not None:
                batch_inputs = batch["inputs"].cpu().numpy()
                batch_labels = batch["labels"].cpu().numpy()
                batch_preds = preds["preds"].cpu().numpy()
                batch_puzzle_ids = batch["puzzle_identifiers"].cpu().numpy()
                batch_loss = loss.item() if hasattr(loss, 'item') else float(loss)
                
                # Visualize each example in batch
                for i in range(len(batch_inputs)):
                    current_idx = global_example_idx + i
                    
                    # Check limit
                    if config.num_visualize is not None and viz_count >= config.num_visualize:
                        continue
                    
                    pid = int(batch_puzzle_ids[i])
                    
                    # Get puzzle name and check if augmented
                    # Handle offset for out-of-range puzzle IDs (common in aug datasets)
                    if pid < len(identifiers_map):
                        full_name = identifiers_map[pid]
                    else:
                        # Try offset correction (pid - len(identifiers_map))
                        offset_pid = pid - len(identifiers_map)
                        if 0 <= offset_pid < len(identifiers_map):
                            full_name = identifiers_map[offset_pid]
                        else:
                            full_name = f"puzzle_{pid}"
                    
                    is_augmented = "|||" in full_name
                    base_name = full_name.split("|||")[0] if is_augmented else full_name
                    
                    # Get demo examples (only useful for non-augmented puzzles)
                    demos = test_puzzles.get(base_name, {}).get("train", [])
                    
                    # Output path
                    save_path = os.path.join(viz_output_dir, f"pred_{viz_count:04d}_{base_name}.png")
                    
                    # Visualize
                    result = visualize_single_prediction(
                        test_input=batch_inputs[i],
                        test_label=batch_labels[i],
                        prediction=batch_preds[i],
                        demo_examples=demos,
                        puzzle_name=base_name,
                        puzzle_id=pid,
                        output_path=save_path,
                        loss=batch_loss,
                        is_augmented=is_augmented,
                        full_identifier=full_name,
                    )
                    
                    viz_results.append(result)
                    viz_count += 1
                    if result['is_correct']:
                        viz_correct_count += 1
                    
                    aug_tag = " [AUG]" if is_augmented else ""
                    status = "✓" if result['is_correct'] else f"✗ ({result['num_diff']} diff)"
                    print(f"  [VIZ {viz_count}] (idx={current_idx}) {base_name}{aug_tag}: {status}")
            
            # Increment global example index by batch size
            global_example_idx += batch_size
            
            # Early exit: if puzzle_ids specified and we've processed all of them
            if config.puzzle_ids is not None:
                max_target_idx = max(config.puzzle_ids)
                if global_example_idx > max_target_idx:
                    print(f"\n  All target puzzle_ids processed (max target idx: {max_target_idx}). Stopping early.")
                    break

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
            
            # Clean up to prevent memory accumulation
            del carry, loss, preds, metrics, all_finish

        # Print summary of processing
        print(f"\n{'='*60}")
        print("PROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"Total batches processed: {processed_batches}")
        print(f"Total batches skipped: {skipped_batches}")
        if config.puzzle_ids is not None:
            print(f"Target puzzle_ids: {len(config.puzzle_ids)} indices")
        print(f"{'='*60}")
        
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
                if count > 0:
                    reduced_metrics[set_name] = {k: v / count for k, v in m.items()}
                else:
                    reduced_metrics[set_name] = {k: 0.0 for k, v in m.items()}
            
            # Print average metrics (especially accuracy)
            print("\n" + "="*60)
            print("Average Metrics (for processed batches only):")
            print("="*60)
            for set_name, metrics_dict in reduced_metrics.items():
                print(f"\n{set_name}:", flush=True)
                for metric_name, metric_value in metrics_dict.items():
                    print(f"  {metric_name}: {metric_value:.4f}", flush=True)
            print("="*60)
        elif processed_batches == 0:
            print("\nNo batches were processed (all were skipped or filtered out)")
            reduced_metrics = {}
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
        
        # Create visualization summary if any visualizations were created
        if config.visualize and len(viz_results) > 0 and viz_output_dir is not None:
            print(f"\n{'='*60}")
            print("VISUALIZATION SUMMARY")
            print(f"{'='*60}")
            print(f"Total visualized: {viz_count}")
            print(f"Correct: {viz_correct_count} ({viz_correct_count/viz_count*100:.1f}%)")
            print(f"Incorrect: {viz_count - viz_correct_count} ({(viz_count-viz_correct_count)/viz_count*100:.1f}%)")
            print(f"Output directory: {viz_output_dir}")
            print(f"{'='*60}")
            
            # Save summary figure
            summary_path = os.path.join(viz_output_dir, "summary.png")
            create_evaluation_summary(viz_results, summary_path)
            
            # Save results JSON
            results_path = os.path.join(viz_output_dir, "visualization_results.json")
            with open(results_path, 'w') as f:
                json.dump({
                    'total': viz_count,
                    'correct': viz_correct_count,
                    'accuracy': viz_correct_count / viz_count if viz_count > 0 else 0,
                    'results': viz_results
                }, f, indent=2)
            print(f"Results saved to: {results_path}")

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
    
    # Load visualization context if visualization is enabled
    viz_context = None
    if config.visualize:
        print("\nLoading visualization context...")
        viz_context = load_visualization_context(config)
        print(f"  Loaded {len(viz_context['identifiers_map'])} identifiers")
        print(f"  Loaded {len(viz_context['test_puzzles'])} test puzzles")

    # Evaluation state
    eval_state = init_eval_state(config, eval_metadata)

    # Save code and config
    save_code_and_config(config)

    # Run evaluation
    print("Starting evaluation...")
    eval_state.model.eval()
    metrics = evaluate(config, eval_state, eval_loader, eval_metadata, evaluators, viz_context)

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
