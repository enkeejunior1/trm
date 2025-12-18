from typing import Optional, Any, Sequence, List, Dict, Tuple
from dataclasses import dataclass
import os
import math
import yaml
import shutil
import copy
import json

import torch
from torch import nn
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

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

# =============================================================================
# VISUALIZATION UTILITIES
# =============================================================================

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


def decode_arc_grid(tokens: np.ndarray, debug: bool = False) -> np.ndarray:
    """
    Decode tokenized ARC grid back to 2D grid.
    
    ARC tokenization:
    - PAD: 0, EOS: 1, Colors 0-9: tokens 2-11
    """
    if debug:
        print(f"  Token shape: {tokens.shape}, unique values: {np.unique(tokens)}")
    
    if tokens.ndim == 1:
        if len(tokens) != 900:
            tokens = tokens[:900] if len(tokens) > 900 else np.pad(tokens, (0, 900 - len(tokens)))
        grid_30x30 = tokens.reshape(30, 30).astype(np.int32)
    else:
        grid_30x30 = tokens.astype(np.int32)
    
    # Find valid rectangle (tokens 2-11)
    max_area, max_size = 0, (0, 0)
    num_c = 30
    for num_r in range(1, 31):
        for c in range(1, num_c + 1):
            x = grid_30x30[num_r - 1, c - 1]
            if x < 2 or x > 11:
                num_c = c - 1
                break
        area = num_r * num_c
        if area > max_area:
            max_area, max_size = area, (num_r, num_c)
    
    if max_size[0] == 0 or max_size[1] == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    
    cropped = grid_30x30[:max_size[0], :max_size[1]]
    return (cropped - 2).astype(np.uint8)


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
    
    ax.set_title(title, fontsize=10, fontweight='bold')
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


def visualize_single_prediction(
    puzzle_name: str,
    puzzle_id: int,
    demo_examples: List[Dict],
    test_input: np.ndarray,
    test_label: np.ndarray,
    prediction: np.ndarray,
    output_path: str,
    loss: float = None,
) -> Dict:
    """Visualize a single puzzle prediction."""
    num_demo = min(len(demo_examples), 4)
    total_rows = num_demo + 2
    
    fig, axes = plt.subplots(total_rows, 2, figsize=(8, total_rows * 2.5))
    if total_rows == 1:
        axes = axes[np.newaxis, :]
    
    # Demo examples
    for row, demo in enumerate(demo_examples[:num_demo]):
        in_g = np.array(demo['input'], dtype=np.uint8)
        out_g = np.array(demo['output'], dtype=np.uint8)
        visualize_grid(in_g, axes[row, 0], f"Demo {row+1} - Input")
        visualize_grid(out_g, axes[row, 1], f"Demo {row+1} - Output")
    
    # Decode grids
    input_grid = decode_arc_grid(test_input)
    label_grid = decode_arc_grid(test_label)
    pred_grid = decode_arc_grid(prediction)
    
    # Test target
    visualize_grid(input_grid, axes[num_demo, 0], "Test - Input")
    visualize_grid(label_grid, axes[num_demo, 1], "Test - Target", highlight_color='#0074D9')
    
    # Prediction
    is_correct = np.array_equal(label_grid, pred_grid)
    diff_mask, num_diff = compute_diff(label_grid, pred_grid)
    
    status = "CORRECT" if is_correct else f"WRONG ({num_diff} diff)"
    color = '#2ECC40' if is_correct else '#FF4136'
    
    visualize_grid(input_grid, axes[num_demo+1, 0], "Pred - Input")
    visualize_grid(pred_grid, axes[num_demo+1, 1], f"Pred - {status}",
                   highlight_color=color, show_diff=diff_mask if not is_correct else None)
    
    title = f"{puzzle_name} (ID: {puzzle_id})"
    if loss is not None:
        title += f" | Loss: {loss:.4f}"
    plt.suptitle(title, fontsize=11, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()
    
    return {
        'puzzle_name': puzzle_name,
        'puzzle_id': puzzle_id,
        'is_correct': is_correct,
        'num_diff': num_diff,
        'loss': loss,
    }


def create_summary_figure(results: List[Dict], output_path: str, title: str = "Evaluation Summary"):
    """Create summary visualization."""
    n = len(results)
    if n == 0:
        return
    
    correct = sum(1 for r in results if r.get('is_correct', False))
    incorrect = n - correct
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Accuracy bar chart
    ax = axes[0]
    bars = ax.bar(['Correct', 'Incorrect'], [correct, incorrect], 
                  color=['#2ECC40', '#FF4136'], alpha=0.8)
    ax.set_ylabel('Count')
    ax.set_title(f'Prediction Accuracy: {correct}/{n} ({100*correct/n:.1f}%)')
    for bar, val in zip(bars, [correct, incorrect]):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, val/2, str(val),
                   ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    
    # Loss distribution
    ax = axes[1]
    losses = [r['loss'] for r in results if r.get('loss') is not None]
    if losses:
        colors = ['#2ECC40' if r.get('is_correct') else '#FF4136' 
                  for r in results if r.get('loss') is not None]
        ax.bar(range(len(losses)), losses, color=colors, alpha=0.7)
        ax.set_xlabel('Example Index')
        ax.set_ylabel('Loss')
        ax.set_title(f'Loss per Example (avg: {np.mean(losses):.4f})')
        ax.axhline(np.mean(losses), color='orange', linestyle='--', label='Mean')
        ax.legend()
    
    plt.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def load_puzzle_metadata(data_path: str) -> Tuple[List[str], Dict]:
    """Load puzzle identifiers and test puzzles."""
    identifiers_path = os.path.join(data_path, "identifiers.json")
    test_puzzles_path = os.path.join(data_path, "test_puzzles.json")
    
    identifiers_map = []
    test_puzzles = {}
    
    if os.path.exists(identifiers_path):
        with open(identifiers_path, 'r') as f:
            identifiers_map = json.load(f)
    
    if os.path.exists(test_puzzles_path):
        with open(test_puzzles_path, 'r') as f:
            test_puzzles = json.load(f)
    
    return identifiers_map, test_puzzles


def visualize_batch(
    batch_data: Dict,
    identifiers_map: List[str],
    test_puzzles: Dict,
    output_dir: str,
    batch_idx: int,
) -> List[Dict]:
    """Visualize all predictions in a batch."""
    results = []
    
    predictions = batch_data.get('predictions', {})
    batch_info = batch_data.get('batch_info', {})
    loss = batch_data.get('loss', None)
    
    # Get predictions tensor
    if 'preds' in predictions:
        preds = predictions['preds'].numpy() if hasattr(predictions['preds'], 'numpy') else predictions['preds']
    else:
        print(f"  Warning: No 'preds' in predictions for batch {batch_idx}")
        return results
    
    inputs = batch_info.get('inputs')
    labels = batch_info.get('labels')
    puzzle_ids = batch_info.get('puzzle_identifiers')
    
    if inputs is None or labels is None:
        print(f"  Warning: Missing inputs/labels for batch {batch_idx}")
        return results
    
    inputs = inputs.numpy() if hasattr(inputs, 'numpy') else inputs
    labels = labels.numpy() if hasattr(labels, 'numpy') else labels
    puzzle_ids = puzzle_ids.numpy() if puzzle_ids is not None and hasattr(puzzle_ids, 'numpy') else puzzle_ids
    
    B = len(inputs)
    loss_val = loss.item() if loss is not None and hasattr(loss, 'item') else loss
    
    for i in range(B):
        # Get puzzle info
        pid = int(puzzle_ids[i]) if puzzle_ids is not None and i < len(puzzle_ids) else i
        if pid < len(identifiers_map):
            puzzle_name = identifiers_map[pid]
        else:
            puzzle_name = f"puzzle_{pid}"
        
        base_name = puzzle_name.split("|||")[0] if "|||" in puzzle_name else puzzle_name
        
        # Get demo examples
        demos = test_puzzles.get(base_name, {}).get("train", [])
        
        # Create visualization
        output_path = os.path.join(output_dir, f"batch{batch_idx:03d}_{i:03d}_{base_name}.png")
        result = visualize_single_prediction(
            puzzle_name=base_name,
            puzzle_id=pid,
            demo_examples=demos,
            test_input=inputs[i],
            test_label=labels[i],
            prediction=preds[i],
            output_path=output_path,
            loss=loss_val,
        )
        results.append(result)
        
        status = "OK" if result['is_correct'] else f"X ({result['num_diff']} diff)"
        print(f"    [{i+1}/{B}] {base_name}: {status}")
    
    return results

# =============================================================================
# END VISUALIZATION UTILITIES
# =============================================================================

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


def evaluate(
    config: EvalConfig,
    eval_state: EvalState,
    eval_loader: torch.utils.data.DataLoader,
    eval_metadata: PuzzleDatasetMetadata,
    evaluators: List[Any],
):
    """Run evaluation on the model"""
    MAX_BATCHES = 20
    RESULT_DIR = config.checkpoint_path.replace("ckpt/", "results/")
    os.makedirs(RESULT_DIR, exist_ok=True)
    reduced_metrics = None
    
    # ===========================================
    # VISUALIZATION SETUP
    # ===========================================
    VIZ_DIR = os.path.join(RESULT_DIR, "visualizations")
    os.makedirs(VIZ_DIR, exist_ok=True)
    
    # Load puzzle metadata for visualization
    data_path = config.data_paths[0] if config.data_paths else None
    if config.data_paths_test:
        data_path = config.data_paths_test[0]
    
    identifiers_map, test_puzzles = [], {}
    if data_path:
        try:
            identifiers_map, test_puzzles = load_puzzle_metadata(data_path)
            print(f"Loaded {len(identifiers_map)} puzzle identifiers for visualization")
        except Exception as e:
            print(f"Warning: Could not load puzzle metadata: {e}")
    
    all_viz_results = []
    # ===========================================

    with torch.inference_mode():
        return_keys = set(config.eval_save_outputs)
        # Always include "preds" for visualization
        return_keys.add("preds")
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
        
        for set_name, batch, global_batch_size in eval_loader:
            processed_batches += 1
            print(f"Processing batch {processed_batches}: {set_name}")
            
            # ===========================
            # LIMIT TO FIRST 20 BATCHES
            # ===========================
            if processed_batches > MAX_BATCHES:
                print(f"Reached max_batches={MAX_BATCHES}, stopping evaluation early.")
                break
            
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
            
            # Save predictions for this batch
            for k, v in preds.items():
                if k not in save_preds:
                    save_preds[k] = []
                save_preds[k].append(v.cpu())
            
            # Save predictions, loss, trajectories, and metrics for this batch immediately
            # if config.checkpoint_path is not None:
            stacked_trajectories_L = torch.stack(batch_trajectories_L, dim=0)
            stacked_trajectories_H = torch.stack(batch_trajectories_H, dim=0)
            batch_data = {
                'batch_index': processed_batches,
                'loss': loss.cpu(),
                'trajectories_L': stacked_trajectories_L.cpu(),
                'trajectories_H': stacked_trajectories_H.cpu(),
                'metrics': {k: v.cpu() for k, v in metrics.items()},
                'predictions': {k: v.cpu() for k, v in preds.items()},
                'batch_info': {k: v.cpu() for k, v in batch.items()}
            }
            
            # # Save predictions and relevant batch data
            # for collection_name, collection in [('preds', preds), ('batch', batch)]:
            #     for k, v in collection.items():
            #         if k in config.eval_save_outputs:
            #             if collection_name == 'preds':
            #                 batch_data['predictions'][k] = v.cpu()
            #             else:
            #                 batch_data['batch_info'][k] = v.cpu()
            
            # os.makedirs(config.checkpoint_path.replace('ckpt/', 'results/'), exist_ok=True)
            # batch_path = os.path.join(
            #     config.checkpoint_path.replace('ckpt/', 'results/'),
            #     f"batch_data_{processed_batches:04d}.pt"
            # )
            
            # Save batch data to RESULT_DIR (not ckpt/)
            batch_path = os.path.join(RESULT_DIR, f"batch_data_{processed_batches:04d}.pt")
            torch.save(batch_data, batch_path)
            
            print(f"  Saved batch data to {batch_path}")
            print(f"    Loss: {loss.item():.4f}")
            print(f"    Metrics: {metrics}")
            print(f"    Predictions keys: {list(batch_data['predictions'].keys())}")
            print(f"    Batch info keys: {list(batch_data['batch_info'].keys())}")
            print(f"    Trajectories_L: {stacked_trajectories_L.shape}")
            print(f"    Trajectories_H: {stacked_trajectories_H.shape}")
            
            # ===========================================
            # VISUALIZE THIS BATCH
            # ===========================================
            if identifiers_map:
                print(f"  Creating visualizations for batch {processed_batches}...")
                try:
                    batch_viz_results = visualize_batch(
                        batch_data=batch_data,
                        identifiers_map=identifiers_map,
                        test_puzzles=test_puzzles,
                        output_dir=VIZ_DIR,
                        batch_idx=processed_batches,
                    )
                    all_viz_results.extend(batch_viz_results)
                    
                    correct = sum(1 for r in batch_viz_results if r.get('is_correct', False))
                    print(f"  Batch {processed_batches} visualization: {correct}/{len(batch_viz_results)} correct")
                except Exception as e:
                    print(f"  Warning: Visualization failed for batch {processed_batches}: {e}")
            # ===========================================
            
            del batch_data, stacked_trajectories_L, stacked_trajectories_H

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

        # # Save predictions
        # if config.checkpoint_path is not None:
        #     os.makedirs(config.checkpoint_path, exist_ok=True)
            
        #     if len(save_preds):
        #         torch.save(
        #             save_preds, os.path.join(config.checkpoint_path, f"all_preds.pt")
        #         )
        # del save_preds
        
        # Save concatenated preds into RESULT_DIR
        if len(save_preds):
            torch.save(save_preds, os.path.join(RESULT_DIR, "all_preds.pt"))
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
        else:
            raise ValueError("No metrics found")

        # Run evaluators
        print(f"\nRunning {len(evaluators)} evaluator(s)...")
        
        for i, evaluator in enumerate(evaluators):
            print(f"Running evaluator {i+1}/{len(evaluators)}: {evaluator.__class__.__name__}")
                
            # # Path for saving
            # evaluator_save_path = None
            # if config.checkpoint_path is not None:
            #     evaluator_save_path = os.path.join(
            #         config.checkpoint_path,
            #         f"evaluator_{evaluator.__class__.__name__}",
            #     )
            # os.makedirs(evaluator_save_path, exist_ok=True)
            
            # Evaluator output directory under RESULTS (not ckpt/)
            evaluator_save_path = os.path.join(
                RESULT_DIR,
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
        
        # ===========================================
        # CREATE VISUALIZATION SUMMARY
        # ===========================================
        if all_viz_results:
            print("\n" + "="*60)
            print("VISUALIZATION SUMMARY")
            print("="*60)
            
            total = len(all_viz_results)
            correct = sum(1 for r in all_viz_results if r.get('is_correct', False))
            
            print(f"Total examples visualized: {total}")
            print(f"Correct predictions: {correct} ({100*correct/total:.1f}%)")
            print(f"Incorrect predictions: {total - correct} ({100*(total-correct)/total:.1f}%)")
            
            # Create summary figure
            summary_path = os.path.join(VIZ_DIR, "evaluation_summary.png")
            try:
                create_summary_figure(all_viz_results, summary_path, title="Evaluation Summary")
                print(f"\nSummary figure saved to: {summary_path}")
            except Exception as e:
                print(f"Warning: Could not create summary figure: {e}")
            
            # Save results to JSON
            results_json_path = os.path.join(VIZ_DIR, "visualization_results.json")
            try:
                with open(results_json_path, 'w') as f:
                    json.dump({
                        'summary': {
                            'total': total,
                            'correct': correct,
                            'accuracy': correct / total if total > 0 else 0,
                        },
                        'results': all_viz_results,
                    }, f, indent=2)
                print(f"Results JSON saved to: {results_json_path}")
            except Exception as e:
                print(f"Warning: Could not save results JSON: {e}")
            
            print(f"\nAll visualizations saved to: {VIZ_DIR}")
            print("="*60)
        # ===========================================

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
