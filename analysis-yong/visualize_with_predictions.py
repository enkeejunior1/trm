"""
Visualization script for ARC-AGI-1 puzzles with TRM model predictions.

This script visualizes:
- Demo examples (input → output)
- Test example (input → target ground truth)
- Model prediction (input → model output)  <-- NEW ROW

Usage:
    python analysis-yong/visualize_with_predictions.py \
        --data_path data/arc1concept-aug-0 \
        --model_data_path data/arc1concept-aug-1000 \
        --checkpoint ckpt/arc_v1_public/step_518071 \
        --config_path ckpt/arc_v1_public \
        --num_puzzles 10
"""
import os
import sys
import json
import argparse
import ast
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import torch
from torch import nn

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from puzzle_dataset import PuzzleDatasetConfig
from dataset.common import PuzzleDatasetMetadata
from models.losses import IGNORE_LABEL_ID
from utils.functions import load_model_class

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


def decode_arc_grid(tokens: np.ndarray, debug: bool = False) -> np.ndarray:
    """
    Decode tokenized ARC grid back to 2D grid
    
    ARC tokenization:
    - PAD: 0
    - EOS: 1  
    - Colors 0-9: tokens 2-11
    
    Grid is 30x30, with EOS tokens marking the boundary of actual content.
    """
    if debug:
        unique_values = np.unique(tokens)
        print(f"  Token shape: {tokens.shape}, unique values: {unique_values}")
    
    grid_30x30 = tokens.reshape(30, 30).astype(np.int32)
    
    # Find maximum-sized rectangle without any EOS token (value < 2 or value > 11) inside
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
    
    # Highlight border if specified
    if highlight_color:
        for spine in ax.spines.values():
            spine.set_edgecolor(highlight_color)
            spine.set_linewidth(3)
    
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)


def build_identifier_mapping(train_data_path: str, test_data_path: str, use_augmented: bool = True, seed: int = 42) -> Dict[int, int]:
    """Build mapping from test dataset puzzle_identifiers to train dataset puzzle_identifiers.
    
    Args:
        use_augmented: If True, map to AUGMENTED embeddings (properly trained).
                       If False, map to original embeddings (barely trained).
    """
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
            # Find ALL augmented versions of this puzzle
            augmented_ids = [idx for idx, name in enumerate(train_identifiers) 
                           if name.startswith(identifier + '|||')]
            
            if augmented_ids:
                # Randomly select one augmented version
                selected_id = random.choice(augmented_ids)
                mapping[test_idx] = selected_id
            else:
                # Fallback to original
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


def load_model(config_path: str, checkpoint_path: str, model_metadata: PuzzleDatasetMetadata):
    """Load TRM model from checkpoint."""
    import yaml
    from omegaconf import OmegaConf
    
    # Load config
    config_file = os.path.join(config_path, "all_config.yaml")
    with open(config_file, 'r') as f:
        config = OmegaConf.load(f)
    
    # Build model config - extract arch settings excluding nested configs
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
    
    # Load model classes
    model_cls = load_model_class(arch_name)
    loss_head_cls = load_model_class(loss_name)
    
    # Create model on device
    with device:
        model = model_cls(model_cfg)
        model = loss_head_cls(model, **loss_config)
    
    # Move model to device first
    model = model.to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # The checkpoint was saved from a compiled model, so keys have _orig_mod prefix.
    # We need to strip that prefix to load into an uncompiled model.
    # Also ensure all tensors are on the correct device.
    new_state_dict = {}
    for k, v in state_dict.items():
        # Remove _orig_mod. prefix if present
        new_key = k.replace("_orig_mod.", "")
        # Ensure tensor is on correct device
        if isinstance(v, torch.Tensor):
            new_state_dict[new_key] = v.to(device)
        else:
            new_state_dict[new_key] = v
    
    # Check puzzle embedding shape
    puzzle_emb_name = "model.inner.puzzle_emb.weights"
    if puzzle_emb_name in new_state_dict:
        expected_shape = model.model.puzzle_emb.weights.shape
        if new_state_dict[puzzle_emb_name].shape != expected_shape:
            raise ValueError(f"Puzzle embedding mismatch: {new_state_dict[puzzle_emb_name].shape} vs {expected_shape}")
    
    # Load state dict without assign=True to avoid device issues
    model.load_state_dict(new_state_dict, strict=True)
    
    # Ensure everything is on device after loading
    model = model.to(device)
    model.eval()
    return model


def move_carry_to_device(carry, device):
    """Recursively move all tensors in a carry object to device."""
    import dataclasses
    if dataclasses.is_dataclass(carry):
        # Get all fields and move tensors to device
        updates = {}
        for field in dataclasses.fields(carry):
            value = getattr(carry, field.name)
            if isinstance(value, torch.Tensor):
                updates[field.name] = value.to(device)
            elif dataclasses.is_dataclass(value):
                updates[field.name] = move_carry_to_device(value, device)
            elif isinstance(value, dict):
                updates[field.name] = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                                        for k, v in value.items()}
            else:
                updates[field.name] = value
        return type(carry)(**updates)
    return carry


def run_inference(model: nn.Module, inputs: torch.Tensor, puzzle_identifiers: torch.Tensor) -> np.ndarray:
    """Run model inference on a single example."""
    # Ensure model is on device
    model = model.to(device)
    
    batch = {
        "inputs": inputs.unsqueeze(0).to(device),
        "labels": torch.zeros_like(inputs).unsqueeze(0).to(device),  # Dummy labels
        "puzzle_identifiers": puzzle_identifiers.unsqueeze(0).to(device),
    }
    
    with torch.inference_mode():
        carry = model.initial_carry(batch)
        # Move carry to device (initial_carry creates CPU tensors)
        carry = move_carry_to_device(carry, device)
        
        # Run until halted
        max_steps = 100
        for step in range(max_steps):
            carry, loss, metrics, preds, all_finish = model(
                carry=carry, batch=batch, return_keys={"preds"}
            )
            if all_finish:
                break
        
        predictions = preds["preds"].cpu().numpy()[0]  # [seq_len]
    
    return predictions


def visualize_puzzle_with_prediction(
    puzzle_id: int,
    puzzle_name: str,
    demo_examples: list,
    test_input: np.ndarray,
    test_label: np.ndarray,
    test_prediction: np.ndarray,
    output_dir: str,
    puzzle_idx: int,
):
    """
    Visualize a complete ARC puzzle with demonstration, test target, and model prediction.
    """
    num_demo = len(demo_examples)
    total_rows = num_demo + 2  # demos + test target + model prediction
    
    # Create figure
    fig_width = 10
    fig_height = total_rows * 3.5
    fig, axes = plt.subplots(total_rows, 2, figsize=(fig_width, fig_height))
    
    if total_rows == 1:
        axes = axes[np.newaxis, :]
    
    # Plot demonstration examples
    for ex_row, demo in enumerate(demo_examples):
        input_grid = np.array(demo['input'], dtype=np.uint8)
        output_grid = np.array(demo['output'], dtype=np.uint8)
        
        visualize_arc_grid(input_grid, axes[ex_row, 0], f"Demo {ex_row+1} - Input")
        visualize_arc_grid(output_grid, axes[ex_row, 1], f"Demo {ex_row+1} - Output")
    
    # Plot test example (ground truth)
    test_row = num_demo
    test_input_grid = decode_arc_grid(test_input)
    test_label_grid = decode_arc_grid(test_label)
    
    visualize_arc_grid(test_input_grid, axes[test_row, 0], "Test - Input")
    visualize_arc_grid(test_label_grid, axes[test_row, 1], "Test - Target (Ground Truth)", 
                       highlight_color='blue')
    
    # Plot model prediction
    pred_row = num_demo + 1
    pred_grid = decode_arc_grid(test_prediction)
    
    # Check if prediction matches target
    is_correct = np.array_equal(test_label_grid, pred_grid)
    status = "✓ CORRECT" if is_correct else "✗ INCORRECT"
    status_color = 'green' if is_correct else 'red'
    
    visualize_arc_grid(test_input_grid, axes[pred_row, 0], "Model - Input (same)")
    visualize_arc_grid(pred_grid, axes[pred_row, 1], f"Model - Prediction ({status})",
                       highlight_color=status_color)
    
    # Title
    plt.suptitle(
        f"Puzzle: {puzzle_name} (ID: {puzzle_id})\n"
        f"{num_demo} demo examples | Test: {status}",
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save
    save_path = os.path.join(output_dir, f"puzzle_{puzzle_idx:03d}_{puzzle_name}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path} [{status}]")
    plt.close()
    
    return is_correct


def main():
    parser = argparse.ArgumentParser(description='Visualize ARC puzzles with TRM model predictions')
    parser.add_argument('--data_path', type=str, 
                        default='data/arc1concept-aug-0',
                        help='Path to test dataset (non-augmented)')
    parser.add_argument('--model_data_path', type=str,
                        default='data/arc1concept-aug-1000',
                        help='Path to model dataset (for puzzle embeddings)')
    parser.add_argument('--checkpoint', type=str,
                        default='ckpt/arc_v1_public/step_518071',
                        help='Path to model checkpoint')
    parser.add_argument('--config_path', type=str,
                        default='ckpt/arc_v1_public',
                        help='Path to config directory')
    parser.add_argument('--output_dir', type=str,
                        default='results-analysis-noaug/visualizations_with_predictions',
                        help='Output directory for visualizations')
    parser.add_argument('--num_puzzles', type=int, default=10,
                        help='Number of puzzles to visualize')
    parser.add_argument('--puzzle_ids', type=str, default=None,
                        help='Comma-separated list of puzzle IDs (e.g., "1,2,3,4,5")')
    parser.add_argument('--start_id', type=int, default=None,
                        help='Start puzzle ID')
    parser.add_argument('--end_id', type=int, default=None,
                        help='End puzzle ID (exclusive)')
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*70)
    print("ARC Puzzle Visualization with TRM Model Predictions")
    print("="*70)
    print(f"Test data: {args.data_path}")
    print(f"Model data: {args.model_data_path}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {args.output_dir}")
    print("="*70)
    
    # Load model metadata (from aug-1000)
    print("\nLoading model metadata...")
    with open(os.path.join(args.model_data_path, "train", "dataset.json"), 'r') as f:
        model_metadata = PuzzleDatasetMetadata(**json.load(f))
    print(f"  Model expects {model_metadata.num_puzzle_identifiers} puzzle identifiers")
    
    # Build identifier mapping (aug-0 -> aug-1000)
    print("\nBuilding identifier mapping...")
    identifier_mapping = build_identifier_mapping(args.model_data_path, args.data_path)
    print(f"  Mapped {len(identifier_mapping)} identifiers")
    
    # Load model
    print("\nLoading TRM model...")
    model = load_model(args.config_path, args.checkpoint, model_metadata)
    print("  Model loaded successfully!")
    
    # Load test data identifiers and puzzles
    print("\nLoading puzzle data...")
    with open(os.path.join(args.data_path, "identifiers.json"), 'r') as f:
        identifiers_map = json.load(f)
    
    with open(os.path.join(args.data_path, "test_puzzles.json"), 'r') as f:
        test_puzzles = json.load(f)
    
    # Load test dataset
    test_data_dir = os.path.join(args.data_path, "test")
    inputs = np.load(os.path.join(test_data_dir, "all__inputs.npy"))
    labels = np.load(os.path.join(test_data_dir, "all__labels.npy"))
    puzzle_ids_arr = np.load(os.path.join(test_data_dir, "all__puzzle_identifiers.npy"))
    puzzle_indices = np.load(os.path.join(test_data_dir, "all__puzzle_indices.npy"))
    
    print(f"  Test inputs: {inputs.shape}")
    print(f"  Test labels: {labels.shape}")
    
    # Group by puzzle
    puzzle_data = {}
    for i in range(len(inputs)):
        if i + 1 < len(puzzle_indices):
            puzzle_idx = int(puzzle_indices[i + 1])
            if puzzle_idx < len(identifiers_map):
                if puzzle_idx not in puzzle_data:
                    puzzle_data[puzzle_idx] = []
                puzzle_data[puzzle_idx].append({
                    'input': inputs[i],
                    'label': labels[i],
                    'original_puzzle_id': puzzle_ids_arr[i]
                })
    
    # Filter to original puzzles only
    original_puzzles = []
    for identifier_idx, examples in puzzle_data.items():
        puzzle_name = identifiers_map[identifier_idx]
        if "|||" not in puzzle_name and puzzle_name != "<blank>":
            original_puzzles.append((identifier_idx, puzzle_name, examples))
    
    original_puzzles.sort(key=lambda x: x[0])
    print(f"  Found {len(original_puzzles)} original puzzles")
    
    # Select puzzles
    if args.puzzle_ids:
        selected_ids = [int(x.strip()) for x in args.puzzle_ids.split(',')]
        selected_puzzles = [(idx, name, ex) for idx, name, ex in original_puzzles if idx in selected_ids]
    elif args.start_id is not None or args.end_id is not None:
        start = args.start_id if args.start_id is not None else 0
        end = args.end_id if args.end_id is not None else len(identifiers_map)
        selected_puzzles = [(idx, name, ex) for idx, name, ex in original_puzzles if start <= idx < end]
    else:
        selected_puzzles = original_puzzles[:args.num_puzzles]
    
    print(f"\nVisualizing {len(selected_puzzles)} puzzles...")
    print("="*70)
    
    # Visualize each puzzle
    correct_count = 0
    total_count = 0
    
    for viz_idx, (identifier_idx, puzzle_name, test_examples) in enumerate(selected_puzzles):
        print(f"\nPuzzle {viz_idx+1}/{len(selected_puzzles)}: {puzzle_name} (ID: {identifier_idx})")
        
        # Get demo examples
        demo_examples = test_puzzles.get(puzzle_name, {}).get("train", [])
        
        # Get first test example
        test_ex = test_examples[0]
        test_input = test_ex['input']
        test_label = test_ex['label']
        
        # Remap puzzle identifier for model
        remapped_id = identifier_mapping.get(identifier_idx, 0)
        
        # Run inference
        print(f"  Running inference (puzzle_id {identifier_idx} -> remapped {remapped_id})...")
        prediction = run_inference(
            model,
            torch.from_numpy(test_input.astype(np.int32)),
            torch.tensor([remapped_id], dtype=torch.int32)
        )
        
        # Visualize
        is_correct = visualize_puzzle_with_prediction(
            puzzle_id=identifier_idx,
            puzzle_name=puzzle_name,
            demo_examples=demo_examples,
            test_input=test_input,
            test_label=test_label,
            test_prediction=prediction,
            output_dir=args.output_dir,
            puzzle_idx=viz_idx,
        )
        
        total_count += 1
        if is_correct:
            correct_count += 1
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total puzzles visualized: {total_count}")
    print(f"Correct predictions: {correct_count} ({correct_count/total_count*100:.1f}%)")
    print(f"Incorrect predictions: {total_count - correct_count} ({(total_count-correct_count)/total_count*100:.1f}%)")
    print(f"\nVisualizations saved to: {args.output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
