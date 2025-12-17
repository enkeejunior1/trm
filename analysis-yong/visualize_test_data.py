"""
Simplified visualization script for ARC-AGI-1 test data only
No model inference - just visualize the test dataset
"""
import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

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
    
    Args:
        tokens: [900] array of token IDs (flattened 30x30 grid)
        debug: if True, print debugging info
    
    Returns:
        grid: [H, W] numpy array of color IDs (0-9), cropped to actual content
    """
    if debug:
        unique_values = np.unique(tokens)
        print(f"  Token shape: {tokens.shape}, unique values: {unique_values}, min: {tokens.min()}, max: {tokens.max()}")
    
    grid_30x30 = tokens.reshape(30, 30).astype(np.int32)
    
    # Find maximum-sized rectangle without any EOS token (value < 2 or value > 11) inside
    max_area = 0
    max_size = (0, 0)
    nr, nc = 30, 30
    
    num_c = nc
    for num_r in range(1, nr + 1):
        # Scan for maximum c
        for c in range(1, num_c + 1):
            x = grid_30x30[num_r - 1, c - 1]
            # Check if this is NOT a valid color token
            if (x < 2) or (x > 11):
                num_c = c - 1
                break
        
        area = num_r * num_c
        if area > max_area:
            max_area = area
            max_size = (num_r, num_c)
    
    if debug:
        print(f"  Cropped size: {max_size[0]}x{max_size[1]}")
    
    # Crop to actual content size
    if max_size[0] == 0 or max_size[1] == 0:
        if debug:
            print(f"  Warning: Empty grid detected!")
        return np.zeros((1, 1), dtype=np.uint8)
    
    cropped = grid_30x30[:max_size[0], :max_size[1]]
    
    # Convert from tokens (2-11) to colors (0-9)
    cropped = (cropped - 2).astype(np.uint8)
    
    return cropped


def visualize_arc_grid(grid: np.ndarray, ax: plt.Axes, title: str = ""):
    """
    Visualize an ARC grid
    
    Args:
        grid: [H, W] numpy array of color IDs (0-9)
        ax: matplotlib axis to plot on
        title: title for the plot
    """
    H, W = grid.shape
    
    # Create colormap
    cmap = ListedColormap(ARC_COLORS)
    
    # Plot grid
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=9, aspect='equal')
    
    # Add grid lines
    for i in range(H + 1):
        ax.axhline(i - 0.5, color='white', linewidth=0.5, alpha=0.3)
    for j in range(W + 1):
        ax.axvline(j - 0.5, color='white', linewidth=0.5, alpha=0.3)
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)  # Flip y-axis


def visualize_puzzle(
    puzzle_id: int,
    puzzle_name: str,
    demo_examples: list,
    test_input: np.ndarray,
    test_label: np.ndarray,
    output_dir: str,
    puzzle_idx: int,
):
    """
    Visualize a complete ARC puzzle with demonstration and test examples
    
    Args:
        puzzle_id: unique puzzle identifier
        puzzle_name: puzzle name (without augmentation info)
        demo_examples: list of dicts with 'input' and 'output' from test_puzzles.json
        test_input: test input grid tokens
        test_label: test label grid tokens
        output_dir: directory to save visualization
        puzzle_idx: index for filename
    """
    num_demo = len(demo_examples)
    total_rows = num_demo + 1  # demos + 1 test
    
    # Create figure
    fig_width = 10  # 2 columns (input, output)
    fig_height = total_rows * 4
    fig, axes = plt.subplots(total_rows, 2, figsize=(fig_width, fig_height))
    
    # Handle single row case
    if total_rows == 1:
        axes = axes[np.newaxis, :]
    
    # Plot demonstration examples
    for ex_row, demo in enumerate(demo_examples):
        # Convert from list to numpy array
        input_grid = np.array(demo['input'], dtype=np.uint8)
        output_grid = np.array(demo['output'], dtype=np.uint8)
        
        # Visualize
        visualize_arc_grid(input_grid, axes[ex_row, 0], 
                         f"Demo {ex_row+1} - Input")
        visualize_arc_grid(output_grid, axes[ex_row, 1], 
                         f"Demo {ex_row+1} - Output")
    
    # Plot test example
    test_row = num_demo
    test_input_grid = decode_arc_grid(test_input, debug=(puzzle_idx == 0))
    test_label_grid = decode_arc_grid(test_label, debug=(puzzle_idx == 0))
    
    visualize_arc_grid(test_input_grid, axes[test_row, 0], 
                     f"Test - Input")
    visualize_arc_grid(test_label_grid, axes[test_row, 1], 
                     f"Test - Target")
    
    # Title
    plt.suptitle(f"Puzzle: {puzzle_name} (ID: {puzzle_id})\n{num_demo} demo + 1 test example", 
                 fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save
    save_path = os.path.join(output_dir, f"puzzle_{puzzle_idx:03d}_{puzzle_name}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved puzzle {puzzle_idx}: {puzzle_name} ({num_demo} demo + 1 test) -> {save_path}")
    plt.close()


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Visualize ARC-AGI-1 test puzzles')
    parser.add_argument('--data_path', type=str, 
                        default='/vast/projects/jgu32/lab/yhpark/trm/data/arc1concept-aug-1000',
                        help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str,
                        default='/vast/projects/jgu32/lab/yhpark/trm/results-analysis/visualizations',
                        help='Output directory for visualizations')
    parser.add_argument('--num_puzzles', type=int, default=10,
                        help='Number of puzzles to visualize')
    parser.add_argument('--start_id', type=int, default=None,
                        help='Start puzzle ID (0-indexed)')
    parser.add_argument('--end_id', type=int, default=None,
                        help='End puzzle ID (exclusive, 0-indexed)')
    parser.add_argument('--puzzle_ids', type=str, default=None,
                        help='Comma-separated list of puzzle IDs to visualize (e.g., "5,10,25,100")')
    parser.add_argument('--random', action='store_true',
                        help='Randomly select puzzles')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for random selection')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load metadata
    identifiers_path = os.path.join(args.data_path, "identifiers.json")
    test_puzzles_path = os.path.join(args.data_path, "test_puzzles.json")
    
    print(f"Loading puzzle metadata from: {args.data_path}")
    print(f"  - identifiers.json: {identifiers_path}")
    print(f"  - test_puzzles.json: {test_puzzles_path}")
    
    with open(identifiers_path, "r") as f:
        identifiers_map = json.load(f)  # list where index is puzzle_id, value is puzzle name
    
    with open(test_puzzles_path, "r") as f:
        test_puzzles = json.load(f)  # dict of puzzle_name -> {train: [...], test: [...]}
    
    print(f"Loaded {len(identifiers_map)} puzzle identifiers")
    print(f"Loaded {len(test_puzzles)} original test puzzles")
    
    # Load test data
    test_data_dir = os.path.join(args.data_path, "test")
    print(f"\nLoading test dataset from: {test_data_dir}")
    
    inputs = np.load(os.path.join(test_data_dir, "all__inputs.npy"))
    labels = np.load(os.path.join(test_data_dir, "all__labels.npy"))
    puzzle_ids = np.load(os.path.join(test_data_dir, "all__puzzle_identifiers.npy"))
    puzzle_indices = np.load(os.path.join(test_data_dir, "all__puzzle_indices.npy"))
    
    print(f"  Test inputs shape: {inputs.shape}")
    print(f"  Test labels shape: {labels.shape}")
    print(f"  Test puzzle IDs shape: {puzzle_ids.shape}")
    print(f"  Test puzzle indices shape: {puzzle_indices.shape}")
    
    # Group test examples by puzzle identifier index
    # IMPORTANT: example i belongs to puzzle at puzzle_indices[i+1]
    puzzle_data = {}
    num_examples = len(inputs)
    print(f"  Processing {num_examples} test examples")
    
    for i in range(num_examples):
        # Example i belongs to puzzle_indices[i+1]
        if i + 1 < len(puzzle_indices):
            puzzle_idx = int(puzzle_indices[i + 1])
            # puzzle_idx directly indexes into identifiers_map
            if puzzle_idx < len(identifiers_map):
                if puzzle_idx not in puzzle_data:
                    puzzle_data[puzzle_idx] = []
                puzzle_data[puzzle_idx].append({
                    'input': inputs[i],
                    'label': labels[i]
                })
    
    print(f"\nFound {len(puzzle_data)} unique puzzles in test set")
    
    # Filter to only original puzzles (no augmentation)
    # puzzle_data keys are indices into identifiers_map
    original_puzzles = []
    for identifier_idx, examples in puzzle_data.items():
        puzzle_name = identifiers_map[identifier_idx]
        # Skip augmented puzzles (those with |||) and blank
        if "|||" not in puzzle_name and puzzle_name != "<blank>":
            original_puzzles.append((identifier_idx, puzzle_name, examples))
    
    # Sort by puzzle ID for consistent ordering
    original_puzzles.sort(key=lambda x: x[0])
    
    print(f"Found {len(original_puzzles)} original (non-augmented) puzzles")
    
    # Select puzzles based on arguments
    # Note: puzzle IDs here refer to identifier indices
    if args.puzzle_ids:
        # Specific puzzle identifier indices
        selected_ids = [int(x.strip()) for x in args.puzzle_ids.split(',')]
        selected_puzzles = [(idx, name, examples) for idx, name, examples in original_puzzles if idx in selected_ids]
        print(f"\nSelected {len(selected_puzzles)} puzzles by identifier index: {selected_ids}")
    elif args.start_id is not None or args.end_id is not None:
        # Identifier index range
        start = args.start_id if args.start_id is not None else 0
        end = args.end_id if args.end_id is not None else len(identifiers_map)
        selected_puzzles = [(idx, name, examples) for idx, name, examples in original_puzzles if start <= idx < end]
        print(f"\nSelected puzzles with identifier index range [{start}, {end}): {len(selected_puzzles)} puzzles")
    elif args.random:
        # Random selection
        np.random.seed(args.seed)
        indices = np.random.choice(len(original_puzzles), 
                                   size=min(args.num_puzzles, len(original_puzzles)), 
                                   replace=False)
        selected_puzzles = [original_puzzles[i] for i in sorted(indices)]
        print(f"\nRandomly selected {len(selected_puzzles)} puzzles (seed={args.seed})")
    else:
        # First N puzzles
        selected_puzzles = original_puzzles[:args.num_puzzles]
        print(f"\nSelected first {len(selected_puzzles)} puzzles")
    
    # Visualize
    print(f"\n{'='*60}")
    print(f"Visualizing {len(selected_puzzles)} puzzles")
    print(f"{'='*60}\n")
    
    for viz_idx, (identifier_idx, puzzle_name, test_examples) in enumerate(selected_puzzles):
        # Get demonstration examples from test_puzzles.json
        # puzzle_name should match the name in test_puzzles
        if puzzle_name in test_puzzles:
            demo_examples = test_puzzles[puzzle_name].get("train", [])
            print(f"  Found {len(demo_examples)} demo examples for puzzle {puzzle_name} (idx: {identifier_idx})")
        else:
            print(f"  Warning: puzzle {puzzle_name} not found in test_puzzles.json (idx: {identifier_idx})")
            demo_examples = []
        
        # Visualize first test example for this puzzle
        visualize_puzzle(
            puzzle_id=identifier_idx,
            puzzle_name=puzzle_name,
            demo_examples=demo_examples,
            test_input=test_examples[0]['input'],
            test_label=test_examples[0]['label'],
            output_dir=args.output_dir,
            puzzle_idx=viz_idx
        )
    
    print(f"\n{'='*60}")
    print(f"Visualization complete! Results saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
