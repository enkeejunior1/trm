"""
Evaluation script for non-augmented validation data with a model trained on augmented data.

This script solves the puzzle embedding mismatch problem by:
1. Loading the model with aug-1000 dataset metadata (maintains 876,406 puzzle embeddings)
2. Loading test data from aug-0 (non-augmented validation)
3. Remapping puzzle_identifiers from aug-0 indices (1-400) to aug-1000 indices

Usage:
    python eval_noaug.py \
      --config-path=ckpt/arc_v1_public \
      --config-name=all_config \
      load_checkpoint=ckpt/arc_v1_public/step_518071 \
      data_paths="[data/arc1concept-aug-1000]" \
      data_paths_test="[data/arc1concept-aug-0]" \
      global_batch_size=1 \
      checkpoint_path=results/arc_v1_public/step_518071_eval_noaug \
      evaluators="[]"
"""

from typing import Optional, Any, Sequence, List, Dict
from dataclasses import dataclass
import os
import math
import yaml
import shutil
import copy
import json
import ast

import torch
from torch import nn
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
import numpy as np

import tqdm
import wandb
import coolname
import hydra
import pydantic
from omegaconf import DictConfig

from puzzle_dataset import PuzzleDatasetConfig
from dataset.common import PuzzleDatasetMetadata
from models.losses import IGNORE_LABEL_ID
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
    data_paths: List[str]  # Used for MODEL initialization (aug-1000)
    data_paths_test: List[str] = []  # Used for TEST data (aug-0)
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
    puzzle_ids: Optional[List[int]] = None  # Filter to specific puzzle IDs
    
    # Puzzle embedding mode for non-augmented evaluation
    # Options: "original", "first_augmented", "random_augmented"
    embedding_mode: str = "original"


@dataclass
class EvalState:
    model: nn.Module
    carry: Any


def build_identifier_mapping(
    train_data_path: str, 
    test_data_path: str, 
    embedding_mode: str = "original",
    seed: int = 42
) -> Dict[int, int]:
    """
    Build a mapping from test dataset puzzle_identifiers to train dataset puzzle_identifiers.
    
    Args:
        train_data_path: Path to the training dataset (e.g., data/arc1concept-aug-1000)
        test_data_path: Path to the test dataset (e.g., data/arc1concept-aug-0)
        embedding_mode: How to select puzzle embeddings from train dataset:
            - "original": Use the ORIGINAL (non-augmented) embedding from aug-1000.
                          Consistent, but these embeddings saw 1/1001 of training data.
            - "first_augmented": Use the FIRST augmented embedding.
                                 Consistent and well-trained.
            - "random_augmented": Randomly select an augmented embedding (causes variance).
        seed: Random seed for selecting augmented versions (only used when mode="random_augmented")
    
    Returns:
        Dictionary mapping test puzzle_id -> train puzzle_id
    """
    import random
    random.seed(seed)
    
    # Load identifiers from both datasets
    with open(os.path.join(test_data_path, "identifiers.json"), 'r') as f:
        test_identifiers = json.load(f)
    
    with open(os.path.join(train_data_path, "identifiers.json"), 'r') as f:
        content = f.read()
        train_identifiers = ast.literal_eval(content)
    
    # Build mapping: test index -> train index
    mapping = {}
    mapping_info = {}  # For logging
    
    print(f"\n{'='*60}")
    print(f"Building identifier mapping with mode: '{embedding_mode}'")
    print(f"{'='*60}")
    
    for test_idx, identifier in enumerate(test_identifiers):
        if identifier == '<blank>':
            mapping[test_idx] = 0
            continue
        
        # Find ORIGINAL embedding index in train dataset
        original_idx = None
        if identifier in train_identifiers:
            original_idx = train_identifiers.index(identifier)
        
        # Find ALL augmented versions of this puzzle (format: "puzzle_name|||...")
        augmented_ids = [idx for idx, name in enumerate(train_identifiers) 
                        if name.startswith(identifier + '|||')]
        
        # Select embedding based on mode
        if embedding_mode == "original":
            # Use ORIGINAL embedding (same as validation data)
            if original_idx is not None:
                mapping[test_idx] = original_idx
                mapping_info[test_idx] = {
                    'name': identifier,
                    'mode': 'original',
                    'selected': original_idx,
                    'selected_name': identifier,
                    'num_augmented_available': len(augmented_ids)
                }
            else:
                mapping[test_idx] = 0
                print(f"Warning: '{identifier}' not found in train data, mapping to blank")
                
        elif embedding_mode == "first_augmented":
            # Use FIRST augmented embedding (consistent + well-trained)
            if augmented_ids:
                selected_id = augmented_ids[0]  # Always first
                mapping[test_idx] = selected_id
                mapping_info[test_idx] = {
                    'name': identifier,
                    'mode': 'first_augmented',
                    'selected': selected_id,
                    'selected_name': train_identifiers[selected_id],
                    'num_augmented_available': len(augmented_ids)
                }
            elif original_idx is not None:
                # Fallback to original if no augmented versions
                mapping[test_idx] = original_idx
                print(f"Warning: No augmented versions for '{identifier}', using original (ID={original_idx})")
            else:
                mapping[test_idx] = 0
                print(f"Warning: '{identifier}' not found, mapping to blank")
                
        elif embedding_mode == "random_augmented":
            # Randomly select an augmented embedding (CAUSES VARIANCE!)
            if augmented_ids:
                selected_id = random.choice(augmented_ids)
                mapping[test_idx] = selected_id
                mapping_info[test_idx] = {
                    'name': identifier,
                    'mode': 'random_augmented',
                    'selected': selected_id,
                    'selected_name': train_identifiers[selected_id],
                    'num_augmented_available': len(augmented_ids)
                }
            elif original_idx is not None:
                mapping[test_idx] = original_idx
                print(f"Warning: No augmented versions for '{identifier}', using original (ID={original_idx})")
            else:
                mapping[test_idx] = 0
                print(f"Warning: '{identifier}' not found, mapping to blank")
        else:
            raise ValueError(f"Unknown embedding_mode: '{embedding_mode}'. "
                           f"Use 'original', 'first_augmented', or 'random_augmented'")
    
    print(f"\nBuilt identifier mapping: {len(mapping)} entries")
    print(f"  Test dataset: {len(test_identifiers)} identifiers")
    print(f"  Train dataset: {len(train_identifiers)} identifiers")
    print(f"  Embedding mode: {embedding_mode}")
    
    if mapping_info:
        print(f"\nSample mappings (mode='{embedding_mode}'):")
        for test_idx in list(mapping_info.keys())[:5]:
            info = mapping_info[test_idx]
            print(f"  Aug-0 ID {test_idx} ({info['name']}) -> Aug-1000 ID {info['selected']}")
            if info['mode'] != 'original':
                print(f"    ({info['num_augmented_available']} augmented versions available)")
                print(f"    Selected: {info['selected_name'][:60]}...")
    
    return mapping


class RemappedPuzzleDataset(IterableDataset):
    """
    A puzzle dataset that remaps puzzle_identifiers from one dataset to another.
    
    This loads test data from one dataset (e.g., aug-0) but remaps the puzzle_identifiers
    to match another dataset (e.g., aug-1000) so they can be used with a model trained
    on the larger dataset.
    """
    
    def __init__(self, config: PuzzleDatasetConfig, split: str, 
                 identifier_mapping: Dict[int, int],
                 override_metadata: PuzzleDatasetMetadata):
        super().__init__()
        self.config = config
        self.split = split
        self.identifier_mapping = identifier_mapping
        
        # Use the override metadata (from aug-1000) for model compatibility
        self.metadata = override_metadata
        
        # Load actual test data metadata for structure info
        self._load_test_metadata()
        
        # Batch size calculation
        assert self.config.global_batch_size % self.config.num_replicas == 0
        self.local_batch_size = self.config.global_batch_size // self.config.num_replicas
        
        self._data = None
    
    def _load_test_metadata(self):
        """Load metadata from the test dataset"""
        dataset_path = self.config.dataset_paths[0]
        with open(os.path.join(dataset_path, self.split, "dataset.json"), "r") as f:
            self._test_metadata = PuzzleDatasetMetadata(**json.load(f))
    
    def _lazy_load_dataset(self):
        if self._data is not None:
            return
        
        field_mmap_modes = {
            "inputs": "r",
            "labels": "r",
            "puzzle_identifiers": None,
            "puzzle_indices": None,
            "group_indices": None
        }
        
        self._data = {}
        for set_name in self._test_metadata.sets:
            for i, dataset_path in enumerate(self.config.dataset_paths):
                if i > 0:
                    set_name_ = set_name + str(i)
                else:
                    set_name_ = set_name
                self._data[set_name_] = {
                    field_name: np.load(
                        os.path.join(dataset_path, self.split, f"{set_name}__{field_name}.npy"),
                        mmap_mode=mmap_mode
                    )
                    for field_name, mmap_mode in field_mmap_modes.items()
                }
    
    def _remap_puzzle_identifiers(self, puzzle_identifiers: np.ndarray) -> np.ndarray:
        """Remap puzzle identifiers from test dataset to train dataset indices"""
        remapped = np.array([
            self.identifier_mapping.get(int(pid), 0)  # Default to blank (0) if not found
            for pid in puzzle_identifiers
        ], dtype=puzzle_identifiers.dtype)
        return remapped
    
    def _collate_batch(self, batch):
        # Convert dtype
        batch = {k: v.astype(np.int32) for k, v in batch.items()}
        
        # REMAP puzzle_identifiers before any other processing
        batch["puzzle_identifiers"] = self._remap_puzzle_identifiers(batch["puzzle_identifiers"])
        
        # Convert ignore label IDs
        if self._test_metadata.ignore_label_id is not None:
            batch["labels"][batch["labels"] == self._test_metadata.ignore_label_id] = IGNORE_LABEL_ID
        
        # Pad
        if batch["puzzle_identifiers"].size < self.local_batch_size:
            pad_size = self.local_batch_size - batch["puzzle_identifiers"].size
            pad_values = {
                "inputs": self._test_metadata.pad_id,
                "labels": IGNORE_LABEL_ID,
                "puzzle_identifiers": self.metadata.blank_identifier_id  # Use override metadata's blank
            }
            batch = {
                k: np.pad(v, ((0, pad_size),) + ((0, 0),) * (v.ndim - 1), constant_values=pad_values[k])
                for k, v in batch.items()
            }
        
        return {k: torch.from_numpy(v) for k, v in batch.items()}
    
    def _iter_test(self):
        for set_i, (set_name, dataset) in enumerate(self._data.items()):
            total_examples = len(dataset["inputs"])
            
            start_index = 0
            while start_index < total_examples:
                end_index = min(total_examples, start_index + self.config.global_batch_size)
                
                local_start = start_index + self.config.rank * self.local_batch_size
                local_end = min(start_index + (self.config.rank + 1) * self.local_batch_size, end_index)
                
                # Get puzzle indices for this batch
                puzzle_indices = []
                puzzle_index = np.searchsorted(dataset["puzzle_indices"], local_start, side="right") - 1
                for i in range(local_start, local_end):
                    while puzzle_index + 1 < len(dataset["puzzle_indices"]) and i >= dataset["puzzle_indices"][puzzle_index + 1]:
                        puzzle_index += 1
                    puzzle_indices.append(puzzle_index)
                
                batch = self._collate_batch({
                    "inputs": dataset["inputs"][local_start:local_end],
                    "labels": dataset["labels"][local_start:local_end],
                    "puzzle_identifiers": dataset["puzzle_identifiers"][puzzle_indices]
                })
                
                yield set_name, batch, end_index - start_index
                
                start_index += self.config.global_batch_size
    
    def __iter__(self):
        worker_info = get_worker_info()
        assert worker_info is None or worker_info.num_workers == 1
        
        self._lazy_load_dataset()
        yield from self._iter_test()


def create_dataloader_for_model(config: EvalConfig, split: str):
    """
    Create dataloader using data_paths (aug-1000) for model metadata.
    This is used to get the correct num_puzzle_identifiers for model initialization.
    """
    from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
    
    dataset = PuzzleDataset(PuzzleDatasetConfig(
        seed=config.seed,
        dataset_paths=config.data_paths,  # Use aug-1000 for model
        rank=0,
        num_replicas=1,
        test_set_mode=True,
        epochs_per_iter=1,
        global_batch_size=config.global_batch_size,
    ), split=split)
    
    return dataset.metadata


def create_remapped_dataloader(config: EvalConfig, model_metadata: PuzzleDatasetMetadata):
    """
    Create dataloader that loads test data from data_paths_test (aug-0)
    but remaps puzzle_identifiers to match data_paths (aug-1000).
    """
    # Build identifier mapping with specified embedding mode
    identifier_mapping = build_identifier_mapping(
        train_data_path=config.data_paths[0],
        test_data_path=config.data_paths_test[0],
        embedding_mode=config.embedding_mode,
        seed=config.seed
    )
    
    # Create remapped dataset
    dataset = RemappedPuzzleDataset(
        config=PuzzleDatasetConfig(
            seed=config.seed,
            dataset_paths=config.data_paths_test,  # Load test data from aug-0
            rank=0,
            num_replicas=1,
            test_set_mode=True,
            epochs_per_iter=1,
            global_batch_size=config.global_batch_size,
        ),
        split="test",
        identifier_mapping=identifier_mapping,
        override_metadata=model_metadata,  # Use aug-1000 metadata for model compatibility
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
        prefetch_factor=8,
        pin_memory=True,
        persistent_workers=True
    )
    
    return dataloader, model_metadata


def create_model(config: EvalConfig, eval_metadata: PuzzleDatasetMetadata):
    """Create model for evaluation"""
    model_cfg = dict(
        **config.arch.__pydantic_extra__,
        batch_size=config.global_batch_size,
        vocab_size=eval_metadata.vocab_size,
        seq_len=eval_metadata.seq_len,
        num_puzzle_identifiers=eval_metadata.num_puzzle_identifiers,
        causal=False
    )
    
    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)
    
    with device:
        model: nn.Module = model_cls(model_cfg)
        print(model)
        model = loss_head_cls(model, **config.arch.loss.__pydantic_extra__)
        if "DISABLE_COMPILE" not in os.environ:
            model = torch.compile(model)
        
        load_checkpoint(model, config)
    
    return model


def init_eval_state(config: EvalConfig, eval_metadata: PuzzleDatasetMetadata):
    """Initialize evaluation state"""
    model = create_model(config, eval_metadata)
    return EvalState(model=model, carry=None)


def load_checkpoint(model: nn.Module, config: EvalConfig):
    """Load model checkpoint"""
    if config.load_checkpoint is not None:
        print(f"Loading checkpoint {config.load_checkpoint}")
        
        state_dict = torch.load(config.load_checkpoint, map_location=device, weights_only=False)
        
        # Check puzzle embedding shape - MUST match exactly
        puzzle_emb_name = "_orig_mod.model.inner.puzzle_emb.weights"
        expected_shape: torch.Size = model.model.puzzle_emb.weights.shape
        if puzzle_emb_name in state_dict:
            puzzle_emb = state_dict[puzzle_emb_name]
            if puzzle_emb.shape != expected_shape:
                raise ValueError(
                    f"Puzzle embedding shape mismatch!\n"
                    f"Checkpoint: {puzzle_emb.shape}, Expected: {expected_shape}\n"
                    f"Make sure data_paths uses the same dataset as the checkpoint was trained on."
                )
        
        model.load_state_dict(state_dict, assign=True)


def create_evaluators(config: EvalConfig, eval_metadata: PuzzleDatasetMetadata) -> List[Any]:
    """Create evaluators for evaluation"""
    data_paths = config.data_paths_test if len(config.data_paths_test) > 0 else config.data_paths
    evaluators = []
    for cfg in config.evaluators:
        for data_path in data_paths:
            cls = load_model_class(cfg.name, "evaluators.")(
                data_path=data_path, eval_metadata=eval_metadata, **cfg.__pydantic_extra__
            )
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
        return_keys.add("preds")
        for evaluator in evaluators:
            evaluator.begin_eval()
            return_keys.update(evaluator.required_outputs)

        set_ids = {k: idx for idx, k in enumerate(eval_metadata.sets)}

        metric_keys = []
        metric_values = None
        processed_batches = 0
        
        # Track per-puzzle-ID statistics
        puzzle_id_stats = {}
        puzzle_id_first_seen = set()
        
        for set_name, batch, global_batch_size in eval_loader:
            processed_batches += 1
            print(f"Processing batch {processed_batches}: {set_name}")
            
            batch = {k: v.to(device) for k, v in batch.items()}
            with device:
                carry = eval_state.model.initial_carry(batch)
            
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
            
            # Update evaluators
            for evaluator in evaluators:
                evaluator.update_batch(batch, preds)
            
            # Track per-puzzle-ID exact accuracy
            if "puzzle_identifiers" in batch:
                puzzle_ids = batch["puzzle_identifiers"].cpu().numpy()
                labels = batch["labels"]
                mask = (labels != -100)
                loss_counts = mask.sum(-1).cpu().numpy()
                is_correct = mask & (preds["preds"] == labels)
                seq_is_correct = (is_correct.sum(-1) == mask.sum(-1)).cpu().numpy()
                halted = carry.halted.cpu().numpy()
                valid = halted & (loss_counts > 0)
                
                blank_id = eval_metadata.blank_identifier_id
                for pid, is_exact_correct, is_valid in zip(puzzle_ids, seq_is_correct, valid):
                    if is_valid and pid != blank_id:
                        pid = int(pid)
                        if config.puzzle_ids is not None and pid not in config.puzzle_ids:
                            continue
                        if pid not in puzzle_id_first_seen:
                            puzzle_id_first_seen.add(pid)
                            if pid not in puzzle_id_stats:
                                puzzle_id_stats[pid] = {"correct": 0, "total": 0}
                            puzzle_id_stats[pid]["total"] += 1
                            if is_exact_correct:
                                puzzle_id_stats[pid]["correct"] += 1

            # Aggregate metrics
            set_id = set_ids[set_name]

            if metric_values is None:
                metric_keys = list(sorted(metrics.keys()))
                metric_values = torch.zeros(
                    (len(set_ids), len(metrics.values())), dtype=torch.float32, device=device
                )

            metric_values[set_id] += torch.stack([metrics[k] for k in metric_keys])
            
            del carry, loss, preds, metrics, batch, all_finish

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

            for set_name, m in reduced_metrics.items():
                count = m.pop("count")
                reduced_metrics[set_name] = {k: v / count for k, v in m.items()}
            
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
                print("Per-Puzzle-ID Exact Accuracy Statistics (Non-Augmented Validation):")
                print("="*80)
                
                sorted_puzzle_ids = sorted(puzzle_id_stats.keys())
                total_puzzles = len(sorted_puzzle_ids)
                total_correct = sum(stats["correct"] for stats in puzzle_id_stats.values())
                
                perfect_puzzles = sum(1 for stats in puzzle_id_stats.values() 
                                     if stats["correct"] == stats["total"] and stats["total"] > 0)
                failed_puzzles = sum(1 for stats in puzzle_id_stats.values() 
                                    if stats["correct"] == 0)
                
                print(f"\nSummary (non-augmented validation - 1st example per puzzle):")
                print(f"  Total unique puzzle IDs evaluated: {total_puzzles}")
                print(f"  Overall accuracy: {total_correct / total_puzzles * 100:.2f}% ({total_correct}/{total_puzzles})")
                print(f"\n  Puzzles SOLVED: {perfect_puzzles} ({perfect_puzzles / total_puzzles * 100:.2f}%)")
                print(f"  Puzzles FAILED: {failed_puzzles} ({failed_puzzles / total_puzzles * 100:.2f}%)")
                
                # Print list of SOLVED puzzle IDs
                solved_ids = [pid for pid, stats in puzzle_id_stats.items() if stats["correct"] == 1]
                failed_ids = [pid for pid, stats in puzzle_id_stats.items() if stats["correct"] == 0]
                
                print(f"\n{'='*80}")
                print(f"✓ SOLVED Puzzle IDs ({len(solved_ids)} puzzles):")
                print(f"{'='*80}")
                if solved_ids:
                    # Print in groups of 10
                    for i in range(0, len(solved_ids), 10):
                        chunk = solved_ids[i:i+10]
                        print(f"  {', '.join(map(str, chunk))}")
                else:
                    print("  (none)")
                
                print(f"\n{'='*80}")
                print(f"✗ FAILED Puzzle IDs ({len(failed_ids)} puzzles):")
                print(f"{'='*80}")
                if failed_ids:
                    # Print in groups of 10
                    for i in range(0, len(failed_ids), 10):
                        chunk = failed_ids[i:i+10]
                        print(f"  {', '.join(map(str, chunk))}")
                else:
                    print("  (none)")
                
                print(f"\n{'='*80}")
                print(f"Detailed breakdown (all puzzle IDs):")
                print(f"{'='*80}")
                print(f"{'Puzzle ID':<12} {'Status':<12}")
                print("-" * 30)
                
                for pid in sorted_puzzle_ids:
                    stats = puzzle_id_stats[pid]
                    status = "✓ SOLVED" if stats["correct"] == 1 else "✗ FAILED"
                    print(f"{pid:<12} {status:<12}")
                
                print("="*80)
        else:
            raise ValueError("No metrics found")

        # Run evaluators
        print(f"\nRunning {len(evaluators)} evaluator(s)...")
        
        for i, evaluator in enumerate(evaluators):
            print(f"Running evaluator {i+1}/{len(evaluators)}: {evaluator.__class__.__name__}")
            
            evaluator_save_path = None
            if config.checkpoint_path is not None:
                evaluator_save_path = os.path.join(
                    config.checkpoint_path,
                    f"evaluator_{evaluator.__class__.__name__}",
                )
                os.makedirs(evaluator_save_path, exist_ok=True)

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
            
            solved_ids = sorted([pid for pid, stats in puzzle_id_stats.items() if stats["correct"] == 1])
            failed_ids = sorted([pid for pid, stats in puzzle_id_stats.items() if stats["correct"] == 0])
            
            puzzle_stats_output = {
                "note": "Non-augmented validation evaluation (puzzle_identifiers remapped from aug-0 to aug-1000)",
                "summary": {
                    "total_puzzles_evaluated": total_puzzles,
                    "puzzles_solved": len(solved_ids),
                    "puzzles_failed": len(failed_ids),
                    "overall_accuracy": total_correct / total_puzzles if total_puzzles > 0 else 0,
                },
                "solved_puzzle_ids": solved_ids,
                "failed_puzzle_ids": failed_ids,
                "per_puzzle": {
                    str(pid): {
                        "status": "solved" if stats["correct"] == 1 else "failed",
                        "correct": stats["correct"],
                    }
                    for pid, stats in sorted(puzzle_id_stats.items())
                }
            }
            
            os.makedirs(config.checkpoint_path, exist_ok=True)
            puzzle_stats_file = os.path.join(config.checkpoint_path, "puzzle_id_stats_noaug.yaml")
            with open(puzzle_stats_file, "wt") as f:
                yaml.dump(puzzle_stats_output, f, default_flow_style=False, sort_keys=False)
            print(f"\nPuzzle ID statistics saved to: {puzzle_stats_file}")
            
            # Also save simple text lists for easy copy-paste
            solved_file = os.path.join(config.checkpoint_path, "solved_puzzle_ids.txt")
            failed_file = os.path.join(config.checkpoint_path, "failed_puzzle_ids.txt")
            
            with open(solved_file, "w") as f:
                f.write(f"# SOLVED Puzzle IDs ({len(solved_ids)} puzzles)\n")
                f.write(f"# Accuracy: {len(solved_ids)/total_puzzles*100:.2f}%\n\n")
                f.write(",".join(map(str, solved_ids)))
            
            with open(failed_file, "w") as f:
                f.write(f"# FAILED Puzzle IDs ({len(failed_ids)} puzzles)\n")
                f.write(f"# Accuracy: 0.00%\n\n")
                f.write(",".join(map(str, failed_ids)))
            
            print(f"Solved puzzle IDs saved to: {solved_file}")
            print(f"Failed puzzle IDs saved to: {failed_file}")

    return reduced_metrics


def save_code_and_config(config: EvalConfig):
    """Save code and configuration files"""
    if config.checkpoint_path is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)

    code_list = [
        get_model_source_path(config.arch.name),
        get_model_source_path(config.arch.loss.name)
    ]
    for code_file in code_list:
        if code_file is not None:
            code_name = os.path.basename(code_file)
            shutil.copy(code_file, os.path.join(config.checkpoint_path, code_name))

    config_file = os.path.join(config.checkpoint_path, "eval_noaug_config.yaml")
    with open(config_file, "wt") as f:
        yaml.dump(config.model_dump(), f)


@hydra.main(config_path="config", config_name="cfg_eval", version_base=None)
def launch(hydra_config: DictConfig):
    """Main evaluation launch function"""
    config = EvalConfig(**hydra_config)

    # Validate that both data_paths and data_paths_test are provided
    if not config.data_paths_test:
        raise ValueError(
            "data_paths_test must be provided for non-augmented evaluation.\n"
            "Example: data_paths_test='[data/arc1concept-aug-0]'"
        )

    print("="*80)
    print("Non-Augmented Validation Evaluation")
    print("="*80)
    print(f"Model dataset (for puzzle embeddings): {config.data_paths}")
    print(f"Test dataset (actual evaluation data): {config.data_paths_test}")
    print(f"Embedding mode: {config.embedding_mode}")
    print("  - 'original': Use ORIGINAL (non-augmented) puzzle embedding from aug-1000")
    print("  - 'first_augmented': Use FIRST augmented puzzle embedding (consistent)")
    print("  - 'random_augmented': Randomly select augmented embedding (causes variance)")
    print("="*80)

    # Naming
    if config.project_name is None:
        config.project_name = f"{os.path.basename(config.data_paths[0]).capitalize()}-ACT-eval-noaug"
    if config.run_name is None:
        config.run_name = f"{config.arch.name.split('@')[-1]} eval-noaug {coolname.generate_slug(2)}"
    if config.checkpoint_path is None:
        config.checkpoint_path = os.path.join("eval_results", config.project_name, config.run_name)

    torch.random.manual_seed(config.seed)

    # Get model metadata from data_paths (aug-1000)
    print("\nLoading model metadata from training dataset...")
    model_metadata = create_dataloader_for_model(config, "train")
    print(f"Model expects {model_metadata.num_puzzle_identifiers} puzzle identifiers")

    # Create remapped dataloader that loads from data_paths_test (aug-0)
    # but uses puzzle_identifiers compatible with the model
    print("\nCreating remapped dataloader for non-augmented test data...")
    eval_loader, eval_metadata = create_remapped_dataloader(config, model_metadata)

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
    print("\nStarting evaluation on non-augmented validation data...")
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

        if config.checkpoint_path is not None:
            metrics_file = os.path.join(config.checkpoint_path, "metrics_noaug.yaml")
            with open(metrics_file, "wt") as f:
                yaml.dump(metrics, f)
            print(f"\nMetrics saved to: {metrics_file}")

    print("Non-augmented evaluation completed!")


if __name__ == "__main__":
    launch()
