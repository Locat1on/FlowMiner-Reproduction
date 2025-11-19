import json
import random
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split


def ensure_dir(path: Path) -> None:
    """Create directory path recursively if it is missing."""
    path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _infer_label(item) -> int:
    """Extract scalar label value from common data containers."""
    if hasattr(item, "y"):
        value = item.y
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError("Expected a scalar target tensor.")
            return int(value.item())
        return int(value)
    raise AttributeError("Unable to infer label; provide labels explicitly.")


def stratified_split(
    items: Sequence,
    splits: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    labels: Optional[Sequence[int]] = None,
) -> Tuple[List, List, List]:
    """Split a list of items into train/val/test with stratified sampling."""
    train_ratio, val_ratio, test_ratio = splits
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Split ratios must sum to 1.")

    if labels is None:
        labels = [_infer_label(item) for item in items]

    indices = np.arange(len(items))
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=1 - train_ratio,
        stratify=labels,
        random_state=seed,
    )

    temp_labels = np.array(labels)[temp_idx]
    val_size = val_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=1 - val_size,
        stratify=temp_labels,
        random_state=seed,
    )

    to_list = lambda idx: [items[i] for i in idx]
    return to_list(train_idx), to_list(val_idx), to_list(test_idx)


def split_by_pcap(
    graphs: Sequence,
    splits: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> Tuple[List, List, List]:
    """Split graphs based on their source PCAP filename to prevent data leakage.
    
    Assumes each graph has a `flow_ids` attribute where the first element
    is formatted as "filename:proto:..."
    """
    # Group graphs by filename
    pcap_groups = {}
    for g in graphs:
        if not hasattr(g, "flow_ids") or not g.flow_ids:
            # Fallback for graphs without flow_ids: treat as unique group
            key = str(id(g))
        else:
            # Extract filename from "filename:proto:..."
            # flow_ids[0] is like "benign_1.pcap:6:1.2.3.4:123-..."
            key = g.flow_ids[0].split(":")[0]
        
        if key not in pcap_groups:
            pcap_groups[key] = []
        pcap_groups[key].append(g)
    
    # Split the filenames (groups) instead of individual graphs
    group_keys = list(pcap_groups.keys())
    # Try to infer a label for the group (majority vote or first graph's label)
    group_labels = []
    for k in group_keys:
        # Use the label of the first graph in the group
        lbl = _infer_label(pcap_groups[k][0])
        group_labels.append(lbl)
        
    # Use stratified split on the groups
    train_keys, val_keys, test_keys = stratified_split(
        group_keys, splits, seed, labels=group_labels
    )
    
    # Flatten back to graph lists
    train_graphs = [g for k in train_keys for g in pcap_groups[k]]
    val_graphs = [g for k in val_keys for g in pcap_groups[k]]
    test_graphs = [g for k in test_keys for g in pcap_groups[k]]
    
    return train_graphs, val_graphs, test_graphs


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def filter_small_classes(
    labels: Sequence[int] | torch.Tensor,
    min_samples: int,
) -> Tuple[set[int], set[int]]:
    """Return (kept_labels, dropped_labels) based on class frequency.

    Any class whose出现次数 < ``min_samples`` 会被视为“样本过小类别”，建议在后续
    预处理或训练中忽略。这是对 ``--min-samples-per-class`` 行为的通用封装，
    方便在节点级 CSV 或图级标签上复用同一过滤逻辑。
    """

    if min_samples <= 1:
        all_labels = set(int(v) for v in labels)
        return all_labels, set()

    if isinstance(labels, torch.Tensor):
        vals = labels.view(-1).cpu().numpy().tolist()
    else:
        vals = [int(v) for v in labels]

    unique, counts = np.unique(vals, return_counts=True)
    kept = set(int(lbl) for lbl, c in zip(unique, counts) if c >= min_samples)
    dropped = set(int(lbl) for lbl, c in zip(unique, counts) if c < min_samples)
    return kept, dropped
