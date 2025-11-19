#!/usr/bin/env python
"""Utility script to reproduce FlowMiner experiments on two datasets in one shot.

Usage examples
------        "vmess": DatasetConfig(
            name="vmess",
            raw_dir=ROOT / "data" / "raw" / "vmess" / "vmess",
            processed_prefix="vmess",
            graph_dir=ROOT / "data" / "graphs" / "vmess",
            results_dir=ROOT / "results" / "vmess",
            label_map=ROOT / "configs" / "label_maps" / "lfett_vmess.json",
            min_samples_per_class=1,
            epochs=50,
        ),
        "cstnet": DatasetConfig(
            name="cstnet",
            raw_dir=ROOT / "data" / "raw" / "CSTNET-TLS_1.3" / "flow_dataset" / "flow_500",
            processed_prefix="cstnet",
            graph_dir=ROOT / "data" / "graphs" / "cstnet",
            results_dir=ROOT / "results" / "cstnet",
            label_map=None,  # 120 类，使用原始标签
            min_samples_per_class=1,
            epochs=100,  # 更多类别，可能需要更多训练轮次
            hidden_dim=16,  # 稍大的隐藏维度应对 120 分类
            runs=3,  # 多次运行取平均
        ),
    }ull pipeline (feature extraction → preprocessing → graph construction → training)
for both datasets defined below:

    python scripts/run_dual_experiments.py

Only run on VPN dataset and skip training (useful when just refreshing graphs):

    python scripts/run_dual_experiments.py --datasets vpn --skip-train

See ``--help`` for more options.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, List, Optional, Set

import torch


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    raw_dir: Path
    processed_prefix: str
    graph_dir: Path
    results_dir: Path
    default_label: int = 0
    label_map: Optional[Path] = None
    min_samples_per_class: int = 10
    max_nodes: int = 500
    preset: Optional[str] = None
    epochs: int = 50
    # Model hyper-parameters (paper-like lightweight defaults)
    # hidden_dim/gat_heads tuned so total params ≈ 1.4k
    hidden_dim: int = 8
    sage_layers: int = 2
    gat_layers: int = 1
    gat_heads: int = 2
    dropout: float = 0.2
    runs: int = 1

    @property
    def flows_csv(self) -> Path:
        return ROOT / "data" / "processed" / f"{self.processed_prefix}_flows.csv"

    @property
    def nodes_csv(self) -> Path:
        return ROOT / "data" / "processed" / f"{self.processed_prefix}_nodes.csv"

    @property
    def nodes_enhanced_csv(self) -> Path:
        return ROOT / "data" / "processed" / f"{self.processed_prefix}_nodes_enhanced.csv"

    @property
    def nodes_preprocessed_csv(self) -> Path:
        return ROOT / "data" / "processed" / f"{self.processed_prefix}_nodes_preprocessed.csv"


def build_default_configs() -> dict[str, DatasetConfig]:
    return {
        "vpn": DatasetConfig(
            name="vpn",
            raw_dir=ROOT / "data" / "raw" / "VPN-PCAPS-small",
            processed_prefix="vpn",
            graph_dir=ROOT / "data" / "graphs" / "vpn",
            results_dir=ROOT / "results" / "vpn",
            label_map=ROOT / "configs" / "label_maps" / "vpn_iscx.json",
            min_samples_per_class=1,
            epochs=50,
        ),
        "vpn_full": DatasetConfig(
            name="vpn_full",
            raw_dir=ROOT / "data" / "raw" / "VPN-PCAPS",
            processed_prefix="vpn_full",
            graph_dir=ROOT / "data" / "graphs" / "vpn_full",
            results_dir=ROOT / "results" / "vpn_full",
            label_map=ROOT / "configs" / "label_maps" / "vpn_iscx.json",
            min_samples_per_class=1,
            epochs=50,
        ),
        "nonvpn": DatasetConfig(
            name="nonvpn",
            raw_dir=ROOT / "data" / "raw" / "NonVPN-PCAPs-01",
            processed_prefix="nonvpn",
            graph_dir=ROOT / "data" / "graphs" / "nonvpn",
            results_dir=ROOT / "results" / "nonvpn",
            label_map=ROOT / "configs" / "label_maps" / "nonvpn_iscx.json",
            min_samples_per_class=1,
            epochs=50,
        ),
        "ustc": DatasetConfig(
            name="ustc",
            raw_dir=ROOT / "data" / "raw" / "USTC-TFC2016-master",
            processed_prefix="ustc",
            graph_dir=ROOT / "data" / "graphs" / "ustc",
            results_dir=ROOT / "results" / "ustc",
            # 如果需要把 20 类流映射到恶意/良性，可以在此放自定义 label_map。
            label_map=None,
            epochs=50,
        ),
        "ssr": DatasetConfig(
            name="ssr",
            raw_dir=ROOT / "data" / "raw" / "ssr" / "ssr",
            processed_prefix="ssr",
            graph_dir=ROOT / "data" / "graphs" / "ssr",
            results_dir=ROOT / "results" / "ssr",
            label_map=ROOT / "configs" / "label_maps" / "lfett_ssr.json",
            min_samples_per_class=0,
            epochs=100,
            hidden_dim=32,
        ),
        "vmess": DatasetConfig(
            name="vmess",
            raw_dir=ROOT / "data" / "raw" / "vmess" / "vmess",
            processed_prefix="vmess",
            graph_dir=ROOT / "data" / "graphs" / "vmess",
            results_dir=ROOT / "results" / "vmess",
            label_map=ROOT / "configs" / "label_maps" / "lfett_vmess.json",
            min_samples_per_class=0,
            epochs=50,
            hidden_dim=64,
        ),
        "cstnet": DatasetConfig(
            name="cstnet",
            raw_dir=ROOT / "data" / "raw" / "CSTNET-TLS_1.3" / "flow_dataset" / "flow_500",
            processed_prefix="cstnet",
            graph_dir=ROOT / "data" / "graphs" / "cstnet",
            results_dir=ROOT / "results" / "cstnet",
            label_map=None,  # 120 类，使用原始标签 0-119
            min_samples_per_class=1,
            epochs=100,  # 更多类别，需要更多训练轮次
            hidden_dim=8,  # 稍大的隐藏维度应对 120 分类
            sage_layers=2,
            gat_layers=1,
            gat_heads=2,
            dropout=0.2,
        ),
    }



def run_cmd(cmd: List[str]) -> None:
    printable = " ".join(shlex.quote(str(item)) for item in cmd)
    print(f"\n>>> {printable}")
    subprocess.run(cmd, cwd=ROOT, check=True)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def cstnet_pipeline(
    cfg: DatasetConfig,
    max_samples: Optional[int] = None,
    top_k: int = 10,
    bin_count: int = 5,
    min_samples_override: Optional[int] = None,
    *,
    disable_byte_features: bool,
    disable_cross_features: bool,
    no_feature_engineering: bool = False,
) -> None:
    """CSTNET 数据集专用处理流程"""
    print(f"\n[CSTNET] Running CSTNET-specific pipeline for {cfg.name}")

    # Step 1: 加载 .npy 文件并转换为节点 CSV
    nodes_dir = cfg.nodes_csv.parent
    ensure_parent(cfg.nodes_csv)

    cmd = [
        PYTHON,
        "-m",
        "src.cstnet_loader",
        "--data-dir",
        str(cfg.raw_dir),
        "--output-dir",
        str(nodes_dir),
    ]
    if max_samples:
        cmd.extend(["--max-samples", str(max_samples)])
    run_cmd(cmd)

    # CSTNET 生成的是 cstnet_nodes.csv，我们需要将其复制/重命名为标准的 nodes.csv
    cstnet_nodes_path = nodes_dir / "cstnet_nodes.csv"
    if cstnet_nodes_path.exists():
        import shutil
        # 备份到标准位置
        shutil.copy(cstnet_nodes_path, cfg.nodes_csv)
        print(f"[CSTNET] Copied {cstnet_nodes_path} to {cfg.nodes_csv}")

    # Step 2: 特征工程 - enhance (添加交叉特征和分箱)
    ensure_parent(cfg.nodes_enhanced_csv)
    
    if no_feature_engineering:
        print(f"\n[CSTNET] Skipping feature engineering (copying raw nodes)...")
        import shutil
        shutil.copy(cfg.nodes_csv, cfg.nodes_enhanced_csv)
        print(f"[CSTNET] Raw nodes copied to {cfg.nodes_enhanced_csv}")
    else:
        print(f"\n[CSTNET] Enhancing features...")
        # 使用 Python 直接调用 enhance 函数，而不是通过命令行
        # 因为 CSTNET 已经有了节点 CSV，只需要做 enhance
        import sys
        import pandas as pd

        # 确保项目根目录在 sys.path 中，以便作为包导入 src.feature_extraction
        root_str = str(ROOT)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
        from src.feature_extraction import enhance_with_cross_features

        nodes_df = pd.read_csv(cfg.nodes_csv)
        print(f"[CSTNET] Loaded {len(nodes_df)} nodes from {cfg.nodes_csv}")

        enhanced_df = enhance_with_cross_features(nodes_df, top_k=top_k, bin_count=bin_count)
        enhanced_df.to_csv(cfg.nodes_enhanced_csv, index=False)
        print(f"[CSTNET] Enhanced features saved to {cfg.nodes_enhanced_csv}")
        print(f"[CSTNET] Features: {len(nodes_df.columns)} -> {len(enhanced_df.columns)}")

    # Step 3: 预处理 (标准化、过滤等)
    print(f"\n[CSTNET] Preprocessing nodes...")
    ensure_parent(cfg.nodes_preprocessed_csv)
    min_samples = (
        cfg.min_samples_per_class if min_samples_override is None else min_samples_override
    )
    cmd = [
        PYTHON,
        "-m",
        "src.preprocess_nodes",
        "--input-csv",
        str(cfg.nodes_enhanced_csv),
        "--output-csv",
        str(cfg.nodes_preprocessed_csv),
        "--min-samples-per-class",
        str(min_samples),
    ]
    run_cmd(cmd)

    # Step 4: 构建图（与其他任务一致，使用标准 graph_construction 流程）
    print(f"\n[CSTNET] Building graphs (standard FlowMiner pipeline)...")
    build_graphs(
        cfg,
        disable_byte_features=disable_byte_features,
        disable_cross_features=disable_cross_features,
    )

    print(f"[CSTNET] CSTNET pipeline completed for {cfg.name}")



def feature_extraction(cfg: DatasetConfig, top_k: int, bin_count: int, no_enhance: bool = False) -> None:
    ensure_parent(cfg.flows_csv)
    ensure_parent(cfg.nodes_csv)
    ensure_parent(cfg.nodes_enhanced_csv)
    cmd = [
        PYTHON,
        "-m",
        "src.feature_extraction",
        "--raw-dir",
        str(cfg.raw_dir),
        "--flows-out",
        str(cfg.flows_csv),
        "--nodes-out",
        str(cfg.nodes_csv),
        "--nodes-enhanced-out",
        str(cfg.nodes_enhanced_csv),
        "--default-label",
        str(cfg.default_label),
        "--top-k",
        str(top_k),
        "--bin-count",
        str(bin_count),
    ]
    if cfg.label_map:
        cmd.extend(["--label-map", str(cfg.label_map)])
    if no_enhance:
        cmd.append("--no-enhance")
    run_cmd(cmd)


def preprocess_nodes(cfg: DatasetConfig, min_samples_override: Optional[int]) -> None:
    ensure_parent(cfg.nodes_preprocessed_csv)
    min_samples = (
        cfg.min_samples_per_class if min_samples_override is None else min_samples_override
    )
    cmd = [
        PYTHON,
        "-m",
        "src.preprocess_nodes",
        "--input-csv",
        str(cfg.nodes_enhanced_csv),
        "--output-csv",
        str(cfg.nodes_preprocessed_csv),
        "--min-samples-per-class",
        str(min_samples),
    ]
    run_cmd(cmd)


def build_graphs(
    cfg: DatasetConfig,
    *,
    disable_byte_features: bool,
    disable_cross_features: bool,
) -> None:
    cfg.graph_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON,
        "-m",
        "src.graph_construction",
        "--nodes-csv",
        str(cfg.nodes_preprocessed_csv),
        "--output-dir",
        str(cfg.graph_dir),
        "--max-nodes",
        str(cfg.max_nodes),
    ]
    if disable_byte_features:
        cmd.append("--disable-byte-features")
    if disable_cross_features:
        cmd.append("--disable-cross-features")
    run_cmd(cmd)


def train(
    cfg: DatasetConfig,
    graph_limit: Optional[int],
    preset: Optional[str],
    train_epochs: Optional[int],
    disable_idp: bool,
) -> None:
    cfg.results_dir.mkdir(parents=True, exist_ok=True)

    label_set = collect_graph_labels(cfg.graph_dir)
    if len(label_set) < 2:
        print(
            f"[run_dual] Skip training for dataset '{cfg.name}' because only "
            f"{sorted(label_set) if label_set else 'no'} class labels are present. "
            "Check preprocessing/filtering or provide more data."
        )
        return

    cmd = [
        PYTHON,
        "-m",
        "src.train",
        "--graph-dir",
        str(cfg.graph_dir),
        "--output-dir",
        str(cfg.results_dir),
    ]
    # 不再默认使用 src.train 的 preset，而是显式传入与论文贴合的结构超参；
    # 仅当用户显式指定 --preset 时，才把控制权交给 src.train.apply_preset。
    effective_preset = preset or cfg.preset
    if effective_preset:
        cmd.extend(["--preset", effective_preset])
    else:
        epochs = train_epochs or cfg.epochs
        if epochs:
            cmd.extend(["--epochs", str(epochs)])
        cmd.extend(
            [
                "--hidden-dim",
                str(cfg.hidden_dim),
                "--sage-layers",
                str(cfg.sage_layers),
                "--gat-layers",
                str(cfg.gat_layers),
                "--gat-heads",
                str(cfg.gat_heads),
                "--dropout",
                str(cfg.dropout),
                "--runs",
                str(cfg.runs),
            ]
        )
    if graph_limit is not None:
        cmd.extend(["--graph-limit", str(graph_limit)])
    if disable_idp:
        cmd.append("--disable-idp")
    run_cmd(cmd)


def run_pipeline(
    cfg: DatasetConfig,
    *,
    top_k: int,
    bin_count: int,
    min_samples_override: Optional[int],
    skip_train: bool,
    only_train: bool,
    graph_limit: Optional[int],
    preset: Optional[str],
    train_epochs: Optional[int],
    disable_byte_features: bool,
    disable_cross_features: bool,
    disable_idp: bool,
    no_feature_engineering: bool = False,
) -> None:
    print(f"\n=== Running FlowMiner pipeline for dataset: {cfg.name} ===")

    # 检查是否是 CSTNET 数据集
    is_cstnet = cfg.name.lower().startswith("cstnet")

    if only_train:
        # 只做训练：假定 graphs 目录已经存在并包含构建好的图
        print("[run_dual] Only training is requested; skip feature extraction, preprocessing and graph construction.")
        train(
            cfg,
            graph_limit=graph_limit,
            preset=preset,
            train_epochs=train_epochs,
            disable_idp=disable_idp,
        )
        print(f"=== Completed dataset (train only): {cfg.name} ===\n")
        return

    # CSTNET 使用专用流程
    if is_cstnet:
        print("[run_dual] Detected CSTNET dataset, using specialized pipeline")
        cstnet_pipeline(
            cfg,
            max_samples=graph_limit,
            top_k=top_k,
            bin_count=bin_count,
            min_samples_override=min_samples_override,
            disable_byte_features=disable_byte_features,
            disable_cross_features=disable_cross_features,
            no_feature_engineering=no_feature_engineering,
        )
    else:
        # 标准 PCAP 流程
        feature_extraction(cfg, top_k=top_k, bin_count=bin_count, no_enhance=no_feature_engineering)
        preprocess_nodes(cfg, min_samples_override=min_samples_override)
        build_graphs(
            cfg,
            disable_byte_features=disable_byte_features,
            disable_cross_features=disable_cross_features,
        )

    if not skip_train:
        train(
            cfg,
            graph_limit=graph_limit,
            preset=preset,
            train_epochs=train_epochs,
            disable_idp=disable_idp,
        )
    print(f"=== Completed dataset: {cfg.name} ===\n")


def parse_args(all_datasets: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FlowMiner experiments on multiple datasets.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(all_datasets),
        help="Datasets to run (default: all). Choices: %(choices)s.",
        choices=list(all_datasets),
    )
    parser.add_argument("--top-k", type=int, default=10, help="Top-K variance features for crosses.")
    parser.add_argument("--bin-count", type=int, default=5, help="Quantile bins for discretization.")
    parser.add_argument(
        "--min-samples",
        type=int,
        default=None,
        help="Override minimum samples per class during preprocessing.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Stop after graph construction (skip the training stage).",
    )
    parser.add_argument(
        "--only-train",
        action="store_true",
        help="Only run training using existing graphs (skip feature extraction, preprocessing, and graph construction).",
    )
    parser.add_argument(
        "--graph-limit",
        type=int,
        default=None,
        help="Optional --graph-limit forwarded to src.train (useful for quick smoke tests).",
    )
    parser.add_argument(
        "--preset",
        choices=["paper"],
        default=None,
        help="Optional preset for training hyper-parameters (default: no preset).",
    )
    parser.add_argument(
        "--train-epochs",
        type=int,
        default=200,
        help="Epochs passed to src.train when no preset is used (default: 200).",
    )
    parser.add_argument(
        "--disable-byte-features",
        action="store_true",
        help="Disable byte-level features (BF ablation, forwarded to src.graph_construction).",
    )
    parser.add_argument(
        "--disable-cross-features",
        action="store_true",
        help="Disable crossing features (CF ablation, forwarded to src.graph_construction).",
    )
    parser.add_argument(
        "--disable-idp",
        action="store_true",
        help="Disable Integrated Decision Pooling (IDP ablation, forwarded to src.train).",
    )
    parser.add_argument(
        "--no-feature-engineering",
        action="store_true",
        help="Skip feature engineering (binning, cross features) and use raw features only.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default=None,
        help=(
            "Optional suffix appended to processed prefix / graph / results directories, "
            "useful for organizing ablation or variant runs (e.g., 'no_bf')."
        ),
    )
    return parser.parse_args()


def collect_graph_labels(graph_dir: Path) -> Set[int]:
    labels: Set[int] = set()

    # 检查 CSTNET 格式的图文件 (train_graphs.pt, test_graphs.pt, valid_graphs.pt)
    for split_name in ["train_graphs.pt", "test_graphs.pt", "valid_graphs.pt", "all_graphs.pt"]:
        split_path = graph_dir / split_name
        if split_path.exists():
            print(f"[collect_labels] Loading labels from {split_name}")
            graphs = torch.load(split_path, weights_only=False)
            for data in graphs:
                if hasattr(data, "y"):
                    labels.update(int(v) for v in torch.as_tensor(data.y).view(-1).tolist())
            if len(labels) >= 2:
                return labels

    # 检查标准格式的图文件 (graph_*.pt)
    graph_paths = sorted(graph_dir.glob("graph_*.pt"))
    if not graph_paths:
        dataset_path = graph_dir / "dataset.pt"
        if dataset_path.exists():
            datasets = torch.load(dataset_path, weights_only=False)
            for data in datasets:
                if hasattr(data, "y"):
                    labels.update(int(v) for v in torch.as_tensor(data.y).view(-1).tolist())
            return labels

    for path in graph_paths:
        data = torch.load(path, weights_only=False)
        if hasattr(data, "y"):
            labels.update(int(v) for v in torch.as_tensor(data.y).view(-1).tolist())
        if len(labels) >= 2:
            break
    return labels


def main() -> None:
    configs = build_default_configs()
    args = parse_args(configs.keys())
    for ds_name in args.datasets:
        base_cfg = configs[ds_name]
        cfg = base_cfg
        # 如果指定了 suffix，为该次实验派生一个新的配置，避免覆盖基线结果。
        if args.suffix:
            suffix = args.suffix
            new_name = f"{base_cfg.name}_{suffix}"
            new_prefix = f"{base_cfg.processed_prefix}_{suffix}"
            new_graph_dir = base_cfg.graph_dir.parent / new_name
            new_results_dir = base_cfg.results_dir.parent / new_name
            cfg = DatasetConfig(
                name=new_name,
                raw_dir=base_cfg.raw_dir,
                processed_prefix=new_prefix,
                graph_dir=new_graph_dir,
                results_dir=new_results_dir,
                default_label=base_cfg.default_label,
                label_map=base_cfg.label_map,
                min_samples_per_class=base_cfg.min_samples_per_class,
                max_nodes=base_cfg.max_nodes,
                preset=base_cfg.preset,
                epochs=base_cfg.epochs,
                hidden_dim=base_cfg.hidden_dim,
                sage_layers=base_cfg.sage_layers,
                gat_layers=base_cfg.gat_layers,
                gat_heads=base_cfg.gat_heads,
                dropout=base_cfg.dropout,
                runs=base_cfg.runs,
            )
        run_pipeline(
            cfg,
            top_k=args.top_k,
            bin_count=args.bin_count,
            min_samples_override=args.min_samples,
            skip_train=args.skip_train,
            only_train=args.only_train,
            graph_limit=args.graph_limit,
            preset=args.preset,
            train_epochs=args.train_epochs,
            disable_byte_features=args.disable_byte_features,
            disable_cross_features=args.disable_cross_features,
            disable_idp=args.disable_idp,
            no_feature_engineering=args.no_feature_engineering,
        )


if __name__ == "__main__":
    main()
