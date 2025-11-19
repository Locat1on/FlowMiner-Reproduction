"""
CSTNET 数据集的图构建模块

由于 CSTNET 数据已经是流级别的特征（每个样本一个流），
我们需要使用不同的策略来构建图：
- 每个样本（流）对应一个图
- 图中只有一个节点（该流本身）
- 或者根据标签/特征相似度构建多节点图
"""

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

from .utils import ensure_dir


def build_single_node_graphs(
    nodes_df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    scale_features: bool = True,
) -> List[Data]:
    """
    为 CSTNET 数据集构建单节点图

    每个流对应一个图，图中只有一个节点

    Args:
        nodes_df: 节点 DataFrame
        feature_cols: 要使用的特征列（None = 自动选择数值列）
        scale_features: 是否标准化特征

    Returns:
        图列表
    """
    # 选择特征列
    if feature_cols is None:
        # 排除非特征列
        exclude_cols = {"flow_id", "label", "split"}
        numeric_cols = nodes_df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c not in exclude_cols]

    print(f"[CSTNET Graph] Using {len(feature_cols)} features")

    # 提取特征矩阵
    X = nodes_df[feature_cols].values
    y = nodes_df["label"].values

    # 标准化
    if scale_features:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    # 构建图
    graphs = []
    for i in range(len(nodes_df)):
        # 单节点图
        x = torch.tensor(X[i:i+1], dtype=torch.float)  # shape: (1, num_features)
        y_tensor = torch.tensor([y[i]], dtype=torch.long)

        # 单节点图没有边
        edge_index = torch.empty((2, 0), dtype=torch.long)

        data = Data(
            x=x,
            edge_index=edge_index,
            y=y_tensor,
        )

        # 添加元数据
        if "split" in nodes_df.columns:
            data.split = nodes_df.iloc[i]["split"]
        if "flow_id" in nodes_df.columns:
            data.flow_id = nodes_df.iloc[i]["flow_id"]

        graphs.append(data)

        if (i + 1) % 5000 == 0:
            print(f"  Built {i + 1}/{len(nodes_df)} graphs")

    return graphs


def save_graphs_by_split(
    graphs: List[Data],
    output_dir: Path,
) -> None:
    """
    按 split 分别保存图

    Args:
        graphs: 图列表
        output_dir: 输出目录
    """
    ensure_dir(output_dir)

    # 按 split 分组
    splits = {"train": [], "valid": [], "test": []}

    for graph in graphs:
        if hasattr(graph, "split"):
            split_name = graph.split
            if split_name in splits:
                splits[split_name].append(graph)

    # 保存每个 split
    for split_name, split_graphs in splits.items():
        if len(split_graphs) > 0:
            output_path = output_dir / f"{split_name}_graphs.pt"
            torch.save(split_graphs, output_path)
            print(f"[CSTNET Graph] Saved {len(split_graphs)} {split_name} graphs to {output_path}")

    # 也保存完整的数据集
    all_graphs_path = output_dir / "all_graphs.pt"
    torch.save(graphs, all_graphs_path)
    print(f"[CSTNET Graph] Saved {len(graphs)} total graphs to {all_graphs_path}")


def run_cstnet_graph_construction(
    nodes_csv: Path,
    output_dir: Path,
    scale_features: bool = True,
) -> None:
    """
    运行 CSTNET 图构建流程

    Args:
        nodes_csv: CSTNET 节点 CSV 文件
        output_dir: 输出目录
        scale_features: 是否标准化特征
    """
    print(f"[CSTNET Graph] Loading nodes from {nodes_csv}")
    nodes_df = pd.read_csv(nodes_csv)

    print(f"[CSTNET Graph] Loaded {len(nodes_df)} nodes")
    print(f"[CSTNET Graph] Features: {len(nodes_df.columns)} columns")

    # 检查标签分布
    if "label" in nodes_df.columns:
        label_counts = nodes_df["label"].value_counts()
        print(f"[CSTNET Graph] Label distribution: {len(label_counts)} classes")
        print(f"  Sample counts per class (first 10):")
        for label, count in label_counts.head(10).items():
            print(f"    Class {label}: {count} samples")

    # 构建图
    print("\n[CSTNET Graph] Building graphs...")
    graphs = build_single_node_graphs(
        nodes_df,
        feature_cols=None,
        scale_features=scale_features,
    )

    print(f"\n[CSTNET Graph] Built {len(graphs)} graphs")

    # 保存
    save_graphs_by_split(graphs, output_dir)

    # 打印统计信息
    print(f"\n[CSTNET Graph] Graph statistics:")
    if len(graphs) > 0:
        sample_graph = graphs[0]
        print(f"  Nodes per graph: {sample_graph.x.shape[0]}")
        print(f"  Features per node: {sample_graph.x.shape[1]}")
        print(f"  Edges per graph: {sample_graph.edge_index.shape[1]}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CSTNET 图构建")
    parser.add_argument(
        "--nodes-csv",
        type=Path,
        default=Path("data/processed/cstnet/cstnet_nodes.csv"),
        help="CSTNET 节点 CSV 文件",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/graphs/cstnet"),
        help="输出目录",
    )
    parser.add_argument(
        "--no-scale",
        action="store_true",
        help="不标准化特征",
    )
    return parser


def main(args: List[str] = None) -> None:
    parser = build_argparser()
    parsed = parser.parse_args(args=args)
    run_cstnet_graph_construction(
        nodes_csv=parsed.nodes_csv,
        output_dir=parsed.output_dir,
        scale_features=not parsed.no_scale,
    )


if __name__ == "__main__":
    main()
