import argparse
import ipaddress
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd
import torch
from torch_geometric.data import Data

from .utils import ensure_dir


def _chunk_single_label(df: pd.DataFrame, max_nodes: int) -> List[pd.DataFrame]:
    """Helper that preserves flow integrity while chunking a single-label dataframe."""

    groups: List[pd.DataFrame] = []
    current_frames: List[pd.DataFrame] = []
    current_count = 0
    for _, group in df.groupby("flow_id"):
        group_size = len(group)
        if current_frames and current_count + group_size > max_nodes:
            groups.append(pd.concat(current_frames, ignore_index=True))
            current_frames = []
            current_count = 0
        current_frames.append(group)
        current_count += group_size
    if current_frames:
        groups.append(pd.concat(current_frames, ignore_index=True))
    return groups


def chunk_by_flow(df: pd.DataFrame, max_nodes: int) -> List[pd.DataFrame]:
    print(f"[Graph] Chunking nodes by flow_id with max_nodes={max_nodes} ...")

    if "label" not in df.columns:
        chunks = _chunk_single_label(df, max_nodes)
        print(f"[Graph] Created {len(chunks)} chunks from {len(df)} nodes (label column missing).")
        return chunks

    chunks: List[pd.DataFrame] = []
    for label_value, label_df in df.groupby("label"):
        label_df = label_df.reset_index(drop=True)
        label_chunks = _chunk_single_label(label_df, max_nodes)
        chunks.extend(label_chunks)
        print(
            f"[Graph] Label={label_value} -> {len(label_chunks)} chunks, nodes={len(label_df)}"
        )

    print(f"[Graph] Created {len(chunks)} chunks from {len(df)} nodes (per-label chunking).")
    return chunks


def _cidr_key(ip: str, ipv4_prefix: int, ipv6_prefix: int) -> Optional[str]:
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return None
    prefix = ipv4_prefix if addr.version == 4 else ipv6_prefix
    network = ipaddress.ip_network(f"{addr}/{prefix}", strict=False)
    return f"{network.network_address}/{prefix}"


def build_edges(nodes: pd.DataFrame, max_neighbors_per_group: int = 4) -> Set[Tuple[int, int]]:
    """构建更稀疏的边集，贴合论文的"链式/有限邻域"思想。

    设计要点：
    - 同一 flow 内：按时间排序做链式连接（相邻节点相连），避免 fully connect。
    - IP / 子网 / content group：只给每个节点连最多 `max_neighbors_per_group` 个邻居，避免 O(k^2)。
    """

    print(f"[Graph] Building edges for {len(nodes)} nodes ...")
    edges: Set[Tuple[int, int]] = set()

    # 1) flow 内链式连接（按时间）
    if "timestamp" in nodes.columns:
        # 假设存在一个时间戳列；如果没有，退化为原始顺序
        for flow_id, group in nodes.sort_values("timestamp").groupby("flow_id"):
            idxs = group.index.to_list()
            for i in range(len(idxs) - 1):
                u, v = idxs[i], idxs[i + 1]
                edges.add((u, v))
                edges.add((v, u))
    else:
        for flow_id, group in nodes.groupby("flow_id"):
            idxs = group.index.to_list()
            for i in range(len(idxs) - 1):
                u, v = idxs[i], idxs[i + 1]
                edges.add((u, v))
                edges.add((v, u))

    # 2) 其他语义 group：为每个节点挑有限个邻居做链式/局部连接
    ip_groups: Dict[str, List[int]] = defaultdict(list)
    subnet_groups: Dict[str, List[int]] = defaultdict(list)
    content_groups: Dict[str, List[int]] = defaultdict(list)

    for idx, row in nodes.iterrows():
        for ip_col in ("src_ip", "dst_ip"):
            ip_val = row.get(ip_col)
            if isinstance(ip_val, str):
                ip_groups[ip_val].append(idx)
                cidr = _cidr_key(ip_val, ipv4_prefix=24, ipv6_prefix=64)
                if cidr:
                    subnet_groups[cidr].append(idx)
        content_key = row.get("content_signature")
        if isinstance(content_key, str):
            content_groups[content_key].append(idx)

    def connect_chain_limited(group: Sequence[int]) -> None:
        # 对 group 做链式连接，并限制每个节点邻居不超过 max_neighbors_per_group
        if not group:
            return
        ordered = group
        if "timestamp" in nodes.columns:
            ordered = sorted(group, key=lambda idx: nodes.at[idx, "timestamp"])
        if max_neighbors_per_group is not None:
            ordered = ordered[: max_neighbors_per_group + 1]
        for i in range(len(ordered) - 1):
            u, v = ordered[i], ordered[i + 1]
            edges.add((u, v))
            edges.add((v, u))

    for group in ip_groups.values():
        connect_chain_limited(group)
    for group in subnet_groups.values():
        connect_chain_limited(group)
    for group in content_groups.values():
        connect_chain_limited(group)

    print(f"[Graph] Total directed edges constructed: {len(edges)}")
    return edges


def dataframe_to_graph(nodes: pd.DataFrame, feature_cols: Sequence[str]) -> Optional[Data]:
    if nodes.empty:
        return None
    print(
        f"[Graph] Converting dataframe to graph: nodes={len(nodes)}, features={len(feature_cols)}"
    )
    edges = build_edges(nodes)
    if not edges:
        return None
    edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
    x = torch.tensor(nodes[feature_cols].values, dtype=torch.float32)
    labels = nodes.get("label")
    label_value = int(labels.mode().iloc[0]) if labels is not None else 0
    print(f"[Graph] Graph edge_index shape: {edge_index.shape}")
    data = Data(
        x=x,
        edge_index=edge_index,
        y=torch.tensor([label_value], dtype=torch.long),
        flow_ids=list(nodes["flow_id"]),
    )
    return data


def build_graphs(
    nodes_csv: Path,
    output_dir: Path,
    max_nodes_per_graph: int = 2048,
    feature_columns: Optional[Sequence[str]] = None,
    disable_byte_features: bool = False,
    disable_cross_features: bool = False,
) -> List[Data]:
    print(f"[Graph] Loading nodes from {nodes_csv} ...")
    ensure_dir(output_dir)
    nodes = pd.read_csv(nodes_csv)
    print(f"[Graph] Loaded {len(nodes)} rows with {len(nodes.columns)} columns.")
    if feature_columns is None:
        numeric_cols = nodes.select_dtypes(include=["number"]).columns.tolist()
        reserved = {"label"}
        base_feature_candidates = [
            "packet_count",
            "byte_count",
            "duration",
            "pkt_len_mean",
            "pkt_len_std",
            "pkt_len_min",
            "pkt_len_max",
            "pkt_len_p25",
            "pkt_len_p50",
            "pkt_len_p75",
            "pkt_len_p90",
            "pkt_len_p95",
            "pkt_len_skew",
            "pkt_len_kurtosis",
            "iat_mean",
            "iat_std",
            "iat_min",
            "iat_max",
            "iat_p25",
            "iat_p50",
            "iat_p75",
            "iat_p90",
            "iat_p95",
            "iat_skew",
            "iat_kurtosis",
            "payload_len",
            "payload_printable_ratio",
            "payload_byte_entropy",
            "payload_popcount_ratio",
            "byte_bucket_0_31",
            "byte_bucket_32_63",
            "byte_bucket_64_95",
            "byte_bucket_96_127",
            "byte_bucket_128_159",
            "byte_bucket_160_191",
            "byte_bucket_192_223",
            "byte_bucket_224_255",
            "byte_value_mean",
            "byte_value_std",
            "byte_value_min",
            "byte_value_max",
            "byte_value_skew",
            "byte_value_kurtosis",
            "first_byte_1",
            "first_byte_2",
            "first_byte_3",
            "first_byte_4",
            "first_byte_5",
            "first_byte_6",
            "byte_rate",
            "packet_rate",
        ]
        # 只保留 numeric_cols 中真实存在的基础特征，并排除 label 等保留列
        base_features = [c for c in base_feature_candidates if c in numeric_cols and c not in reserved]

        if disable_byte_features:
            # 移除所有显式的字节级特征：payload 统计、byte_bucket 分布和首字节值。
            byte_prefixes = ("payload_", "byte_bucket_", "first_byte_")
            base_features = [
                c for c in base_features if not any(c.startswith(prefix) for prefix in byte_prefixes)
            ]

        # 同时允许少量 crossing 特征：按名字规则筛选少数 *_plus_* / *_times_* 列
        if disable_cross_features:
            cross_features: List[str] = []
        else:
            cross_features = [
                c
                for c in numeric_cols
                if ("_plus_" in c or "_times_" in c) and c not in reserved
            ][:10]  # 最多取 10 个 crossing 特征，贴近论文设定

        feature_columns = base_features + cross_features
    if not feature_columns:
        raise ValueError("No numeric features available to build graph representations.")
    print(f"[Graph] Using {len(feature_columns)} numeric feature columns.")

    graphs: List[Data] = []
    for chunk_id, chunk in enumerate(chunk_by_flow(nodes, max_nodes_per_graph)):
        print(
            f"[Graph] Processing chunk {chunk_id} with {len(chunk)} nodes (max_nodes_per_graph={max_nodes_per_graph}) ..."
        )
        graph = dataframe_to_graph(chunk.reset_index(drop=True), feature_columns)
        if graph is None:
            print(f"[Graph] Chunk {chunk_id} produced an empty graph, skipping.")
            continue
        graphs.append(graph)
        out_path = output_dir / f"graph_{chunk_id:04}.pt"
        torch.save(graph, out_path)
        print(f"[Graph] Saved subgraph {chunk_id} to {out_path}.")

    dataset_path = output_dir / "dataset.pt"
    torch.save(graphs, dataset_path)
    print(f"[Graph] Saved dataset with {len(graphs)} graphs to {dataset_path}.")
    return graphs


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Construct interaction graphs for FlowMiner.")
    parser.add_argument("--nodes-csv", type=Path, default=Path("data/processed/nodes_enhanced.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/graphs"))
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=500,
        help="Maximum nodes per subgraph (paper default is 500).",
    )
    parser.add_argument("--features", nargs="*", default=None, help="Optional list of feature columns.")
    parser.add_argument(
        "--disable-byte-features",
        action="store_true",
        help="Drop all explicit byte-level features (BF ablation).",
    )
    parser.add_argument(
        "--disable-cross-features",
        action="store_true",
        help="Drop all crossing features such as *_plus_* and *_times_* (CF ablation).",
    )
    return parser


def main(args: Optional[List[str]] = None) -> None:
    parser = build_argparser()
    parsed = parser.parse_args(args=args)
    print("[Graph] Starting graph construction ...")
    print(f"[Graph] nodes_csv = {parsed.nodes_csv}")
    print(f"[Graph] output_dir = {parsed.output_dir}")
    print(f"[Graph] max_nodes_per_graph = {parsed.max_nodes}")
    if parsed.features:
        print(f"[Graph] Using user-specified features: {len(parsed.features)} columns.")
    build_graphs(
        nodes_csv=parsed.nodes_csv,
        output_dir=parsed.output_dir,
        max_nodes_per_graph=parsed.max_nodes,
        feature_columns=parsed.features,
        disable_byte_features=parsed.disable_byte_features,
        disable_cross_features=parsed.disable_cross_features,
    )
    print("[Graph] Graph construction finished.")


if __name__ == "__main__":
    main()
