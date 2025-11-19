"""
CSTNET-TLS 1.3 数据集加载器

这个数据集已经是预处理好的特征数组，不需要 PCAP 解析。
数据格式：.npy 文件，包含 direction, length, time, message_type, datagram 等特征。
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .utils import ensure_dir


def load_cstnet_split(
    data_dir: Path,
    split: str = "train",
    max_samples: int = None,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    从 CSTNET-TLS 1.3 数据集加载一个 split (train/test/valid)

    Args:
        data_dir: 数据目录路径（例如 flow_500 或 packet_5000）
        split: 'train', 'test', 或 'valid'
        max_samples: 可选，限制加载的最大样本数（用于快速测试）

    Returns:
        features: 字典 {feature_name: array}
        labels: 标签数组
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory {data_dir} does not exist")

    # 加载标签
    y_path = data_dir / f"y_{split}.npy"
    if not y_path.exists():
        raise FileNotFoundError(f"Label file {y_path} does not exist")

    labels = np.load(y_path)

    if max_samples:
        labels = labels[:max_samples]

    n_samples = len(labels)

    # 加载所有特征
    features = {}
    feature_names = ["direction", "len", "time", "message_type", "datagram"]

    for feat_name in feature_names:
        feat_path = data_dir / f"x_{feat_name}_{split}.npy"
        if feat_path.exists():
            try:
                # datagram 和 message_type 可能需要 allow_pickle
                if feat_name in ["datagram", "message_type"]:
                    feat_array = np.load(feat_path, allow_pickle=True)
                else:
                    feat_array = np.load(feat_path)

                if max_samples:
                    feat_array = feat_array[:max_samples]

                features[feat_name] = feat_array
                print(f"  Loaded {feat_name}: shape={feat_array.shape}, dtype={feat_array.dtype}")
            except Exception as e:
                print(f"  Warning: Failed to load {feat_name}: {e}")
        else:
            print(f"  Warning: Feature file {feat_path} not found, skipping")

    return features, labels


def cstnet_to_nodes_dataframe(
    features: Dict[str, np.ndarray],
    labels: np.ndarray,
    flow_id_prefix: str = "flow",
) -> pd.DataFrame:
    """
    将 CSTNET 特征转换为类似于 PCAP 流程的节点 DataFrame

    每个样本（流）生成一个节点，特征从原始数组中提取统计量

    Args:
        features: 特征字典
        labels: 标签数组
        flow_id_prefix: 流 ID 前缀

    Returns:
        nodes DataFrame，包含标准化的特征列
    """
    n_samples = len(labels)
    rows = []

    for i in range(n_samples):
        row = {
            "flow_id": f"{flow_id_prefix}_{i}",
            "label": int(labels[i]),
        }

        # 从各个特征中提取统计量
        # Direction: 0/1 序列，长度 5000
        if "direction" in features:
            direction = features["direction"][i]
            if len(direction) > 0:
                row["direction_mean"] = float(np.mean(direction))
                row["direction_sum"] = float(np.sum(direction))  # 某个方向的包数
                row["direction_changes"] = float(np.sum(np.diff(direction) != 0))  # 方向切换次数

        # Length: 包长度序列，长度 1000
        if "len" in features:
            lengths = features["len"][i]
            # 过滤掉 padding 的 0 值（假设真实长度不会是 0）
            valid_lengths = lengths[lengths > 0]
            if len(valid_lengths) > 0:
                row["pkt_len_mean"] = float(np.mean(valid_lengths))
                row["pkt_len_std"] = float(np.std(valid_lengths))
                row["pkt_len_min"] = float(np.min(valid_lengths))
                row["pkt_len_max"] = float(np.max(valid_lengths))
                row["pkt_len_p50"] = float(np.percentile(valid_lengths, 50))
                row["pkt_len_p75"] = float(np.percentile(valid_lengths, 75))
                row["pkt_len_p95"] = float(np.percentile(valid_lengths, 95))
                row["packet_count"] = len(valid_lengths)
                row["byte_count"] = float(np.sum(valid_lengths))
            else:
                # 空流
                row["pkt_len_mean"] = 0.0
                row["pkt_len_std"] = 0.0
                row["pkt_len_min"] = 0.0
                row["pkt_len_max"] = 0.0
                row["pkt_len_p50"] = 0.0
                row["pkt_len_p75"] = 0.0
                row["pkt_len_p95"] = 0.0
                row["packet_count"] = 0
                row["byte_count"] = 0.0

        # Time: 时间间隔序列，长度 1000
        if "time" in features:
            times = features["time"][i]
            valid_times = times[times >= 0]  # 过滤负值 padding
            if len(valid_times) > 1:
                # 计算 IAT (Inter-Arrival Time)
                iat = np.diff(valid_times)
                iat = iat[iat >= 0]  # 确保非负
                if len(iat) > 0:
                    row["iat_mean"] = float(np.mean(iat))
                    row["iat_std"] = float(np.std(iat))
                    row["iat_min"] = float(np.min(iat))
                    row["iat_max"] = float(np.max(iat))
                    row["iat_p50"] = float(np.percentile(iat, 50))
                    row["iat_p75"] = float(np.percentile(iat, 75))
                    row["iat_p95"] = float(np.percentile(iat, 95))
                    row["duration"] = float(valid_times[-1] - valid_times[0])
                else:
                    row["iat_mean"] = 0.0
                    row["iat_std"] = 0.0
                    row["iat_min"] = 0.0
                    row["iat_max"] = 0.0
                    row["iat_p50"] = 0.0
                    row["iat_p75"] = 0.0
                    row["iat_p95"] = 0.0
                    row["duration"] = 0.0
            else:
                row["iat_mean"] = 0.0
                row["iat_std"] = 0.0
                row["iat_min"] = 0.0
                row["iat_max"] = 0.0
                row["iat_p50"] = 0.0
                row["iat_p75"] = 0.0
                row["iat_p95"] = 0.0
                row["duration"] = 0.0

        # Message Type: 可能是类别编码
        if "message_type" in features:
            msg_type = features["message_type"][i]
            if isinstance(msg_type, (list, np.ndarray)) and len(msg_type) > 0:
                # 转换为 numpy 数组再处理
                msg_type_array = np.array(msg_type, dtype=int)
                row["message_type_mode"] = float(np.bincount(msg_type_array).argmax())
                row["message_type_variety"] = float(len(np.unique(msg_type_array)))
            elif isinstance(msg_type, (int, float, np.integer, np.floating)):
                row["message_type_mode"] = float(msg_type)
                row["message_type_variety"] = 1.0

        rows.append(row)

        if (i + 1) % 5000 == 0:
            print(f"  Processed {i + 1}/{n_samples} samples")

    df = pd.DataFrame(rows)

    # 填充缺失值为 0
    df.fillna(0.0, inplace=True)

    return df


def run_cstnet_pipeline(
    data_dir: Path,
    output_dir: Path,
    max_samples: int = None,
) -> None:
    """
    运行 CSTNET 数据集的完整处理流程

    Args:
        data_dir: CSTNET 数据目录 (例如 flow_500)
        output_dir: 输出目录
        max_samples: 可选，限制每个 split 的最大样本数
    """
    ensure_dir(output_dir)

    print(f"[CSTNET] Loading data from {data_dir}")

    # 加载三个 split
    all_nodes = []

    for split in ["train", "valid", "test"]:
        print(f"\n[CSTNET] Processing {split} split...")
        try:
            features, labels = load_cstnet_split(data_dir, split=split, max_samples=max_samples)
            print(f"  Loaded {len(labels)} samples with {len(features)} feature types")

            nodes = cstnet_to_nodes_dataframe(
                features,
                labels,
                flow_id_prefix=f"{split}_flow",
            )

            # 添加 split 标记
            nodes["split"] = split
            all_nodes.append(nodes)

            print(f"  Converted to {len(nodes)} nodes with {len(nodes.columns)} features")

        except Exception as e:
            print(f"  Error processing {split}: {e}")
            continue

    if not all_nodes:
        raise RuntimeError("No data was successfully loaded")

    # 合并所有 split
    combined = pd.concat(all_nodes, ignore_index=True)

    print(f"\n[CSTNET] Total nodes: {len(combined)}")
    print(f"[CSTNET] Total features: {len(combined.columns)}")
    print(f"[CSTNET] Label distribution:")
    print(combined["label"].value_counts().sort_index().head(20))

    # 保存
    output_path = output_dir / "cstnet_nodes.csv"
    combined.to_csv(output_path, index=False)
    print(f"\n[CSTNET] Saved to {output_path}")

    # 也保存 split 特定的文件，方便后续处理
    for split in ["train", "valid", "test"]:
        split_df = combined[combined["split"] == split].copy()
        if len(split_df) > 0:
            split_path = output_dir / f"cstnet_nodes_{split}.csv"
            split_df.to_csv(split_path, index=False)
            print(f"[CSTNET] Saved {split} split to {split_path}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CSTNET-TLS 1.3 数据加载器")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/CSTNET-TLS_1.3/flow_dataset/flow_500"),
        help="CSTNET 数据目录路径",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/cstnet"),
        help="输出目录",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="每个 split 的最大样本数（用于快速测试）",
    )
    return parser


def main(args: List[str] = None) -> None:
    parser = build_argparser()
    parsed = parser.parse_args(args=args)
    run_cstnet_pipeline(
        data_dir=parsed.data_dir,
        output_dir=parsed.output_dir,
        max_samples=parsed.max_samples,
    )


if __name__ == "__main__":
    main()
