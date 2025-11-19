import argparse
import itertools
import json
import socket
import struct
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .utils import ensure_dir


@dataclass
class DirectionalPackets:
    timestamps: List[float] = field(default_factory=list)
    lengths: List[int] = field(default_factory=list)
    # Concatenated payload bytes (truncated to a configurable budget per direction).
    payload_bytes: List[int] = field(default_factory=list)

    def add(self, timestamp: float, length: int, payload: Optional[bytes] = None, max_payload_bytes: int = 4096) -> None:
        self.timestamps.append(timestamp)
        self.lengths.append(length)
        if payload:
            # Append only the first 6 bytes per packet (per paper) and cap total length.
            remaining = max_payload_bytes - len(self.payload_bytes)
            if remaining > 0:
                prefix = payload[:6]
                self.payload_bytes.extend(list(prefix[:remaining]))


@dataclass
class FlowAccumulator:
    proto: int
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    label: int
    client_ip: Optional[str] = None
    server_ip: Optional[str] = None
    client_packets: DirectionalPackets = field(default_factory=DirectionalPackets)
    server_packets: DirectionalPackets = field(default_factory=DirectionalPackets)

    def add_packet(self, timestamp: float, length: int, src_ip: str, dst_ip: str, payload: Optional[bytes] = None) -> None:
        if self.client_ip is None:
            self.client_ip = src_ip
            self.server_ip = dst_ip

        direction = "client" if src_ip == self.client_ip else "server"
        bucket = self.client_packets if direction == "client" else self.server_packets
        bucket.add(timestamp, length, payload=payload)


def ip_to_str(raw_ip: bytes) -> str:
    try:
        return socket.inet_ntop(socket.AF_INET, raw_ip)
    except OSError:
        return socket.inet_ntop(socket.AF_INET6, raw_ip)


def iter_pcap_packets(pcap_path: Path) -> Iterable[Tuple[float, bytes]]:
    import dpkt

    with pcap_path.open("rb") as handle:
        magic = handle.read(4)
        handle.seek(0)

        def is_pcapng_header(magic_bytes: bytes) -> bool:
            return magic_bytes == b"\x0a\x0d\x0d\x0a"

        def is_pcap_header(magic_bytes: bytes) -> bool:
            return magic_bytes in {
                b"\xd4\xc3\xb2\xa1",
                b"\xa1\xb2\xc3\xd4",
                b"\x4d\x3c\xb2\xa1",
                b"\xa1\xb2\x3c\x4d",
            }

        if is_pcapng_header(magic):
            reader = dpkt.pcapng.Reader(handle)
        elif is_pcap_header(magic):
            reader = dpkt.pcap.Reader(handle)
        else:
            # Fallback: try PCAP first, then PCAPNG.
            try:
                reader = dpkt.pcap.Reader(handle)
            except (ValueError, dpkt.NeedData):
                handle.seek(0)
                reader = dpkt.pcapng.Reader(handle)

        for ts, buf in reader:
            yield ts, buf


@dataclass
class LabelResolver:
    default_label: int
    directory_labels: Dict[str, int] = field(default_factory=dict)
    filename_patterns: List[Tuple[str, int]] = field(default_factory=list)

    def resolve(self, pcap_path: Path) -> int:
        stem = pcap_path.stem.lower()
        for pattern, label in self.filename_patterns:
            if pattern in stem:
                return label
        parent_name = pcap_path.parent.name.lower()
        if parent_name in self.directory_labels:
            return self.directory_labels[parent_name]
        return self.default_label


def build_label_resolver(default_label: int, label_map_path: Optional[Path]) -> LabelResolver:
    """Create a :class:`LabelResolver` from optional JSON mapping.

    The JSON file should be a dictionary where keys are either:

    - Substrings to look for in the PCAP filename (e.g., ``"vpn_skype_audio"``)
    - Directory specifiers prefixed with ``"dir:"`` (e.g., ``"dir:benign"``)

    Values can be integers or strings. When strings are provided, they are mapped
    to incrementing integer IDs in insertion order.
    """

    directory_labels = {"benign": 0, "malware": 1}
    filename_patterns: List[Tuple[str, int]] = []
    used_ids = set(directory_labels.values()) | {default_label}
    string_to_id: Dict[str, int] = {}
    next_id = 0

    def allocate_label(value: int | str) -> int:
        nonlocal next_id
        if isinstance(value, int):
            used_ids.add(value)
            return value
        key = str(value).lower()
        if key in string_to_id:
            return string_to_id[key]
        while next_id in used_ids:
            next_id += 1
        string_to_id[key] = next_id
        used_ids.add(next_id)
        assigned = next_id
        next_id += 1
        return assigned

    if label_map_path:
        if not label_map_path.exists():
            raise FileNotFoundError(f"Label map file {label_map_path} does not exist")
        data = json.loads(label_map_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Label map JSON must be an object/dictionary")
        for raw_key, raw_value in data.items():
            if raw_value is None:
                continue
            label_id = allocate_label(raw_value)
            key = str(raw_key).strip().lower()
            if key.startswith("dir:"):
                directory_labels[key[4:]] = label_id
            else:
                filename_patterns.append((key, label_id))

    return LabelResolver(
        default_label=default_label,
        directory_labels=directory_labels,
        filename_patterns=filename_patterns,
    )


def parse_pcaps(raw_dir: Path, label_resolver: LabelResolver) -> Dict[str, FlowAccumulator]:
    """Parse all PCAPs under ``raw_dir`` into bidirectional flows with labels."""

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory {raw_dir} does not exist.")

    try:
        import dpkt
    except ImportError as exc:
        raise ImportError("dpkt is required for parsing PCAP files. Install it via `pip install dpkt`.") from exc

    flows: Dict[str, FlowAccumulator] = {}

    for pcap_file in raw_dir.glob("**/*.pcap*"):
        pc_label = label_resolver.resolve(pcap_file)
        pkt_count = 0
        parse_fail_count = 0
        flow_count_before = len(flows)
        first_error_logged = False

        for ts, buf in iter_pcap_packets(pcap_file):
            pkt_count += 1
            try:
                # Try Ethernet first
                eth = dpkt.ethernet.Ethernet(buf)
                ip = eth.data

                # If eth.data is raw bytes (common in VPN/Linux cooked capture),
                # manually parse it as IP
                if isinstance(ip, bytes):
                    try:
                        ip = dpkt.ip.IP(ip)
                    except Exception:
                        try:
                            ip = dpkt.ip6.IP6(ip)
                        except Exception:
                            parse_fail_count += 1
                            if not first_error_logged:
                                print(f"  [DEBUG] {pcap_file.name} eth.data is bytes but not IP/IPv6")
                                first_error_logged = True
                            continue

            except (dpkt.dpkt.UnpackError, AttributeError, struct.error) as e:
                if not first_error_logged:
                    print(f"  [DEBUG] {pcap_file.name} first parse (Ethernet): {type(e).__name__}")
                    first_error_logged = True
                # Maybe it's raw IP (no Ethernet header)
                try:
                    ip = dpkt.ip.IP(buf)
                except Exception:
                    try:
                        ip = dpkt.ip6.IP6(buf)
                    except Exception:
                        parse_fail_count += 1
                        continue

            # Check if it's actually an IP packet
            if not isinstance(ip, (dpkt.ip.IP, dpkt.ip6.IP6)):
                parse_fail_count += 1
                if not first_error_logged:
                    print(f"  [DEBUG] {pcap_file.name} eth.data not IP: {type(ip)}")
                    first_error_logged = True
                continue

            try:
                # Extract protocol and addresses
                proto = ip.p
                src_ip = socket.inet_ntop(socket.AF_INET, ip.src) if isinstance(ip, dpkt.ip.IP) else socket.inet_ntop(socket.AF_INET6, ip.src)
                dst_ip = socket.inet_ntop(socket.AF_INET, ip.dst) if isinstance(ip, dpkt.ip.IP) else socket.inet_ntop(socket.AF_INET6, ip.dst)

                # Try to extract ports if TCP/UDP; otherwise use protocol-based pseudo-ports
                if isinstance(ip.data, (dpkt.tcp.TCP, dpkt.udp.UDP)):
                    trans = ip.data
                    src_port = int(trans.sport)
                    dst_port = int(trans.dport)
                    payload = bytes(trans.data) if isinstance(trans.data, (bytes, bytearray)) else b""
                else:
                    # For non-TCP/UDP (e.g., ESP, GRE, ICMP, or encrypted VPN tunnels),
                    # use proto number as pseudo-port to still create flows
                    src_port = proto
                    dst_port = proto
                    payload = bytes(ip.data) if isinstance(ip.data, bytes) else b""

                key = f"{pcap_file.name}:{proto}:{src_ip}:{src_port}-{dst_ip}:{dst_port}"
                if key not in flows:
                    flows[key] = FlowAccumulator(
                        proto=proto,
                        src_ip=src_ip,
                        dst_ip=dst_ip,
                        src_port=src_port,
                        dst_port=dst_port,
                        label=pc_label,
                    )
                flows[key].add_packet(ts, len(buf), src_ip, dst_ip, payload=payload)
            except Exception as e:
                parse_fail_count += 1
                if pkt_count == 1:  # Log first packet parse failure for debugging
                    print(f"  [DEBUG] First packet parse error in {pcap_file.name}: {type(e).__name__}: {e}")
                continue

        flow_count_after = len(flows)
        new_flows = flow_count_after - flow_count_before
        print(f"[PCAP] {pcap_file.name}: {pkt_count} packets, {parse_fail_count} parse fails, {new_flows} new flows, label={pc_label}")

    return flows


def _packets_to_features(packets: DirectionalPackets) -> Dict[str, float]:
    if not packets.timestamps:
        return {
            "packet_count": 0,
            "byte_count": 0,
            "duration": 0.0,
            "pkt_len_mean": 0.0,
            "pkt_len_std": 0.0,
            "pkt_len_min": 0.0,
            "pkt_len_max": 0.0,
            "pkt_len_p25": 0.0,
            "pkt_len_p50": 0.0,
            "pkt_len_p75": 0.0,
            "pkt_len_p90": 0.0,
            "pkt_len_p95": 0.0,
            "pkt_len_skew": 0.0,
            "pkt_len_kurtosis": 0.0,
            "iat_mean": 0.0,
            "iat_std": 0.0,
            "iat_min": 0.0,
            "iat_max": 0.0,
            "iat_p25": 0.0,
            "iat_p50": 0.0,
            "iat_p75": 0.0,
            "iat_p90": 0.0,
            "iat_p95": 0.0,
            "iat_skew": 0.0,
            "iat_kurtosis": 0.0,
            # Byte-level features
            "payload_len": 0.0,
            "payload_printable_ratio": 0.0,
            "payload_byte_entropy": 0.0,
            "payload_popcount_ratio": 0.0,
            # Byte distribution buckets (8 buckets over 0-255)
            "byte_bucket_0_31": 0.0,
            "byte_bucket_32_63": 0.0,
            "byte_bucket_64_95": 0.0,
            "byte_bucket_96_127": 0.0,
            "byte_bucket_128_159": 0.0,
            "byte_bucket_160_191": 0.0,
            "byte_bucket_192_223": 0.0,
            "byte_bucket_224_255": 0.0,
            "byte_value_mean": 0.0,
            "byte_value_std": 0.0,
            "byte_value_min": 0.0,
            "byte_value_max": 0.0,
            "byte_value_skew": 0.0,
            "byte_value_kurtosis": 0.0,
            # First 6 bytes of payload (normalized to [0,1])
            "first_byte_1": 0.0,
            "first_byte_2": 0.0,
            "first_byte_3": 0.0,
            "first_byte_4": 0.0,
            "first_byte_5": 0.0,
            "first_byte_6": 0.0,
        }

    ts = np.array(packets.timestamps, dtype=np.float64)
    lengths = np.array(packets.lengths, dtype=np.float64)
    duration = float(ts.max() - ts.min()) if ts.size > 1 else 0.0
    iat = np.diff(np.sort(ts))

    def _moments(arr: np.ndarray) -> tuple[float, float, float, float]:
        """Return (mean, std, skew, kurtosis) for a 1D array (population version)."""
        if arr.size == 0:
            return 0.0, 0.0, 0.0, 0.0
        mean_val = float(arr.mean())
        std_val = float(arr.std(ddof=0))
        if std_val <= 0.0:
            return mean_val, std_val, 0.0, 0.0
        centered = arr - mean_val
        m3 = float((centered ** 3).mean())
        m4 = float((centered ** 4).mean())
        skew = m3 / (std_val**3)
        kurt = m4 / (std_val**4)
        return mean_val, std_val, skew, kurt

    # Packet length statistics with quantiles and higher moments
    pkt_len_mean, pkt_len_std, pkt_len_skew, pkt_len_kurt = _moments(lengths)
    pkt_len_min = float(lengths.min())
    pkt_len_max = float(lengths.max())
    pkt_len_p25, pkt_len_p50, pkt_len_p75, pkt_len_p90, pkt_len_p95 = [
        float(v)
        for v in np.percentile(lengths, [25, 50, 75, 90, 95])
    ]

    # IAT statistics (may be empty if只有一个包)
    if iat.size:
        iat_mean, iat_std, iat_skew, iat_kurt = _moments(iat)
        iat_min = float(iat.min())
        iat_max = float(iat.max())
        iat_p25, iat_p50, iat_p75, iat_p90, iat_p95 = [
            float(v)
            for v in np.percentile(iat, [25, 50, 75, 90, 95])
        ]
    else:
        iat_mean = iat_std = iat_min = iat_max = 0.0
        iat_p25 = iat_p50 = iat_p75 = iat_p90 = iat_p95 = 0.0
        iat_skew = iat_kurt = 0.0

    # Byte-level statistics from concatenated payload bytes
    payload_arr = np.array(packets.payload_bytes, dtype=np.uint8) if packets.payload_bytes else None
    if payload_arr is not None and payload_arr.size > 0:
        # Total payload length
        payload_len = float(payload_arr.size)
        # Printable ASCII ratio (rough approximation of "printable" content)
        printable_mask = (payload_arr >= 32) & (payload_arr <= 126)
        printable_ratio = float(printable_mask.mean())
        # Byte value distribution entropy & statistics over raw byte values
        counts = np.bincount(payload_arr, minlength=256).astype(np.float64)
        probs = counts / counts.sum() if counts.sum() > 0 else counts
        nonzero = probs > 0
        entropy = float(-(probs[nonzero] * np.log2(probs[nonzero])).sum())
        byte_mean, byte_std, byte_skew, byte_kurt = _moments(payload_arr.astype(np.float64))
        byte_min = float(payload_arr.min())
        byte_max = float(payload_arr.max())
        # Bit-level popcount: proportion of ones in (truncated) payload bytes
        total_bits = int(payload_arr.size * 8)
        popcount_ratio = float(np.unpackbits(payload_arr).sum() / total_bits) if total_bits > 0 else 0.0

        # Compressed byte distribution: 8 buckets over 0-255
        bucket_size = 32
        bucket_probs = []
        for i in range(8):
            start = i * bucket_size
            end = (i + 1) * bucket_size
            bucket_probs.append(float(probs[start:end].sum()))

        # First 6 bytes of payload, normalized to [0,1] (value/255)
        first_bytes = payload_arr[:6]
        first_bytes_norm = (first_bytes.astype(np.float32) / 255.0).tolist()
        # Pad to length 6
        while len(first_bytes_norm) < 6:
            first_bytes_norm.append(0.0)
    else:
        payload_len = 0.0
        printable_ratio = 0.0
        entropy = 0.0
        popcount_ratio = 0.0
        bucket_probs = [0.0] * 8
        first_bytes_norm = [0.0] * 6
        byte_mean = byte_std = byte_min = byte_max = 0.0
        byte_skew = byte_kurt = 0.0

    return {
        "packet_count": len(packets.timestamps),
        "byte_count": float(lengths.sum()),
        "duration": duration,
        "pkt_len_mean": pkt_len_mean,
        "pkt_len_std": pkt_len_std,
        "pkt_len_min": pkt_len_min,
        "pkt_len_max": pkt_len_max,
        "pkt_len_p25": pkt_len_p25,
        "pkt_len_p50": pkt_len_p50,
        "pkt_len_p75": pkt_len_p75,
        "pkt_len_p90": pkt_len_p90,
        "pkt_len_p95": pkt_len_p95,
        "pkt_len_skew": pkt_len_skew,
        "pkt_len_kurtosis": pkt_len_kurt,
        "iat_mean": iat_mean,
        "iat_std": iat_std,
        "iat_min": iat_min,
        "iat_max": iat_max,
        "iat_p25": iat_p25,
        "iat_p50": iat_p50,
        "iat_p75": iat_p75,
        "iat_p90": iat_p90,
        "iat_p95": iat_p95,
        "iat_skew": iat_skew,
        "iat_kurtosis": iat_kurt,
        "payload_len": payload_len,
        "payload_printable_ratio": printable_ratio,
        "payload_byte_entropy": entropy,
        "payload_popcount_ratio": popcount_ratio,
        "byte_bucket_0_31": bucket_probs[0],
        "byte_bucket_32_63": bucket_probs[1],
        "byte_bucket_64_95": bucket_probs[2],
        "byte_bucket_96_127": bucket_probs[3],
        "byte_bucket_128_159": bucket_probs[4],
        "byte_bucket_160_191": bucket_probs[5],
        "byte_bucket_192_223": bucket_probs[6],
        "byte_bucket_224_255": bucket_probs[7],
        "byte_value_mean": byte_mean,
        "byte_value_std": byte_std,
        "byte_value_min": byte_min,
        "byte_value_max": byte_max,
        "byte_value_skew": byte_skew,
        "byte_value_kurtosis": byte_kurt,
        "first_byte_1": first_bytes_norm[0],
        "first_byte_2": first_bytes_norm[1],
        "first_byte_3": first_bytes_norm[2],
        "first_byte_4": first_bytes_norm[3],
        "first_byte_5": first_bytes_norm[4],
        "first_byte_6": first_bytes_norm[5],
    }


def flows_to_nodes(flows: Dict[str, FlowAccumulator]) -> pd.DataFrame:
    """Convert flows to nodes. Process in chunks to avoid memory issues with large datasets."""
    chunk_size = 100000  # Process 100k flows at a time
    chunks = []
    batch = []

    for flow_id, flow in flows.items():
        for role, packets in (("client", flow.client_packets), ("server", flow.server_packets)):
            stats = _packets_to_features(packets)
            stats.update(
                {
                    "flow_id": flow_id,
                    "proto": flow.proto,
                    "src_ip": flow.src_ip,
                    "dst_ip": flow.dst_ip,
                    "src_port": flow.src_port,
                    "dst_port": flow.dst_port,
                    "role": role,
                    "direction": 0 if role == "client" else 1,
                    "label": flow.label,
                    "content_signature": f"{flow.dst_port}_{flow.proto}",
                    "timestamp": packets.timestamps[0] if packets.timestamps else 0.0,
                    "byte_rate": stats["byte_count"] / stats["duration"]
                    if stats["duration"]
                    else 0.0,
                    "packet_rate": stats["packet_count"] / stats["duration"]
                    if stats["duration"]
                    else 0.0,
                }
            )
            batch.append(stats)

            # Convert batch to DataFrame when it reaches chunk_size
            if len(batch) >= chunk_size:
                chunks.append(pd.DataFrame(batch))
                batch = []

    # Don't forget the last partial batch
    if batch:
        chunks.append(pd.DataFrame(batch))

    # Concatenate all chunks
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()



def enhance_with_cross_features(df: pd.DataFrame, top_k: int, bin_count: int) -> pd.DataFrame:
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    protected_cols = {"flow_id", "direction", "timestamp"}
    numeric_cols = [c for c in numeric_cols if c not in protected_cols]
    if not numeric_cols:
        return df

    std_scores = df[numeric_cols].std().nlargest(min(top_k, len(numeric_cols)))
    selected = std_scores.index.tolist()
    if not selected:
        return df

    # 为避免在大 DataFrame 上频繁逐列赋值导致 "highly fragmented" 性能问题，
    # 这里先在字典中构造所有新列，最后一次性 concat。

    new_cols: Dict[str, pd.Series] = {}

    # 1) 分箱特征
    for col in selected:
        try:
            # 检查数据分布，决定分箱策略
            p75_val = df[col].quantile(0.75)
            nunique = df[col].nunique()

            # 如果 75% 分位数 <= 0 或唯一值太少，说明数据高度偏斜或接近常数
            # 此时 qcut 容易退化为单一箱，改用 cut（等宽）或跳过
            if p75_val <= 0 or nunique < bin_count:
                # 尝试等宽分箱，如果仍然失败则跳过该列
                try:
                    binned = pd.cut(df[col], bins=bin_count, duplicates="drop").cat.codes
                    # 如果等宽分箱也只产生一个箱，就不添加这个特征
                    if binned.nunique() <= 1:
                        continue
                except (ValueError, TypeError):
                    continue
            else:
                # 数据分布相对正常，使用等频分箱
                binned = pd.qcut(
                    df[col], q=min(bin_count, nunique), duplicates="drop"
                ).cat.codes
                # 同样检查是否退化为单箱
                if binned.nunique() <= 1:
                    continue

            new_cols[f"{col}_bin"] = binned
        except (ValueError, TypeError) as e:
            # 任何异常都跳过该列的分箱
            continue

    # 2) 两两交叉特征（和、差、积）
    for a, b in itertools.combinations(selected, 2):
        plus_name = f"{a}_plus_{b}"
        minus_name = f"{a}_minus_{b}"
        times_name = f"{a}_times_{b}"
        new_cols[plus_name] = df[a] + df[b]
        new_cols[minus_name] = df[a] - df[b]
        new_cols[times_name] = df[a] * df[b]

    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    # 相关性特征在大规模数据上非常耗时且会显著增加特征维度，
    # 为贴近论文中“少量 crossing 特征”的设定，这里暂时关闭 per-flow 相关性特征。
    # 如需开启，可取消下面注释，但请注意计算量和显存开销：
    # for flow_id, group in df.groupby("flow_id"):
    #     if len(group) < 2:
    #         continue
    #     corr = group[selected].corr()
    #     for i, a in enumerate(selected):
    #         for j, b in enumerate(selected):
    #             if j <= i:
    #                 continue
    #             value = corr.iloc[i, j]
    #             df.loc[group.index, f"corr_{a}_{b}"] = value if not np.isnan(value) else 0.0

    df.fillna(0.0, inplace=True)
    return df


def run_pipeline(
    raw_dir: Path,
    flows_path: Path,
    nodes_path: Path,
    enhanced_path: Path,
    default_label: int,
    top_k: int,
    bin_count: int,
    label_map_path: Optional[Path],
) -> None:
    ensure_dir(flows_path.parent)
    ensure_dir(nodes_path.parent)
    ensure_dir(enhanced_path.parent)

    print(f"[FlowMiner] Step 1/4: parsing PCAP files from {raw_dir} ...")
    label_resolver = build_label_resolver(default_label=default_label, label_map_path=label_map_path)
    flows = parse_pcaps(raw_dir, label_resolver)
    print(f"[FlowMiner] Step 1/4 done: parsed {len(flows)} flows.")

    print("[FlowMiner] Step 2/4: converting flows to node-level features ...")
    nodes = flows_to_nodes(flows)
    print(f"[FlowMiner] Step 2/4 done: generated {len(nodes)} nodes.")

    print("[FlowMiner] Step 3/4: writing raw flow/node CSV files ...")
    pd.DataFrame(
        [
            {
                "flow_id": fid,
                "proto": acc.proto,
                "src_ip": acc.src_ip,
                "dst_ip": acc.dst_ip,
                "src_port": acc.src_port,
                "dst_port": acc.dst_port,
                "label": acc.label,
            }
            for fid, acc in flows.items()
        ]
    ).to_csv(flows_path, index=False)

    nodes.to_csv(nodes_path, index=False)

    print("[FlowMiner] Step 4/4: enhancing node features with cross & binned features ...")
    enhanced = enhance_with_cross_features(nodes, top_k=top_k, bin_count=bin_count)
    enhanced.to_csv(enhanced_path, index=False)
    print(f"[FlowMiner] Step 4/4 done: enhanced features saved to {enhanced_path}.")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Feature extraction pipeline for FlowMiner reproduction.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"), help="Directory with PCAP/PCAPNG files.")
    parser.add_argument("--flows-out", type=Path, default=Path("data/processed/flows.csv"))
    parser.add_argument("--nodes-out", type=Path, default=Path("data/processed/nodes.csv"))
    parser.add_argument("--nodes-enhanced-out", type=Path, default=Path("data/processed/nodes_enhanced.csv"))
    parser.add_argument("--default-label", type=int, default=0, help="Label assigned when ground truth is unavailable.")
    parser.add_argument(
        "--label-map",
        type=Path,
        default=None,
        help="Optional JSON file mapping filename substrings or dir:<name> to label IDs (or names).",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Number of high-variance features used for crosses.")
    parser.add_argument("--bin-count", type=int, default=5, help="Number of quantile bins for discretization.")
    return parser


def main(args: Optional[List[str]] = None) -> None:
    parser = build_argparser()
    parsed = parser.parse_args(args=args)
    run_pipeline(
        raw_dir=parsed.raw_dir,
        flows_path=parsed.flows_out,
        nodes_path=parsed.nodes_out,
        enhanced_path=parsed.nodes_enhanced_out,
        default_label=parsed.default_label,
        top_k=parsed.top_k,
        bin_count=parsed.bin_count,
        label_map_path=parsed.label_map,
    )


if __name__ == "__main__":
    main()
