import argparse
import json
from pathlib import Path
from typing import List, Optional

import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn
from torch_geometric.loader import DataLoader

from .model import FlowMiner


def load_graph_dataset(graph_dir: Path, limit: Optional[int] = None) -> List:
    """从 graph_dir 加载图数据，优先使用分散的 graph_*.pt，避免强依赖巨大 dataset.pt。

    参数
    ------
    graph_dir: 图文件所在目录
    limit: 可选，只加载前 N 个图，用于调试或减小内存占用
    """

    graph_paths = sorted(graph_dir.glob("graph_*.pt"))
    if limit is not None:
        graph_paths = graph_paths[:limit]

    if graph_paths:
        graphs = [torch.load(p, weights_only=False) for p in graph_paths]
    else:
        # 尝试加载 CSTNET 风格的 split 文件（train_graphs.pt, test_graphs.pt, valid_graphs.pt）
        split_files = ["train_graphs.pt", "test_graphs.pt", "valid_graphs.pt"]
        all_graphs = []
        for split_file in split_files:
            split_path = graph_dir / split_file
            if split_path.exists():
                split_graphs = torch.load(split_path, weights_only=False)
                all_graphs.extend(split_graphs)

        if all_graphs:
            graphs = all_graphs
        else:
            # 兼容 all_graphs.pt 格式
            all_graphs_path = graph_dir / "all_graphs.pt"
            if all_graphs_path.exists():
                graphs = torch.load(all_graphs_path, weights_only=False)
            else:
                # 兼容老的整体 dataset.pt 形式
                dataset_path = graph_dir / "dataset.pt"
                if not dataset_path.exists():
                    raise FileNotFoundError(f"No graphs found in {graph_dir}")
                graphs = torch.load(dataset_path, weights_only=False)

    if not graphs:
        raise FileNotFoundError(f"No graphs found in {graph_dir}")

    if limit is not None and len(graphs) > limit:
        graphs = graphs[:limit]

    return graphs


@torch.no_grad()
def evaluate_model(model: FlowMiner, loader: DataLoader, device: torch.device) -> dict:
    criterion = nn.CrossEntropyLoss()
    model.eval()
    losses, preds, labels = [], [], []

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        target = batch.y.view(-1)
        loss = criterion(logits, target)
        losses.append(loss.item())
        preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
        labels.extend(target.cpu().tolist())

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    return {
        "loss": float(sum(losses) / len(losses)) if losses else 0.0,
        "accuracy": accuracy_score(labels, preds),
        "precision": precision,
        "recall": recall,
        "f1_macro": f1,
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained FlowMiner checkpoint.")
    parser.add_argument("--graph-dir", type=Path, default=Path("data/graphs"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--use-cpu", action="store_true", help="Force CPU evaluation even if CUDA is available.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional path to save metrics as JSON.")
    return parser


def load_model(checkpoint_path: Path, input_dim: int, num_classes: int) -> FlowMiner:
    payload = torch.load(checkpoint_path, map_location="cpu")
    config = payload.get(
        "config",
        {
            "input_dim": input_dim,
            "num_classes": num_classes,
            "hidden_dim": 128,
            "sage_layers": 2,
            "gat_layers": 2,
            "gat_heads": 4,
            "dropout": 0.2,
        },
    )
    config["input_dim"] = config.get("input_dim", input_dim)
    config["num_classes"] = config.get("num_classes", num_classes)
    # 兼容旧 checkpoint：若未记录 use_idp，则默认启用 IDP。
    if "use_idp" not in config:
        config["use_idp"] = True
    model = FlowMiner(**config)
    model.load_state_dict(payload["model_state"])
    return model


def main(args: Optional[List[str]] = None) -> None:
    parser = build_argparser()
    parsed = parser.parse_args(args=args)

    graphs = load_graph_dataset(parsed.graph_dir)
    input_dim = graphs[0].num_features
    num_classes = int(torch.unique(torch.cat([g.y for g in graphs], dim=0)).numel())
    model = load_model(parsed.checkpoint, input_dim=input_dim, num_classes=num_classes)

    device = torch.device("cpu" if parsed.use_cpu or not torch.cuda.is_available() else "cuda")
    loader = DataLoader(graphs, batch_size=parsed.batch_size)
    model.to(device)

    metrics = evaluate_model(model, loader, device)
    if parsed.output_json:
        parsed.output_json.parent.mkdir(parents=True, exist_ok=True)
        with parsed.output_json.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
