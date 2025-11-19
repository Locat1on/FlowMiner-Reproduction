import argparse
import copy
import json
import random
import time
from pathlib import Path
from statistics import mean
from typing import List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch_geometric.loader import DataLoader

from .evaluate import evaluate_model, load_graph_dataset
from .model import FlowMiner, count_parameters
from .utils import ensure_dir, save_json, set_seed, stratified_split, split_by_pcap


def train_one_epoch(model: FlowMiner, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    criterion = nn.CrossEntropyLoss()
    model.train()
    running_loss = 0.0
    for batch_idx, batch in enumerate(loader):
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch)
        loss = criterion(logits, batch.y.view(-1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / max(1, len(loader))


def split_dataset(graphs: Sequence, seed: int) -> Tuple[List, List, List]:
    # 优先尝试基于 PCAP 文件名的切分，以防止同源流泄露
    if graphs and hasattr(graphs[0], "flow_ids") and graphs[0].flow_ids:
        try:
            print("[Train] Using PCAP-based splitting to prevent data leakage...")
            return split_by_pcap(graphs, seed=seed)
        except Exception as e:
            print(f"[Train] PCAP-based splitting failed ({e}), falling back to random stratified split.")

    try:
        return stratified_split(graphs, seed=seed)
    except ValueError:
        shuffled = list(graphs)
        random.Random(seed).shuffle(shuffled)
        n = len(shuffled)
        train_end = max(1, int(0.8 * n))
        val_end = max(train_end + 1, int(0.9 * n))
        return (
            shuffled[:train_end],
            shuffled[train_end:val_end],
            shuffled[val_end:] or shuffled[-1:],
        )


def run_training(
    graphs,
    args,
    output_dir: Path,
) -> dict:
    device = torch.device("cpu" if args.use_cpu or not torch.cuda.is_available() else "cuda")
    print(f"[Train] Using device: {device}")
    checkpoints_dir = output_dir / "checkpoints"
    logs_dir = output_dir / "logs"
    ensure_dir(checkpoints_dir)
    ensure_dir(logs_dir)

    run_metrics = []

    if len(graphs) < 3:
        raise ValueError("At least three graphs are required to perform train/val/test splits.")

    print(f"[Train] Total graphs loaded: {len(graphs)}")

    for run_idx in range(args.runs):
        print("=" * 80)
        print(f"[Train] Starting run {run_idx+1}/{args.runs} ...")
        seed = args.seed + run_idx
        set_seed(seed)
        train_data, val_data, test_data = split_dataset(graphs, seed=seed)

        if not train_data or not val_data or not test_data:
            raise ValueError("Dataset split produced an empty partition; provide more graphs or adjust ratios.")

        print(
            f"[Train] Split sizes - train: {len(train_data)}, val: {len(val_data)}, test: {len(test_data)}"
        )

        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=args.batch_size)
        test_loader = DataLoader(test_data, batch_size=args.batch_size)

        input_dim = train_data[0].num_features
        # 从所有图的 y 中统计类别数量
        all_labels = torch.cat([g.y.view(-1) for g in graphs], dim=0)
        unique_labels = torch.unique(all_labels)
        num_classes = int(unique_labels.numel())
        if num_classes < 2:
            raise ValueError(
                f"[Train] Detected only {num_classes} class in labels: "
                f"unique labels = {unique_labels.tolist()}. "
                "Training a classifier需要至少两个类别，请检查特征提取/预处理/图构建阶段的 label 管道。"
            )
        model = FlowMiner(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=args.hidden_dim,
            sage_layers=args.sage_layers,
            gat_layers=args.gat_layers,
            gat_heads=args.gat_heads,
            dropout=args.dropout,
            use_idp=not args.disable_idp,
        ).to(device)

        # 将任意标签值（例如 {2,3,4,6}）重映射为连续的 [0..num_classes-1]，
        # 以满足 CrossEntropyLoss 对标签范围的要求 `0 <= t < num_classes`。
        label_map = {int(lbl): idx for idx, lbl in enumerate(unique_labels.tolist())}
        print(f"[Train] Original labels: {unique_labels.tolist()}")
        print(f"[Train] Remapped label map: {label_map}")
        for g in graphs:
            g.y = torch.tensor([label_map[int(v)] for v in g.y.view(-1)], dtype=torch.long)
        print(
            f"[Train] Model initialized with input_dim={input_dim}, num_classes={num_classes}, "
            f"hidden_dim={args.hidden_dim}, sage_layers={args.sage_layers}, "
            f"gat_layers={args.gat_layers}, gat_heads={args.gat_heads}, dropout={args.dropout}, "
            f"use_idp={not args.disable_idp}"
        )
        print(f"[Train] Total trainable parameters: {count_parameters(model)}")

        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

        best_state = None
        best_val_f1 = -1.0
        history = []

        for epoch in range(1, args.epochs + 1):
            start = time.time()
            train_loss = train_one_epoch(model, train_loader, optimizer, device)
            val_metrics = evaluate_model(model, val_loader, device)
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics["accuracy"],
                    "val_precision": val_metrics["precision"],
                    "val_recall": val_metrics["recall"],
                    "val_f1": val_metrics["f1_macro"],
                    "elapsed_sec": time.time() - start,
                }
            )
            if epoch % max(1, args.epochs // 10) == 0 or epoch == 1:
                print(
                    f"[Train][Run {run_idx+1}/{args.runs}][Epoch {epoch}/{args.epochs}] "
                    f"train_loss={train_loss:.4f}, val_loss={val_metrics['loss']:.4f}, "
                    f"val_acc={val_metrics['accuracy']:.4f}, val_f1={val_metrics['f1_macro']:.4f}"
                )
            if val_metrics["f1_macro"] > best_val_f1:
                best_val_f1 = val_metrics["f1_macro"]
                best_state = copy.deepcopy(model.state_dict())
                print(
                    f"[Train][Run {run_idx+1}/{args.runs}] New best val_f1={best_val_f1:.4f} at epoch {epoch}"
                )

        model.load_state_dict(best_state)
        test_metrics = evaluate_model(model, test_loader, device)
        run_info = {
            "run": run_idx,
            "seed": seed,
            "best_val_f1": best_val_f1,
            "test_metrics": test_metrics,
            "history": history,
            "parameter_count": count_parameters(model),
        }
        run_metrics.append(run_info)

        print(
            f"[Train][Run {run_idx+1}/{args.runs}] Test metrics: "
            f"acc={test_metrics['accuracy']:.4f}, f1={test_metrics['f1_macro']:.4f}"
        )

        checkpoint_payload = {
            "model_state": best_state,
            "config": {
                "input_dim": input_dim,
                "num_classes": num_classes,
                "hidden_dim": args.hidden_dim,
                "sage_layers": args.sage_layers,
                "gat_layers": args.gat_layers,
                "gat_heads": args.gat_heads,
                "dropout": args.dropout,
                "use_idp": not args.disable_idp,
            },
            "test_metrics": test_metrics,
        }
        ckpt_path = checkpoints_dir / f"flowminer_run{run_idx}.pt"
        log_path = logs_dir / f"run_{run_idx}_log.json"
        torch.save(checkpoint_payload, ckpt_path)
        save_json(run_info, log_path)
        print(f"[Train][Run {run_idx+1}/{args.runs}] Saved checkpoint to {ckpt_path}")
        print(f"[Train][Run {run_idx+1}/{args.runs}] Saved log to {log_path}")

    summary = {
        "runs": args.runs,
        "average_test_f1": mean(item["test_metrics"]["f1_macro"] for item in run_metrics),
        "average_accuracy": mean(item["test_metrics"]["accuracy"] for item in run_metrics),
        "details": run_metrics,
    }
    save_json(summary, logs_dir / "summary.json")
    print(
        f"[Train] Finished all {args.runs} runs. "
        f"Average acc={summary['average_accuracy']:.4f}, "
        f"average f1={summary['average_test_f1']:.4f}"
    )
    print(f"[Train] Summary saved to {logs_dir / 'summary.json'}")
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the FlowMiner model.")
    parser.add_argument("--graph-dir", type=Path, default=Path("data/graphs"))
    parser.add_argument(
        "--graph-limit",
        type=int,
        default=None,
        help="Optional limit on number of graphs to load (for debugging or memory control).",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--sage-layers", type=int, default=2)
    parser.add_argument("--gat-layers", type=int, default=2)
    parser.add_argument("--gat-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-cpu", action="store_true")
    parser.add_argument(
        "--disable-idp",
        action="store_true",
        help="Disable Integrated Decision Pooling (IDP) for ablation experiments.",
    )
    parser.add_argument(
        "--preset",
        choices=["paper"],
        default=None,
        help="Optional preset for hyper-parameters (e.g., 'paper').",
    )
    return parser


def apply_preset(args: argparse.Namespace) -> None:
    if args.preset is None:
        return

    if args.preset == "paper":
        overrides = {
            "epochs": 100,
            "batch_size": 64,
            "learning_rate": 1e-3,
            "weight_decay": 5e-4,
            "runs": 1,
            "graph_limit": None,
            "hidden_dim": 32,
            "sage_layers": 2,
            "gat_layers": 1,
            "gat_heads": 4,
            "dropout": 0.2,
        }
        for key, value in overrides.items():
            setattr(args, key, value)
        print(
            "[Train] Applied 'paper' preset (epochs=500, batch_size=64, runs=10, "
            "hidden_dim=128, gat_layers=1, gat_heads=4, graph_limit=None)"
        )


def main(args: Optional[List[str]] = None) -> None:
    parser = build_argparser()
    parsed = parser.parse_args(args=args)
    apply_preset(parsed)
    print("[Train] Starting FlowMiner training ...")
    print(f"[Train] graph_dir = {parsed.graph_dir}")
    print(f"[Train] output_dir = {parsed.output_dir}")
    print(
        f"[Train] epochs = {parsed.epochs}, batch_size = {parsed.batch_size}, "
        f"learning_rate = {parsed.learning_rate}, runs = {parsed.runs}, seed = {parsed.seed}, "
        f"graph_limit = {parsed.graph_limit}"
    )
    graphs = load_graph_dataset(parsed.graph_dir, limit=parsed.graph_limit)
    print(f"[Train] Loaded {len(graphs)} graphs from {parsed.graph_dir}")
    summary = run_training(graphs, parsed, parsed.output_dir)
    print("[Train] Training finished. Summary:")
    print(json.dumps(summary, indent=2))
if __name__ == "__main__":
    main()
