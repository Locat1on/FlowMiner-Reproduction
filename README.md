FlowMiner 实验复现指南

本仓库实现了《FlowMiner: A Powerful Model Based on Flow Correlation Mining for Encrypted Traffic Classification》中的完整复现流水线，涵盖数据解析、特征工程、图构建、GNN 训练与经典基线对比。按照下列步骤即可在本地重新跑通实验。

---

## 1. 环境准备

1. 可选：创建虚拟环境
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate  # Windows
   # or source .venv/bin/activate on Linux/macOS
   ```
2. 安装依赖
   ```bash
   pip install -r requirements.txt
   ```
   - PyTorch / PyTorch Geometric 版本与论文一致（2.0.0 / 2.0.4），若有 GPU 请根据本机 CUDA 版本调整官方安装命令。
   - `dpkt` 用于解析 PCAP，`nfstream` 可选（对大流量解析更友好）。

---

## 2. 数据准备与特征提取

1. 将下载好的 PCAP（或 PCAPNG）文件放入 `data/raw/`，同论文一样可使用 CSTNET、ISCX VPN/Tor、LFETT、USTC-TFC 等数据集；若无法获取，可先用公开 HTTPS/TLS 数据调试。
2. 运行特征提取脚本，自动完成：
   - 流级别五元组聚合、客户端/服务器拆分；
   - 基本时序、长度、速率等单向特征；
   - payload Byte 级特征（长度、可打印字符比例、熵等）；
   - 论文描述的分箱 / 算术 / 统计交叉特征。

   ```bash
   python -m src.feature_extraction \
       --raw-dir data/raw \
       --flows-out data/processed/flows.csv \
       --nodes-out data/processed/nodes.csv \
       --nodes-enhanced-out data/processed/nodes_enhanced.csv \
       --default-label 0 \
       --top-k 10 \
       --bin-count 5
   ```

     如需对不同应用/流量类别设置标签，可提供 `--label-map` 指向一个 JSON 文件。键支持文件名子串匹配（忽略大小写），或使用 `dir:目录名` 为某个目录统一赋值；值可以是整数 ID，也可以是字符串类名（自动映射到整数）。

     例如针对 ISCX VPN 数据集，可以使用 `configs/label_maps/vpn_iscx.json`：

     ```jsonc
     {
         "vpn_aim_chat": "aim_chat",
         "vpn_bittorrent": "bittorrent",
         "vpn_skype_audio": "skype_audio",
         // ... 其他模式见文件
     }
     ```

     调用方式：

     ```bash
     python -m src.feature_extraction \
             --raw-dir data/raw/VPN-PCAPs-01 \
             --label-map configs/label_maps/vpn_iscx.json \
             --flows-out data/processed/vpn_flows.csv \
             --nodes-out data/processed/vpn_nodes.csv \
             --nodes-enhanced-out data/processed/vpn_nodes_enhanced.csv
     ```

   生成的 `nodes_enhanced.csv` 将作为后续预处理与图构建的输入。

---

## 3. 节点预处理（缺失填补与归一化）

在构图前，对节点特征进行缺失值填补与 Min–Max 归一化，同时可以过滤样本极少的类别，使数据分布更稳定。

```bash
python -m src.preprocess_nodes \
    --input-csv data/processed/nodes_enhanced.csv \
    --output-csv data/processed/nodes_preprocessed.csv \
    --min-samples-per-class 10
```

- 数值列会先用列均值填补缺失，再缩放到 [0, 1] 区间；
- `--min-samples-per-class` 控制最小类别样本数，过小的类别会被丢弃（可按数据集实际情况调整或设为 1 表示不过滤）。

之后的图构建默认使用 `nodes_preprocessed.csv`。

---

## 4. 构建流交互图

脚本会在单个子图中混入若干流，并按以下准则连边：

- **真实连接边**：同一 flow 的客户端/服务器节点互连；
- **节点信息关联边**：IP 完全相同或同一 C 类/IPv6 /64 前缀；
- **内容关联边**：相同 `content_signature`（端口+协议，可扩展为 SNI、DNS 等）的节点相连。

```bash
python -m src.graph_construction \
    --nodes-csv data/processed/nodes_preprocessed.csv \
    --output-dir data/graphs \
    --max-nodes 500
```

- `--max-nodes` 控制每个子图的最大节点数，默认 500，与论文中敏感性实验的默认设定一致；
- 脚本会把每个子图保存为 `data/graphs/graph_XXXX.pt`，并额外生成 `dataset.pt` 方便批量加载。

---

## 5. 训练 FlowMiner

`src/model.py` 实现了 GraphSAGE→GATConv→集成池化→全连接分类头，配合 BatchNorm 与非线性激活；`src/train.py` 负责数据加载、stratified 8/1/1 划分、500 epoch 训练、10 次重复实验并保存日志/权重。

```bash
# 论文同款配置（推荐）：
python -m src.train --graph-dir data/graphs --output-dir results --preset paper

# 或者自行指定超参：
python -m src.train \
    --graph-dir data/graphs \
    --output-dir results \
    --epochs 500 \
    --batch-size 64 \
    --learning-rate 1e-3 \
    --runs 10
```

- `--preset paper` 会自动套用论文默认超参（epochs=500, batch=64, runs=10, hidden_dim=128, gat_layers=1, gat_heads=4 等），便于一键复现实验。
- Checkpoint 将写入 `results/checkpoints/flowminer_run*.pt`；
- 训练与验证指标逐 epoch 记录到 `results/logs/run_*_log.json`；
- `results/logs/summary.json` 汇总多次实验的平均 Accuracy / Precision / Recall / F1（宏平均）。

需要估算模型规模或 FLOPs 时，可运行 `scripts/analyze_model.py`：

```bash
python scripts/analyze_model.py \
    --input-dim 256 \
    --num-classes 5 \
    --hidden-dim 128 \
    --sage-layers 2 \
    --gat-layers 1 \
    --gat-heads 4 \
    --num-nodes 2000 \
    --num-edges 40000
```

- 参数量来自 `FlowMiner` 可训练权重的精确计数；
- FLOPs 由 `torch.profiler` 在一张合成图上的单次前向推理统计，`--num-nodes/--num-edges` 建议填写数据集中“单 batch 的平均节点/边数”，即可得到更贴近真实的估算。

### 可视化训练曲线

`scripts/plot_training.py` 可直接读取训练日志生成 PNG 图像：

- **单个 run**：

    ```bash
    python scripts/plot_training.py --log results/logs/run_0_log.json
    ```

- **多 run 均值 + 标准差阴影**（自动聚合 `run_*_log.json`）：

    ```bash
    python scripts/plot_training.py \
            --log-dir results/logs \
            --output results/plots/runs_avg.png
    ```

    也可组合 `--log run_0_log.json run_2_log.json` 精确挑选 run。

- 图中会自动用散点和文字标出 **验证 Accuracy** 与 **验证 F1** 的最佳 epoch，并给出对应数值，方便比对不同实验；
- 对多 run 聚合时，曲线默认绘制均值并以浅色带展示标准差，可用 `--output` 自定义文件名。

---

## 6. 模型评估

可单独加载任意 checkpoint，计算宏平均 Accuracy/Precision/Recall/F1。

```bash
python -m src.evaluate \
    --graph-dir data/graphs \
    --checkpoint results/checkpoints/flowminer_run0.pt \
    --batch-size 64 \
    --output-json results/reports/eval_run0.json
```

如需对比不同 run 或不同超参组合，可多次运行评估脚本，将结果写入不同的 JSON 文件后统一整理。

---

## 7. 经典基线与消融

`src/baselines.py` 提供随机森林与（可选）LightGBM，对不同特征子集（仅长度、仅字节、全部）做 K 折评估，以复现实验中的传统方法对比。

```bash
python -m src.baselines \
    --nodes-csv data/processed/nodes_preprocessed.csv \
    --feature-mode full \
    --folds 5 \
    --with-lightgbm \
    --output-json results/reports/baselines.json
```

- 若未安装 LightGBM，可去掉 `--with-lightgbm`，仅运行随机森林基线；
- 如需做特征消融，可在生成 `nodes_enhanced.csv` 时修改特征配置，或在预处理前后对 CSV 进行列筛选后再运行 baselines。

---

## 8. 针对不同任务的数据组织说明

论文中包含多个数据集/任务（如 CSTNET-TLS、ISCX VPN/Tor、LFETT-Vmess/SSR、USTC-TFC 等）。当前代码采用**统一流水线**：

1. 每个任务的数据单独放在 `data/raw/<TASK_NAME>/` 目录下，例如：
   - `data/raw/CSTNET/`
   - `data/raw/ISCX_VPN_TOR/` 等。
2. 针对某个任务实验时：
   - 仅将该任务的 PCAP 放入对应目录；
   - 在特征提取脚本中使用该目录作为 `--raw-dir`，并设置合适的 `--default-label` 或在生成的 CSV 里手动修改 `label` 列；
   - 按照本 README 第 2–7 步完成预处理、构图、训练和基线评估。

未来可以进一步加入 YAML/JSON 配置文件，实现“一键运行 Task1–Task5”的脚本化流程，但当前版本已经可以通过更换 `raw-dir` 和标签配置来分别复现各个任务。

---

## 9. 推荐目录结构

```text
FlowMiner-Reproduction/
├── data/
│   ├── raw/                       # 原始 PCAP/日志，按任务或数据集分子目录
│   ├── processed/
│   │   ├── flows.csv              # 流记录（便于调试）
│   │   ├── nodes.csv              # 基础节点特征
│   │   ├── nodes_enhanced.csv     # 含交叉与 Byte 特征的节点
│   │   └── nodes_preprocessed.csv # 填补+归一化后的节点特征
│   └── graphs/
│       ├── graph_0000.pt
│       └── dataset.pt
├── results/
│   ├── checkpoints/
│   ├── logs/                      # 每个 run 的训练日志与 summary
│   └── reports/                   # 评估、基线、消融报告
├── src/
│   ├── feature_extraction.py
│   ├── preprocess_nodes.py
│   ├── graph_construction.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── baselines.py
│   └── utils.py
├── requirements.txt
└── README.md
```

---

## 10. 额外提示

- **自定义标签**：`feature_extraction.py` 中默认给所有流打同一标签，可在解析数据前对不同目录/文件赋值或在生成的 CSV 里更新 `label` 列，然后继续后续步骤。
- **扩展内容特征**：可在 `content_signature` 字段中加入 SNI、DNS Query 或应用特定指纹，以加强内容关联边。
- **设备配置**：所有脚本默认优先使用 CUDA，如需强制 CPU，可在训练/评估脚本中提供相应的 `--device cpu` 或修改默认设置（视代码实现而定）。
- **类别不平衡**：预处理阶段可通过 `--min-samples-per-class` 去掉极端少数类；若仍然不平衡，可在 `src.train` 中加入类别权重或重新采样逻辑。

完成以上步骤即可得到与论文高度一致的实验流程，并能根据需要扩展更多特征与模型模块。祝实验顺利!

---

## 11. 一键多数据集与消融实验脚本

为了更方便地在多个数据集上重复整个流水线（特征提取 → 预处理 → 构图 → 训练），以及进行论文中的 BF/CF/IDP 消融，本仓库提供了统一脚本：

```bash
python scripts/run_dual_experiments.py [选项...]
```

### 11.1 已内置的数据集配置

脚本中预定义了若干 `DatasetConfig`（见 `scripts/run_dual_experiments.py`）：

- `vpn`：ISCX VPN 小子集（`data/raw/VPN-PCAPS-small`）
- `vpn_full`：完整 ISCX VPN 数据（`data/raw/VPN-PCAPS`）
- `nonvpn`：ISCX Non-VPN 数据（`data/raw/NonVPN-PCAPs-01`）
- `ustc`：USTC-TFC2016（`data/raw/USTC-TFC2016-master`）
- `ssr`：LFETT-SSR（`data/raw/ssr/ssr`）
- `vmess`：LFETT-Vmess（`data/raw/vmess/vmess`）

调用时用 `--datasets` 选择要运行的数据集，默认会跑全部。

### 11.2 基本用法（完整流水线）

以 SSR 数据集为例，只跑 50 个 epoch：

```bash
python scripts/run_dual_experiments.py \
    --datasets ssr \
    --train-epochs 50
```

常用参数说明：

- `--datasets <name1> <name2> ...`：要运行的数据集名（默认全部）。
- `--train-epochs N`：训练 epoch 数（不要同时加 `--preset paper`，否则会被覆盖为 500）。
- `--min-samples K`：在预处理阶段过滤样本数少于 K 的类别，对应论文中“移除样本过少的类别”的做法。
- `--skip-train`：只做特征提取 + 预处理 + 构图，不训练。
- `--only-train`：只在已有图上训练（跳过特征提取和构图）。
- `--graph-limit N`：训练时只加载前 N 个子图（快速 smoke test 用）。

### 11.3 FlowMiner 模块消融（BF / CF / IDP）

脚本支持直接打开/关闭三个模块，方便复现论文的消融实验：

- `--disable-byte-features`：在构图阶段关闭 **字节特征（BF）**  
  - 底层效果：转发给 `src.graph_construction` 的 `--disable-byte-features`，移除 `payload_*`、`byte_bucket_*`、`first_byte_*` 等列。
- `--disable-cross-features`：关闭 **交叉特征（CF）**  
  - 底层效果：转发到 `--disable-cross-features`，不再选取 `*_plus_*`、`*_times_*` 特征列。
- `--disable-idp`：关闭 **Integrated Decision Pooling（IDP）**  
  - 底层效果：转发到 `src.train`，令 `FlowMiner` 使用简单的 global mean pooling，而不是节点级 MLP + Dropout。

推荐的四组配置（以 ISCX VPN 为例）：

```bash
# 1) 完整模型（Full）
python scripts/run_dual_experiments.py \
    --datasets vpn \
    --train-epochs 50

# 2) 去掉 BF
python scripts/run_dual_experiments.py \
    --datasets vpn \
    --disable-byte-features \
    --suffix no_bf \
    --train-epochs 50

# 3) 去掉 CF
python scripts/run_dual_experiments.py \
    --datasets vpn \
    --disable-cross-features \
    --suffix no_cf \
    --train-epochs 50

# 4) 去掉 IDP
python scripts/run_dual_experiments.py \
    --datasets vpn \
    --disable-idp \
    --suffix no_idp \
    --train-epochs 50
```

其中 `--suffix TAG` 会自动把本次实验的中间文件与结果隔离开，例如：

- `processed`：`vpn_flows_TAG.csv` / `vpn_nodes_TAG*.csv`
- `graphs`：`data/graphs/vpn_TAG`
- `results`：`results/vpn_TAG`

这样可以同时保存多组实验（Full / -BF / -CF / -IDP），方便对比论文中的消融结果。
