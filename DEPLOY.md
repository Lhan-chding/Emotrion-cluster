# CVCL Music Emotion Clustering — A100 部署与训练指南

## 1. 需要上传的文件

### 1.1 在服务器上创建目录结构

先在服务器上创建所有需要的目录（假设工作根目录为 `~/cvcl`）：

```bash
# 项目根目录
mkdir -p ~/cvcl

# ---- 代码目录 ----
mkdir -p ~/cvcl/cluster/scripts
mkdir -p ~/cvcl/cluster/cluster/data
mkdir -p ~/cvcl/cluster/cluster/models
mkdir -p ~/cvcl/cluster/cluster/pipeline
mkdir -p ~/cvcl/cluster/cluster/preprocessing

# ---- 数据目录 ----
mkdir -p ~/cvcl/processed_dir
mkdir -p ~/cvcl/aligned_root

# ---- 训练输出目录 ----
mkdir -p ~/cvcl/output
```

### 1.2 上传代码文件

将项目代码上传到 `~/cvcl/cluster/`，对应关系如下：

```
~/cvcl/cluster/                       # 项目根目录
├── requirements.txt                  # Python 依赖
├── scripts/                          # 上传以下文件到此目录
│   ├── __init__.py
│   ├── run_pipeline.py               # 主训练入口
│   ├── rerun_search.py               # 复用 checkpoint 重跑 K 搜索
│   └── build_metadata.py             # 单独构建 metadata 特征
└── cluster/                          # Python 包，上传以下文件到对应子目录
    ├── __init__.py
    ├── config.py                     # 配置 & dataclass
    ├── utils.py                      # ArrayCache + 共享工具函数
    ├── data/
    │   ├── __init__.py
    │   ├── loader.py                 # 数据集 & DataLoader
    │   └── metadata.py               # 元数据特征工程 (TF-IDF)
    ├── models/
    │   ├── __init__.py
    │   └── discovery_net.py          # MusicMetadataDiscoveryNet 模型
    ├── pipeline/
    │   ├── __init__.py
    │   ├── k_selection.py            # K 选择策略 (composite/hierarchical/bic_only)
    │   ├── train.py                  # 完整训练 + 聚类流水线
    │   └── rerun.py                  # 复用 checkpoint 重跑聚类
    └── preprocessing/
        ├── __init__.py
        └── align.py                  # 音频-歌词对齐
```

### 1.3 上传数据文件

将**已处理好的数据文件**上传到 `~/cvcl/processed_dir/`：

```
~/cvcl/processed_dir/                 # 逐个上传以下文件
├── audio.npy                         # 音频 VA 特征 [N, 2]
├── lyrics.npy                        # 歌词 VA 特征 [N, 2]
├── consistency.npy                   # 一致性分数 [N]
├── va_diff.npy                       # VA 差异 [N, 2]
├── labels_emotion.npy                # 情感象限标签 [N]
├── metadata.npy                      # 元数据特征矩阵 [N, D]
├── metadata_feature_names.json       # 特征名列表
├── metadata_vocab.json               # 词汇表
├── canonical_metadata.csv            # 规范化元数据表
├── track_index.tsv                   # 曲目索引
├── meta.json                         # 数据集元信息
└── split_70_15_15.json               # 数据划分
```

将**对齐后的元数据文件**上传到 `~/cvcl/aligned_root/`：

```
~/cvcl/aligned_root/                  # 上传以下 2 个文件
├── aligned_audio_metadata.csv
└── aligned_lyrics_metadata.csv
```

### 1.4 验证目录完整性

上传完成后，运行以下命令确认所有文件就位：

```bash
# 检查代码文件
echo "=== Code ==="
find ~/cvcl/cluster -type f -name '*.py' | sort
echo ""
echo "=== requirements.txt ==="
ls ~/cvcl/cluster/requirements.txt

# 检查数据文件
echo ""
echo "=== Processed Data ==="
ls ~/cvcl/processed_dir/

echo ""
echo "=== Aligned Root ==="
ls ~/cvcl/aligned_root/
```

---

## 2. 服务器环境配置

### 2.1 系统要求

- **OS**: Ubuntu 22.04
- **GPU**: NVIDIA A100-SXM4-40GB
- **CUDA**: 12.4
- **Python**: 3.10+

### 2.2 创建虚拟环境

```bash
# 登录服务器后
cd ~

# 创建 conda 环境（推荐）
conda create -n cvcl python=3.10 -y
conda activate cvcl

# 或者用 venv
python3 -m venv cvcl_env
source cvcl_env/bin/activate
```

### 2.3 安装 PyTorch (CUDA 12.4)

```bash
# PyTorch 官方 CUDA 12.4 版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 2.4 安装项目依赖

```bash
cd ~/cvcl/cluster   # 项目根目录
pip install -r requirements.txt

# 额外依赖（requirements.txt 未列出但代码需要）
pip install scipy joblib
```

### 2.5 验证安装

```bash
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'BF16 support: {torch.cuda.is_bf16_supported()}')
"

# 验证包导入
python -c "from cluster.pipeline.train import main; print('Import OK')"
```

---

## 3. 训练命令

### 3.1 完整训练流水线（推荐）

```bash
# 设置项目根目录到 PYTHONPATH
cd ~/cvcl/cluster
export PYTHONPATH=~/cvcl/cluster:$PYTHONPATH

# 完整训练 + composite K 选择（推荐配置）
python scripts/run_pipeline.py \
    --aligned_root ~/cvcl/aligned_root \
    --processed_dir ~/cvcl/processed_dir \
    --out_dir ~/cvcl/output \
    --epochs 120 \
    --batch_size 256 \
    --learning_rate 5e-4 \
    --weight_decay 1e-4 \
    --gpu 0 \
    --seed 42 \
    --dropout 0.1 \
    --metadata_logit_offset -0.5 \
    --grad_clip_norm 1.0 \
    --use_amp true \
    --early_stopping_patience 15 \
    --scheduler_T0 20 \
    --scheduler_Tmult 2 \
    --gate_entropy_weight 0.01 \
    --k_strategy composite \
    --k_min 4 \
    --k_max 20 \
    --covariance_type full \
    --stability_runs 5 \
    --cluster_feature_strategy full \
    --split_protocol 70_15_15
```

### 3.2 层次聚类策略（论文推荐）

适合需要可解释标签（M1-a, M1-b, M2-a...）的论文写作：

```bash
python scripts/run_pipeline.py \
    --aligned_root ~/cvcl/aligned_root \
    --processed_dir ~/cvcl/processed_dir \
    --out_dir ~/cvcl/output_hierarchical \
    --epochs 120 \
    --batch_size 256 \
    --learning_rate 5e-4 \
    --weight_decay 1e-4 \
    --gpu 0 \
    --dropout 0.1 \
    --use_amp true \
    --early_stopping_patience 15 \
    --k_strategy hierarchical \
    --covariance_type full \
    --cluster_feature_strategy fused_residual
```

### 3.3 复用已有 checkpoint 重跑 K 搜索

不重新训练模型，只用不同 K 策略重新聚类：

```bash
python scripts/rerun_search.py \
    --processed_dir ~/cvcl/processed_dir \
    --run_dir ~/cvcl/output \
    --out_dir ~/cvcl/rerun_output \
    --k_strategy composite \
    --k_min 4 \
    --k_max 24 \
    --covariance_type full \
    --stability_runs 5 \
    --cluster_feature_strategy fused_residual \
    --pca_target_dim 32
```

### 3.4 单独构建元数据特征

```bash
python scripts/build_metadata.py \
    --aligned_root ~/cvcl/aligned_root \
    --processed_dir ~/cvcl/processed_dir \
    --min_token_freq 3 \
    --max_tokens_per_field 128
```

---

## 4. K 选择策略说明

| 策略 | 命令参数 | 说明 |
|------|---------|------|
| **composite** | `--k_strategy composite` | 综合评分：BIC(0.3) + 轮廓系数(0.3) + 最小簇大小(0.2) + 稳定性(0.2) |
| **hierarchical** | `--k_strategy hierarchical` | 两级聚类：宏观(4-8) + 微观(2-5)，标签格式 M1-a, M1-b |
| **bic_only** | `--k_strategy bic_only` | 仅 BIC + 最小簇大小约束（向后兼容） |

## 5. 聚类特征策略说明

| 策略 | 命令参数 | 说明 |
|------|---------|------|
| **full** | `--cluster_feature_strategy full` | z_fused + z_audio + z_lyrics + z_metadata + gate + conflict (~70D) |
| **fused_residual** | `--cluster_feature_strategy fused_residual` | z_fused + 残差(z_audio-z_fused) + 残差(z_lyrics-z_fused) + gate + conflict |
| **fused_only** | `--cluster_feature_strategy fused_only` | z_fused + gate + conflict (~22D) |
| **pca_reduced** | `--cluster_feature_strategy pca_reduced` | full 策略 + PCA 降维到 `--pca_target_dim` |

---

## 6. 训练输出

训练完成后 `out_dir` 包含：

```
out_dir/
├── models/
│   ├── music_discovery_model_best.pth      # 最佳模型权重
│   └── music_discovery_model_best.meta.json # 模型元信息
├── training_history.csv                     # 训练损失历史
├── discovery_gmm_bundle.pkl                 # GMM 模型 + scaler
├── pipeline_summary.json                    # 流水线摘要
├── pipeline_report.md                       # 可读报告
├── hierarchical_label_names.json            # (仅 hierarchical 策略)
├── train/                                   # 各 split 的聚类结果
│   ├── cluster_assignments.npy
│   ├── cluster_summary.json
│   ├── va_scatter_by_cluster.png
│   └── ...
├── val/
├── test/
└── all/
```

---

## 7. 常用调参建议

### A100 专属优化（已默认启用）

- `--use_amp true`：BF16 混合精度，A100 原生支持
- `--covariance_type full`：A100 内存充足，使用完整协方差矩阵
- `--batch_size 256`：可根据数据量调大到 512

### 如果训练不稳定

- 降低学习率：`--learning_rate 1e-4`
- 增大 dropout：`--dropout 0.2`
- 减小梯度裁剪：`--grad_clip_norm 0.5`

### 如果聚类质量不佳

- 扩大搜索范围：`--k_min 3 --k_max 30`
- 增加稳定性评估次数：`--stability_runs 10`
- 尝试 fused_residual 策略：`--cluster_feature_strategy fused_residual`
- 尝试层次聚类：`--k_strategy hierarchical`

### 并行控制

```bash
# 控制 GMM 搜索并行度（默认使用所有 CPU 核心）
export CLUSTER_N_JOBS=8
```

---

## 8. 快速验证（5 epoch 烟雾测试）

```bash
python scripts/run_pipeline.py \
    --aligned_root ~/cvcl/aligned_root \
    --processed_dir ~/cvcl/processed_dir \
    --out_dir ~/cvcl/smoke_test \
    --epochs 5 \
    --k_strategy composite \
    --k_min 4 \
    --k_max 8 \
    --use_amp true \
    --gpu 0
```

验证：
- 训练 loss 逐 epoch 下降
- `pipeline_summary.json` 生成且 `selected_k` 合理
- 各 split 目录下有 `cluster_assignments.npy` 和可视化图
