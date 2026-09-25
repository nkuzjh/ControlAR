# ControlAR 接入 CSGO Benchmark v2 Seen-10

本文统一记录 ControlAR 的 Seen-10 生成任务：背景与比较口径、数据和模型、环境权重、可执行命令、恢复与评测、现有结果。命令从 `/home/jiahao/task/ControlAR` 执行。只覆盖 **radar/map + 当前 5DoF pose → FPV**，不增加定位任务，不包含 CrossMap-4。

文档分工参考 X-VLA 的运行说明与方案说明，但不复制其定位模型、训练参数或运行状态：

- 本文维护**已经实现**的 legacy、原 aligned 和 PEFT 入口及有证据的运行状态。
- [CSGO_SEEN10_PLAN.md](CSGO_SEEN10_PLAN.md) 维护 PEFT 设计依据、实施边界和验收标准。
- 原 `CSGO_SEEN10_ENV.md` 和 `CSGO_SEEN10_EXP32GEN_ALIGNED.md` 保留跳转入口，内容归并到本文；`csgo_benchmark_v2_start.md` 保留为首次接入历史，不作为当前实验参数的权威来源。

## 1. 实验范围与比较口径

公平比较的主参考是 UniLIP generation-only `exp32_gen`；joint generation+localization `exp32` 仅作次要参考。对齐数据、可用条件信息、generation 样本曝光量、checkpoint 选择和评测协议；原生生成目标、图像处理尺寸、优化方式和采样机制作为模型差异披露，不要求机械一致。

| 实验 | 用途与状态 | 配置/选择方式 | 输出根目录（相对项目根目录） | 主 checkpoint |
| --- | --- | --- | --- | --- |
| 首次接入 legacy | 已实现；seed 0 训练完成，compiled b16 有完整评测 | `configs/csgo_seen10.json`；不传 `--experiment` | `outputs/csgo_benchmark_v2_seen10/ControlAR/seed_0` | 历史推理使用 `best` |
| 原 aligned | 已实现；seed 42 正在训练，见第 8 节快照 | `configs/csgo_seen10_exp32gen_aligned.json`；`--experiment csgo_seen10_exp32gen_aligned` | `outputs/csgo_seen10_exp32gen_aligned/ControlAR/seed_42` | `late`；`best` 为补充 |
| aligned PEFT | 已实现，小批量 GPU 验收完成；待用户手动验收，未启动正式训练 | `configs/csgo_seen10_exp32gen_aligned_peft.json`；`--experiment csgo_seen10_exp32gen_aligned_peft` | `outputs/csgo_seen10_exp32gen_aligned_peft/ControlAR/seed_42` | `late`；`best` 为补充 |

legacy 的 compiled b16 是推理方式，不是另一套训练实验；其已有预测单独位于 `outputs/csgo_benchmark_v2_seen10_compiled_b16/ControlAR/seed_0`。

| 项目 | UniLIP exp32_gen | ControlAR legacy（默认单卡） | ControlAR 已实现 aligned |
| --- | --- | --- | --- |
| 数据与任务 | Seen-10，generation-only | 同一生成数据协议 | 同左 |
| 有效 generation batch | 128 | 1 | 128 = 1 卡 × micro 1 × 累计 128 |
| optimizer updates / 曝光量 | 19,500 / 2,496,000 | 300,000 / 300,000 | 19,500 / 2,496,000 |
| 训练组织 | 50 epochs，实际 19,500 updates | 6 epochs | 50 epochs，以 19,500 updates 为终止条件 |
| 条件视觉输入 / FPV 目标 | 224 / 448 | 448 / 448 | 448 / 448 |
| 条件注入 | radar/map、地图名称、pose 文本 | radar control + 数值 pose/map embedding | 同 legacy |
| 视觉 encoder / 视觉输入 projector | 冻结 | 全量训练 | 全量训练 |
| 主生成网络 | 活跃 LLM、生成 DiT 使用 LoRA；生成 connector 使用 LoRA | GPT 全量训练 | 活跃 GPT 全量训练 |
| tokenizer / decoder | 冻结 VAE | 冻结 VQ | 冻结 VQ |
| 原生目标 | flow matching | AR token cross-entropy | AR token cross-entropy |
| LR / AdamW betas | 1e-4 / (0.9, 0.999) | 5e-5 / (0.9, 0.95) | 5e-5 / (0.9, 0.95) |
| weight decay | 0 | 矩阵 0.05，低维 0 | 同 legacy |
| scheduler | warmup 0.003 + cosine，最低 1e-5 | constant，无 warmup | constant，无 warmup |
| 随机图像增强 | 无 | 无 | 无 |
| 完整 validation / 保存 | 原训练无 validation 选点，使用 final | 每 60,000 updates，共 5 次 | 每 3,900 updates，共 5 次 |
| 主报告选点 | final | 历史 best，不作严格同预算主比较 | late/final；best 单列 |

UniLIP 证据入口：`/home/jiahao/task/UniLIP/csgo_configs/exp32_gen.yaml`、`train_csgo.py`、`unilip/model/language_model/unified_unilip.py`、`record.md` 和 `outputs/csgo_1b/exp32_gen` 的 trainer/adapter 元数据。实际训练记录与代码优先于配置注释。ControlAR 证据为上述 JSON、`train_seen10.py` 与第 8 节运行产物。

## 2. 数据、条件与输出边界

数据根目录为 `/home/jiahao/task/UniLIP/data/csgo_benchmark_v2`。使用发布的 manifest、selection、split 和 calibration，不扫描图片目录自行构造划分。

Seen-10 地图固定为：`cs_agency`、`cs_italy`、`de_ancient`、`de_anubis`、`de_dust2`、`de_inferno`、`de_mirage`、`de_nuke`、`de_overpass`、`de_train`。

| split | 数量 | 用途 |
| --- | ---: | --- |
| seen_train | 50,000 | 训练 |
| seen_validation | 5,000 | 同模型原生 generation loss 验证 |
| seen_discrete_test | 20,000 | 每图 2,000 条离散生成 |
| seen_continuous | 12,800 | 200 clips × 64 frames；每图 20 clips |

模型条件只有当前 radar/map 图、map identity、当前 `[x, y, z, pitch, yaw]`。数值归一化为：

```text
x / 1024
y / 1024
(z - frozen_z_min_map) / (frozen_z_max_map - frozen_z_min_map)
pitch / (2*pi)
yaw / (2*pi)
```

Z min/max 来自发布的逐地图 frozen exact calibration；不使用测试集重新统计。`NumericPoseMapEmbedder` 将数值 pose 和 10 类地图 embedding 转为 120 个原生 caption 条件 token，不运行 T5。该新增模块共 2,371,584 参数；map 名称通过固定 ID 编码，不添加额外场景知识。

ControlAR 使用 GPT-XL、DINOv2-small control adapter、VQ-16；448×448 FPV 对应 28×28 = 784 个 VQ token，词表 16,384。radar 经 DINO 和 projector 注入生成 GPT；维持 448 条件尺寸以匹配现有空间控制路径。图像执行 RGB 转换、确定性 bicubic resize 和模型必需归一化，无随机 crop、flip、ColorJitter、擦除等图像增强。训练的 resid/FFN/token/CFG condition dropout 0.1 是模型正则，不属于随机图像增强。

训练、验证可以读取 target FPV；离散和连续推理在 `infer_seen10.py` 中以 `require_images=False` 解析记录、以 `include_target=False` 构造数据集。target 路径可存在于元数据，但不读取为模型条件。不使用相邻真实 FPV、历史位姿或前一生成帧；连续集逐帧独立生成，保留 clip/frame identity 和官方顺序。

每个 condition 只保存一张 **448×448 RGB JPEG**，相对路径为 `<map>/<file_frame>.jpg`，使用 Pillow `format="JPEG"` 默认编码参数，与已核对的 UniLIP saver 一致。不做 best-of-N，不用 GT 或测试指标筛选图片、checkpoint 或超参数。

## 3. 训练模块、预算和 checkpoint

### 3.1 已实现 aligned 的可训练范围

| ControlAR 参数前缀/模块 | 已实现 aligned 状态 |
| --- | --- |
| `gpt.adapter.model.*`（DINO 视觉 encoder） | 全量训练 |
| `gpt.adapter_mlp.*`（视觉输入 projector） | 全量训练 |
| `gpt.layers.*`（统一生成 GPT，36 层） | 全量训练，不使用 LoRA |
| `gpt.tok_embeddings.*`、`gpt.norm.*` | 全量训练 |
| `pose_map_embedder.*` | 全量训练 |
| `gpt.cls_embedding.cap_proj.*`、`gpt.condition_mlp.cap_proj.*`、`gpt.condition_layers.*` | 全量训练 |
| `gpt.output.weight`（VQ token logits 分类投影） | 全量训练；不是图像 decoder |
| `gpt.condition_embeddings.*`（当前生成路径未使用） | 冻结 |
| 独立 VQ encoder / codebook / decoder | 全部冻结 |

启动时生成逐参数 `audits/trainable_parameters.json`，记录 requires_grad 和 optimizer membership。当前实际 audit 为：不含独立 VQ 的模型参数 838,314,880，可训练/optimizer 收录 817,343,360，冻结 20,971,520。legacy 未显式排除未使用的 condition embedding；不要将 aligned 的审计数直接当成 legacy optimizer 参数数。

优化为 AdamW（环境支持时 fused），LR 5e-5，betas `(0.9, 0.95)`，epsilon 1e-8；ndim≥2 参数 weight decay 0.05，低维参数 0；梯度裁剪 1.0，BF16 autocast，允许 TF32。constant scheduler 不做 warmup。

### 3.2 已实现 aligned 的训练预算

```text
world_size × micro_batch × gradient_accumulation = 1 × 1 × 128 = 128
每 epoch：49,920 个源样本 / 128 = 390 optimizer updates，尾部丢弃 80 条
390 × 50 = 19,500 optimizer updates
19,500 × 128 = 2,496,000 次 generation 样本曝光
2,496,000 / 50,000 = 49.92 个等价完整 epoch
```

loss 在 backward 前除以累计次数；scheduler 和 global step 只在 optimizer update 后更新。CFG 分支和内部 token 不重复计作源样本。原 aligned 代码固定 micro=1、accumulation=128；不能直接传 micro=8/16。独立 PEFT 入口默认 micro=8、accumulation=16；两者允许覆盖，只要求 world size、micro、累计均为正整数且乘积为 128。此规则不改变正在运行的原 aligned 实验。

完整 5,000 条 validation 和保存仅发生在 **3,900、7,800、11,700、15,600、19,500**。以同模型 validation AR loss 选择 `best.pt`；`late.pt` 仅在训练完成时指向 step 19,500。论文主比较用 late/final，best 仅作为补充，不能将外部模型 best 与 UniLIP final 称为相同选点规则。

五个 `step_*.pt` 均为完整恢复点，包含模型、optimizer、scheduler、scaler、global optimizer step、RNG、sampler/dataloader 状态和累计边界。best/late 为同文件系统硬链接，不额外复制大 checkpoint；aligned 不创建 last/latest。只能从已有里程碑恢复，不能从任意未保存的日志 step 恢复，也不允许在同一 run 中退回比已有 checkpoint 更早的进度。

legacy 默认单卡 micro=1、无累计，6 epochs = 300,000 updates；每 60,000 updates 验证保存，共保留 `step_060000.pt` 至 `step_300000.pt` 五个完整状态。`last.pt` 指向最近里程碑，`best.pt` 是验证最优模型，推理默认 best；恢复训练用 last 或完整 step 文件，不用 best。无 latest 别名。

## 4. 环境和官方权重

### 4.1 新服务器环境准备

从新服务器的 ControlAR checkout 根目录执行；不要求用户名、安装目录或 Conda 环境名称相同。Git 不包含 `.venv`、`.venv-eval`、模型权重、数据、共享 evaluator 或历史输出，这些内容须分别准备。不要复制旧服务器的 Python 环境目录。

```bash
# 环境与权重可以分开准备；下列命令由用户手动执行。
bash scripts/setup_csgo_seen10.sh --env-only
./.venv/bin/python scripts/download_csgo_seen10_assets.py

# 共享评测器单独拉取代码、管理环境；已准备过时无需重复安装。
# 在新服务器按共享评测器 README 执行，例如：
bash ../csgo_benchmark_v2_eval_general/setup_env.sh

# 只读检查，不安装、不下载，也不初始化 CUDA。
bash scripts/setup_csgo_seen10.sh --check
./.venv/bin/python scripts/download_csgo_seen10_assets.py --check

# 在目标计算节点另行验证 CUDA/实际 GPU；会执行小矩阵检查。
bash scripts/setup_csgo_seen10.sh --check-cuda
```

不加参数的 `setup_csgo_seen10.sh` 仍为训练环境与模型资产的一键准备。已有兼容环境优先保留，不自动替换已安装的 nightly PyTorch；新环境从 PATH 选择兼容 Python，必要时用 Conda 创建项目内解释器。可用 `CONTROLAR_BOOTSTRAP_PYTHON=/path/to/python3.11` 显式指定，不再默认克隆另一项目环境。显式 `CONTROLAR_CLONE_FROM` 仍用于确实需要复制某个已有 Conda 环境的情况。

新环境默认采用稳定 cu128 PyTorch/torchvision 配对。后端选项见 `bash scripts/setup_csgo_seen10.sh --help`；CPU 后端只适合配置检查，正式 ControlAR BF16 训练/推理仍需支持的 NVIDIA GPU 与驱动。环境选择不改变实验 batch、学习率或模型配置，也不承诺跨 GPU/库版本逐位一致。`--check` 执行关键包 CPU 导入及本地资产哈希检查（`--env-only` / `--eval-only` 时不检查模型资产），确认 CUDA 未初始化；它不能替代新服务器的 `--check-cuda` 和独立 smoke。共享评测环境由其独立仓库的 `setup_env.sh` 和 README 管理；ControlAR 训练环境支持 Python 3.10–3.12。新环境的 PyTorch 2.7.1 / torchvision 0.22.1 配对依据 [PyTorch 官方版本表](https://pytorch.org/get-started/previous-versions/)。

首次接入的已验证环境记录：Python 3.11.14、Torch `2.11.0.dev20260124+cu128`、torchvision `0.25.0.dev20260124+cu128`、CUDA runtime 12.8、包含 `sm_120`；RTX PRO 6000 Blackwell，彼时驱动 580.173.02 / 系统 CUDA 13.0。依赖 import、CUDA 可用性和 4×4 矩阵计算曾通过。这是历史验收记录，不代表新服务器或未来环境自动具有相同版本。

`setup_csgo_seen10.sh` 最后调用 `scripts/download_csgo_seen10_assets.py`：默认四个可恢复 HTTP Range worker，按官方固定 revision 下载，合并后核对官方 SHA256 才接受权重。所有新训练从官方原始 Canny-MR 初始化，不从已训练的 CSGO checkpoint 初始化。

| 资产 | 官方固定 revision | 大小（bytes） | SHA256 | 项目路径 |
| --- | --- | ---: | --- | --- |
| ControlAR Canny MR | `wondervictor/ControlAR@22cecd7a873db8df97ae2b2dc88befee72e97a3a` | 3,356,608,032 | `ef59b3c51e582e4742406480fb81160044b902bd46b2f00d923734800258545e` | `checkpoints/t2i/canny_MR.safetensors` |
| LlamaGen T2I VQ-16 | `peizesun/llamagen_t2i@276f5c5a3d915b922899a03f1912605531574747` | 287,920,306 | `0e21fc1318e2e9ee641a07bdad0e20675e9ec35e6e3eb911d58b5d7a2cd8d4cb` | `checkpoints/vq/vq_ds16_t2i.pt` |
| DINOv2-small | `facebook/dinov2-small@ed25f3a31f01632728cabb09d1542f84ab7b0056` | 88,249,960 | `ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1` | `autoregressive/models/dinov2-small/model.safetensors` |

DINO 同 revision 的 `config.json`、`preprocessor_config.json` Git blob ID 分别为 `5664b325e6258d3960fad8c4c1cff958f3cc2272`、`ff5b47c2edcd1d3556d63c01a65d93b58b9efce1`。首次准备时三个权重及两个 JSON 均通过校验。新 PEFT 若冻结视觉 encoder，应冻结 Canny-MR 加载后的视觉权重，不能另换成原始 DINO 权重。

### 4.2 数据、共享评测器与路径覆盖

建议保持同级布局；只需完整 Benchmark v2 数据 bundle，不要求安装 UniLIP 模型代码：

```text
workspace/
  ControlAR/
  UniLIP/data/csgo_benchmark_v2/
  csgo_benchmark_v2_eval_general/
```

数据与评测器目录的路径优先级是 **CLI → 环境变量 → 配置中的可用旧默认路径 → checkout 相对默认值**；评测 Python 使用下述单独优先级。自定义或显式给出的错误路径不会被自动替换。相对路径都以 ControlAR 根目录为基准，与调用 shell 的当前目录无关。三个 canonical JSON 保持原字节，机器路径在运行时解析，不修改实验含义。

| 用途 | 默认 | 环境变量 | runner CLI |
| --- | --- | --- | --- |
| 数据 | 原配置旧路径存在时保留；否则 `../UniLIP/data/csgo_benchmark_v2` | `CSGO_DATA_ROOT`，兼容 `CSGO_BENCHMARK_V2_DATA`、`DATA_ROOT` | `--data-root` |
| 共享评测器 | `../csgo_benchmark_v2_eval_general` | `SHARED_EVAL_DIR`，兼容 `CSGO_EVAL_ROOT` | `--eval-root` |
| 评测 Python | 所选共享评测器目录内 `.venv/bin/python` | 仅当评测器自身环境不可用时使用 `EVAL_PYTHON`，其次 `UNILIP_PYTHON`；均未设置时默认旧 UniLIP | `--eval-python`，兼容 `--unilip-python`，最高优先级 |
| 训练/推理 Python | `.venv/bin/python` | `CONTROLAR_PYTHON` | — |
| 标准库路径检查 Python | PATH 中的 python3/python | `CONTROLAR_PATHS_PYTHON` | — |
| GPU 进程数 | 1；原 aligned 固定单卡，PEFT 保持有效 batch128 | `NPROC_PER_NODE` | — |

评测解释器的准确优先级为：

1. CLI `--eval-python` / `--unilip-python`。
2. **最终选定的共享评测器目录**下 `.venv/bin/python`（跟随 `--eval-root` / `SHARED_EVAL_DIR` / `CSGO_EVAL_ROOT`）。
3. 显式设置的 `EVAL_PYTHON`，兼容 `UNILIP_PYTHON`；这两个变量不会覆盖已就绪的评测器自身环境。
4. 两个变量均未设置时，默认 `/home/jiahao/miniconda3/envs/UniLIP/bin/python`。

运行评测时，所选解释器必须是可执行文件。CLI 或显式环境变量指定的解释器无效时明确报错，不静默改用其他解释器；共享评测器自身环境缺失、断链或不可执行时才尝试后续候选。候选均不可用则 `eval` / `smoke` 报错，不回退到 `ControlAR/.venv-eval` 或训练 `.venv`，也不自动安装环境。训练和推理本身不依赖评测环境就绪。

此前 `setup_csgo_seen10.sh --eval-only` 保留为历史兼容选项，仍创建 `ControlAR/.venv-eval`，但它不再被自动选择。若确实要用该环境，应显式传 `--eval-python .venv-eval/bin/python`。默认流程改为共享评测器独立准备环境。

多个别名同时设置时按表内从左到右优先；清理不再使用的环境变量。官方 GPT/VQ/DINO 仍放在项目内的固定相对位置；整个 checkout 可换位置，单独数据目录则用显式覆盖。新服务器本地生成的模型和结果沿用相同训练、推理、评测命令。

```bash
# 非默认布局时，只需设置本机路径。变量应同时用于训练、推理和评测。
export CSGO_DATA_ROOT=/actual/path/to/csgo_benchmark_v2
export SHARED_EVAL_DIR=/actual/path/to/csgo_benchmark_v2_eval_general
# 可选：评测器自身 .venv 不可用时的后备解释器。
# export EVAL_PYTHON=/actual/path/to/eval-env/bin/python

# 无需模型环境或 GPU，只打印解析结果，不创建 run、不加载 checkpoint。
bash scripts/run_csgo_seen10.sh train \
  --experiment csgo_seen10_exp32gen_aligned_peft --print-paths
bash scripts/run_csgo_seen10.sh eval \
  --experiment csgo_seen10_exp32gen_aligned_peft --print-paths

# metadata、split/calibration、官方权重哈希与实验配置检查；不启动训练。
./.venv/bin/python scripts/validate_csgo_seen10_peft.py
```

共享 evaluator 必须另行同步完整目录及 `benchmark_v2.yaml`；没有回退到项目内旧指标实现。评测预训练资产与 ControlAR 模型权重是两组资产：FID/LPIPS/VGG 使用 Torch Hub 缓存（`TORCH_HOME` 或默认 `~/.cache/torch`），FVD 的 I3D 使用 `UNILIP_FVD_CACHE_DIR`（未设置时 evaluator 相对工作目录下 `loaded_models`）。离线服务器还需提前准备这些 metric 权重；有网络时首次完整评测可能下载。模型下载脚本支持 `HF_ENDPOINT`，始终保留官方固定 revision/SHA 校验。

### 4.3 恢复边界

本次迁移支持在另一服务器从官方 base **重新开始同配方实验**，随后在那台服务器训练、推理和评测。旧 checkpoint/预测 manifest 中的绝对数据路径及完整身份校验没有放宽；直接复制旧 run 到另一数据路径不等于支持精确续训或续写，默认会拒绝身份不匹配。不要手工改旧 checkpoint 或 manifest。

本次路径接入修改了训练入口源码，因此历史同路径 checkpoint 的源码 SHA 与当前版本不同。默认恢复仍严格拒绝；仅对本次明确登记的旧版本，可在原数据路径、配置、权重等全部一致时，在原 `train ... --resume <checkpoint>` 命令末尾追加 `--allow-legacy-source-resume`。该选项检查已登记的旧/新源码完整 SHA 集合并写入兼容审计，新 checkpoint 保存当前真实源码身份；它不能跨路径迁移，也不能跳过任意代码变化。无需该选项的新版本同路径恢复沿用第 5 节命令。原运行中的训练进程不会被停止或重启。

本轮迁移实现与只读验收记录见 [迁移验收报告](outputs/csgo_seen10_portability_checks/20260925_005013/ACCEPTANCE.md)。该历史验收通过 CPU 导入与资产检查，当时未安装项目 `.venv-eval`。当前默认改用共享评测器自身 `.venv`；环境由共享仓库独立管理，完整硬件与 GPU smoke 仍由目标服务器执行。

## 5. 直接执行命令

以下是手动操作说明，不表示本次文档整理执行这些命令。已有 legacy 结果和正在运行的 aligned 不应重复启动或覆盖；新实验必须隔离目录。正式 aligned 固定 seed 42 和 canonical run root，runner 不接受另一个正式 `--run-root`。

### 5.1 legacy 训练、恢复、默认推理与评测

```bash
# 历史 seed 0 已完成；这是复现入口，不要对已有运行重复启动。
bash scripts/run_csgo_seen10.sh train --seed 0

# 仅在任务已停止且需要恢复时使用；保持原 seed/world size/batch 和数据。
bash scripts/run_csgo_seen10.sh train --seed 0 \
  --resume outputs/csgo_benchmark_v2_seen10/ControlAR/seed_0/checkpoints/last.pt

# 默认 eager batch=1，读取 legacy best，输出到 legacy 默认目录。
bash scripts/run_csgo_seen10.sh infer --seed 0 --task all
bash scripts/run_csgo_seen10.sh eval --seed 0 --task all
```

legacy 如需重新训练，选择未使用的 seed/目录，不覆盖 seed 0 的结果。不传 `--experiment` 的命令继续保留旧语义。

### 5.2 legacy compiled batch=16

明确指定 legacy checkpoint 和独立预测目录；这个目录不需要复制 checkpoint：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=4 \
  .venv/bin/python infer_seen10.py \
  --seed 0 \
  --data-root /home/jiahao/task/UniLIP/data/csgo_benchmark_v2 \
  --checkpoint /home/jiahao/task/ControlAR/outputs/csgo_benchmark_v2_seen10/ControlAR/seed_0/checkpoints/best.pt \
  --output-root /home/jiahao/task/ControlAR/outputs/csgo_benchmark_v2_seen10_compiled_b16/ControlAR/seed_0 \
  --task all --inference-engine compiled --batch-size 16

bash scripts/run_csgo_seen10.sh eval --seed 0 --task all \
  --run-root /home/jiahao/task/ControlAR/outputs/csgo_benchmark_v2_seen10_compiled_b16/ControlAR/seed_0
```

该示例路径已有完整评测结果；不要为重新评测删除原结果。`eval` 缺少 `--run-root` 时仍读取旧 eager 目录：即使 compiled 目录生成完成，也可能报告旧目录缺图。此前 `missing=2600` 的错误应结合实际 prediction root 检查，不把其他目录的图片混进去补数。

legacy compiled 按 manifest 固定分组，随机 seed 与 batch index、组内 sample IDs 绑定。中断后保持原命令恢复；某组缺图时重算该组，只写缺失图片，已有有效 JPEG 保留，尾批补齐项不写出。manifest 拒绝混合不兼容引擎/batch/checkpoint。不要改变 batch size 后续写同一个 legacy compiled 目录。

### 5.3 已实现 aligned 训练与恢复

```bash
CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 \
  bash scripts/run_csgo_seen10.sh train \
  --experiment csgo_seen10_exp32gen_aligned --seed 42

# 示例：仅当 step_007800.pt 已保存、任务已停止，且它是最新保存点时恢复。
bash scripts/run_csgo_seen10.sh train \
  --experiment csgo_seen10_exp32gen_aligned --seed 42 \
  --resume /home/jiahao/task/ControlAR/outputs/csgo_seen10_exp32gen_aligned/ControlAR/seed_42/checkpoints/step_007800.pt
```

原 aligned 的 2026-09-24 运行快照见第 8 节；新 PEFT 正式训练未启动。原 aligned 恢复要求 data/config/base-weight/code identity 一致，完整 optimizer/scheduler/scaler/RNG 等状态均需匹配；不要通过改 JSON、切换 LoRA 或改变 batch 来续跑原 checkpoint。

### 5.4 aligned late/best 推理与评测

```bash
# 主结果：训练结束的同一个 late checkpoint 生成两个任务。
bash scripts/run_csgo_seen10.sh infer \
  --experiment csgo_seen10_exp32gen_aligned --checkpoint-role late \
  --seed 42 --inference-seed 42 --task all
bash scripts/run_csgo_seen10.sh eval \
  --experiment csgo_seen10_exp32gen_aligned --checkpoint-role late \
  --seed 42 --inference-seed 42 --task all

# 补充结果：best 独立目录，不与 late 混用。
bash scripts/run_csgo_seen10.sh infer \
  --experiment csgo_seen10_exp32gen_aligned --checkpoint-role best \
  --seed 42 --inference-seed 42 --task all
bash scripts/run_csgo_seen10.sh eval \
  --experiment csgo_seen10_exp32gen_aligned --checkpoint-role best \
  --seed 42 --inference-seed 42 --task all
```

需要分开执行时，将 `--task all` 改为 `--task discrete` 或 `--task continuous`，role 和两个 seed 保持相同。推理固定 compiled batch=16、BF16、CFG 4.0、temperature 1.0、top-k 2000、top-p 1.0、官方 VQ。这是 AR 逐 token 采样，生成 784 个 VQ token，不使用 diffusion timestep/time shift；不能把它当作 20-NFE diffusion 配置。

aligned 使用 `inference_seed + sample_id + token index` 派生的随机流；与 legacy 的 batch seed 规则不同。随机流不随分组/恢复改变，但不据此承诺不同 CUDA 实现的 logits 逐位一致；正式入口仍固定 batch=16。训练/推理配置、checkpoint SHA、任务身份和数据合同不匹配时拒绝续写；已有文件还需通过解码、RGB、尺寸与身份检查。

### 5.5 aligned PEFT：手动训练、恢复、推理和评测

以下正式命令供小批量验收通过后的**用户手动执行**；当前没有启动 PEFT 正式训练。训练从官方 Canny-MR 初始化，不读取原 aligned 的 checkpoint。默认单卡 micro8×累计16；可覆盖分解方式，但 `world_size × micro_batch × accumulation` 必须等于 128。`train_seen10_peft.py`、`csgo_seen10/peft.py`、`csgo_seen10/peft_artifact_contract.py`、`csgo_seen10/peft_compiled_inference.py` 和 `scripts/validate_csgo_seen10_peft.py` 负责独立 PEFT 行为。`csgo_seen10/compiled_inference.py` 增加 PEFT 专用 opt-in 分支，legacy 和原 aligned 的默认分支未变。此前 PEFT 验收时，现有 aligned 恢复所绑定的八个旧文件（`train_seen10.py`、`csgo_seen10/model.py`、`csgo_seen10/data.py`、`csgo_seen10/artifact_contract.py`、`autoregressive/models/gpt_t2i.py`、`autoregressive/train/train_c2i.py`、`configs/csgo_seen10.json`、`configs/csgo_seen10_exp32gen_aligned.json`）的 SHA 均保持不变，证据见第 8 节；之后本轮路径迁移的源码兼容边界见第 4.3 节。

```bash
# 默认：单卡 micro 8 × 累计 16 = 有效 batch 128。
CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 \
  bash scripts/run_csgo_seen10.sh train \
  --experiment csgo_seen10_exp32gen_aligned_peft --seed 42

# 可选组合示例：单卡 micro 16 × 累计 8；仅用于新 run 的首次启动。
CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 \
  bash scripts/run_csgo_seen10.sh train \
  --experiment csgo_seen10_exp32gen_aligned_peft --seed 42 \
  --batch-size 16 --gradient-accumulation-steps 8

# 示例：已有同一 run 的 step_007800.pt 且它是最新里程碑时，原组合恢复。
CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 \
  bash scripts/run_csgo_seen10.sh train \
  --experiment csgo_seen10_exp32gen_aligned_peft --seed 42 \
  --resume /home/jiahao/task/ControlAR/outputs/csgo_seen10_exp32gen_aligned_peft/ControlAR/seed_42/checkpoints/step_007800.pt
```

上面的两种首次启动组合是互斥示例，不能在同一 run 中切换。恢复必须使用该 run 最新保存的 `step_*.pt`，并保持训练时的 micro/累计、数据、配置、LoRA 代码和完整 optimizer/scheduler/scaler/RNG/sampler 身份一致。五个正式验证保存点仍为 3,900、7,800、11,700、15,600、19,500；step 19,500 的 `late.pt` 是主报告，验证最优的 `best.pt` 是补充报告。

```bash
# 主结果：同一个 PEFT late checkpoint 生成并评测离散、连续两个任务。
bash scripts/run_csgo_seen10.sh infer \
  --experiment csgo_seen10_exp32gen_aligned_peft --checkpoint-role late \
  --seed 42 --inference-seed 42 --task all
bash scripts/run_csgo_seen10.sh eval \
  --experiment csgo_seen10_exp32gen_aligned_peft --checkpoint-role late \
  --seed 42 --inference-seed 42 --task all

# 补充结果：best 使用独立预测与评测目录。
bash scripts/run_csgo_seen10.sh infer \
  --experiment csgo_seen10_exp32gen_aligned_peft --checkpoint-role best \
  --seed 42 --inference-seed 42 --task all
bash scripts/run_csgo_seen10.sh eval \
  --experiment csgo_seen10_exp32gen_aligned_peft --checkpoint-role best \
  --seed 42 --inference-seed 42 --task all
```

PEFT 推理严格加载未 merge checkpoint，在临时 FP32 模型合并 LoRA 后转为 BF16，沿用 compiled batch16 的 sample/token 随机流。PEFT 的 `compiled_logits+cuda_aten_fp32_inverse_cdf` 分界只编译 Transformer、CFG 与 top-k logits；softmax、FP32 cumsum 和逆 CDF 采样由未编译的 CUDA ATen 执行，避开当前 Inductor 对 `[16,16384]` fused scan 的代码生成错误。预测分别写入 PEFT run 的 `predictions/{late,best}/inference_seed_42/`，评测写入 `evaluation/{late,best}/inference_seed_42/`；跨 experiment、role、checkpoint SHA 或样本身份的续写会被拒绝。评测前还要通过独立 PEFT preflight 和共享 evaluator 的完整覆盖检查。

## 6. 输出、共享评测与恢复隔离

legacy 默认 run 内包含 `train.log`、`run_config.json`、`checkpoints/`、`discrete/gen_imgs/`、`continuous/gen_imgs/`、`inference_manifest.json` 和 `evaluation/{discrete,continuous}/`。独立 compiled run 只保存对应推理、manifest 和评测产物，训练 checkpoint 仍在原训练 run。

aligned 目录结构如下，未到对应阶段时相关文件不存在：

```text
outputs/csgo_seen10_exp32gen_aligned/ControlAR/seed_42/
├── train.log
├── run_config.json
├── audits/{identity.json,trainable_parameters.json}
├── checkpoints/
│   ├── step_{003900,007800,011700,015600,019500}.pt
│   ├── best.pt
│   ├── late.pt
│   └── checkpoint_index.json
├── predictions/{best,late}/inference_seed_42/
│   ├── discrete/gen_imgs/<map>/<file_frame>.jpg
│   └── continuous/gen_imgs/<map>/<file_frame>.jpg
└── evaluation/{best,late}/inference_seed_42/{discrete,continuous}/
```

PEFT 使用同样的相对结构，但根目录为 `outputs/csgo_seen10_exp32gen_aligned_peft/ControlAR/seed_42/`，checkpoint、预测、评测、日志和审计均与原 aligned 分离。推理还写入各自的 manifest/completion 等记录用于校验；元数据完成标记不能代替实际文件覆盖检查。

唯一正式指标入口为 `/home/jiahao/task/csgo_benchmark_v2_eval_general/run_eval.py` 和同目录 `benchmark_v2.yaml`。本项目旧 `csgo_benchmark_v2_eval/` 副本保留为历史；runner 不回退到它。读取数据合同的历史依赖不代表使用旧副本计算正式指标。指标实现不复制进模型代码。

| 任务 | 正式指标 | 聚合与时序协议 |
| --- | --- | --- |
| discrete | PSNR ↑、SSIM ↑、LPIPS ↓、Boundary_F1 ↑、FID ↓ | 先逐地图，再 equal-map macro |
| continuous | PSNR ↑、SSIM ↑、LPIPS ↓、TWE ↓、TDE ↓、FVD ↓ | equal-map macro；FVD 窗口 16 帧（clip_length=16）、stride 16、FVD size 224 |

共享配置还固定 frame-difference threshold=2、min_track_len=4、每地图 20 clips × 64 frames；Boundary_F1 的 edge quantile=0.85、tolerance=2 pixels。不要自行修改指标参数。FID/FVD 的跨地图 pooled 分数不能替代逐图计算后取宏平均。

正式评测要求精确完整覆盖，缺图或多余图片都拒绝。原 aligned 和 PEFT 各自的 runner preflight 在评测前检查 experiment、role、checkpoint/config SHA、checkpoint index、当前数据 manifest/split/calibration 合同，并重查图片是否可解码、448×448 RGB JPEG；通过后才调用共享 evaluator。正式输出目录已有结果时 evaluator 拒绝覆盖。

## 7. 验收和 smoke

不加载大模型的 aligned 静态检查：

```bash
./.venv/bin/python scripts/validate_csgo_seen10_aligned.py

# 训练完成后才检查 late 和五个完整 checkpoint 的 SHA；会产生较多磁盘读取。
./.venv/bin/python scripts/validate_csgo_seen10_aligned.py \
  --run-root /home/jiahao/task/ControlAR/outputs/csgo_seen10_exp32gen_aligned/ControlAR/seed_42 \
  --checkpoint-role late --verify-all-checkpoint-sha256
```

脚本检查 JSON、数据 identity、样本数、路径、代码约束和 checkpoint index，不导入模型、不加载 `.pt` tensor。静态通过不等于真实 forward/backward、GPU 峰值或精确恢复已验收。

PEFT 使用独立静态入口；下面命令只检查配置/数据与 batch 规则，不启动训练。实际训练目录出现后才传 `--run-root` 检查审计和 checkpoint。默认是 8×16，覆盖示例 16×8 仍为有效 batch 128。

```bash
./.venv/bin/python scripts/validate_csgo_seen10_peft.py
./.venv/bin/python scripts/validate_csgo_seen10_peft.py \
  --batch-size 16 --gradient-accumulation-steps 8
```

以下 smoke **会运行小量训练/推理/评测**，仅在安排好资源且获准测试时手动使用；目录必须是新的：

```bash
CSGO_SMOKE_ROOT="$PWD/outputs/csgo_benchmark_v2_smoke/ControlAR/seed_0_rerun1" \
  bash scripts/run_csgo_seen10.sh smoke --seed 0

CSGO_ALIGNED_SMOKE_ROOT="$PWD/outputs/csgo_seen10_exp32gen_aligned_smoke/ControlAR/seed_42_rerun1" \
  bash scripts/run_csgo_seen10.sh smoke --experiment csgo_seen10_exp32gen_aligned --seed 42
```

wrapper smoke 覆盖少量训练/验证、保存和离散生成/paired metrics，不等于完整连续时序指标验收。legacy 连续单帧补充检查的历史命令如下；smoke checkpoint 须先存在，推理输出采用另一个新目录：

```bash
CSGO_SMOKE_ROOT="$PWD/outputs/csgo_benchmark_v2_smoke/ControlAR/seed_0_continuous_rerun1"
SHARED_EVAL_DIR=/home/jiahao/task/csgo_benchmark_v2_eval_general
UNILIP_PYTHON=/home/jiahao/miniconda3/envs/UniLIP/bin/python
PYTHONDONTWRITEBYTECODE=1 ./.venv/bin/python infer_seen10.py --seed 0 --task continuous --smoke \
  --checkpoint outputs/csgo_benchmark_v2_smoke/ControlAR/seed_0/checkpoints/best.pt \
  --output-root "$CSGO_SMOKE_ROOT"
PYTHONDONTWRITEBYTECODE=1 "$UNILIP_PYTHON" "$SHARED_EVAL_DIR/run_eval.py" \
  smoke continuous --pred-root "$CSGO_SMOKE_ROOT" \
  --data-root /home/jiahao/task/UniLIP/data/csgo_benchmark_v2 --frame-only
```

历史 legacy smoke 证据为 `outputs/csgo_benchmark_v2_smoke/ControlAR/seed_0/smoke_acceptance.json`：step 1 train loss 9.361092，pose/map 与 radar adapter 梯度范数 4.145674 / 30.503811，validation loss 6.99104166 两次一致；离散/连续单图分别约 17.1/18.9 秒。保存重载、step 1 状态恢复及已有图片跳过均有历史检查记录。frame-only 只计算 paired metrics，不计算 temporal/FVD；这些 smoke 数字不是正式 benchmark 结果。

## 8. 已有结果、速度与执行状态

以下为 **2026-09-24 文档整理时的只读快照**，不是持续监控，也不代表本次启动了任务。实际新进度以日志和 checkpoint 为准。

- legacy seed 0：`outputs/csgo_benchmark_v2_seen10/ControlAR/seed_0/train.log` 记录训练于 2026-09-16 完成 300,000 updates；best validation loss 6.952757，发生在 step 180,000；last 为 step 300,000。
- legacy 旧 eager 目录：本次只读计数为离散 17,400/20,000 张 JPEG，尚无 continuous 或 evaluation 目录。它与已完成的 compiled 目录不同，不能直接用默认 eval 命令评测 compiled 结果。
- legacy compiled b16：`inference_manifest.json` 指向 legacy `best.pt`，seed 0；discrete/continuous 的共享 evaluator 汇总均记录 `coverage_complete=true`，分别 20,000/12,800 张。这不是 aligned 结果。
- aligned seed 42：读取到主进程 PID 2705672；日志在 2026-09-24 23:36:05 记录 step 2960、epoch 8、曝光量 378,880。`run_config.json` 确认 micro=1、accumulation=128、effective batch=128；尚未到首个 step 3900 保存点，检查时 checkpoint 目录为空。当前阶段无 aligned 正式测试结果。
- PEFT（2026-09-25 小批量验收，证据根 `outputs/csgo_seen10_exp32gen_aligned_peft_smoke/acceptance_20260925_000701`）：默认单卡 micro8×累计16 完成 2 个 optimizer updates，源样本曝光 256；峰值 allocated 26.64 GiB、reserved 29.88 GiB。逐参数审计核对 866,921,344 个模型参数（不含 VQ）、69,316,608 个 trainable/optimizer 参数；480 个冻结参数张量与官方权重逐位一致。八个旧源码/配置文件 SHA 保持不变，详见 `original_source_preservation.json`。这两步的耗时仅是短测，不能外推正式训练 ETA。
- PEFT 恢复验收：从同一 step 1 checkpoint 分叉，在测试专用 deterministic math/SDPA 设置下，step 2 的模型、optimizer、scheduler、scaler、RNG、sampler 等状态逐位一致，见 `deterministic_resume_comparison.json`。默认高性能 GPU 后端下恢复的下一步 forward loss 完全相同，但 backward 非确定性导致梯度范数有微小差异；不宣称默认后端逐位恢复。离散 64 张、连续 1 clip×64 帧的 compiled 推理、输出合同与共享 evaluator smoke 均通过；重复启动生成 0 张，128 张已有图片的 SHA、大小与 mtime 均不变。共享 smoke 覆盖连续 TWE/TDE，按协议跳过 FID/FVD；正式 19,500-step 训练及全量推理/评测未启动。完整证据与限制见 [PEFT 验收报告](outputs/csgo_seen10_exp32gen_aligned_peft_smoke/acceptance_20260925_000701/ACCEPTANCE.md)。

legacy compiled b16 的 equal-map macro 结果如下，源文件为对应 run 的 `evaluation/discrete/summary_equal_map.json` 和 `evaluation/continuous/summary_equal_map.json`：

| 任务 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Boundary_F1 ↑ | FID ↓ | TWE ↓ | TDE ↓ | FVD ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| discrete | 13.640973 | 0.414244 | 0.654617 | 0.504800 | 84.641917 | — | — | — |
| continuous | 14.041270 | 0.413868 | 0.660458 | — | — | 30.065167 | 37.052706 | 1153.235919 |

这组结果使用较少曝光量和 validation-best，不能冒充与 exp32_gen 同预算、同 final 选点的主比较。尚未得到 aligned/PEFT 正式结果，不据此判断后续方法优劣。

共享 GPU 的历史 32 张微基准记录在 `outputs/inference_speed_study/REPORT.md`：compiled b16 离散约 1.03 秒/张、连续约 1.02 秒/张，reserved 峰值约 8.92 GiB，首次预热约 52 秒；全量线性外推约 9.34 小时。用户随后提供的完整生成日志为离散 25,909.5 秒（7.20 小时）、连续 13,505.8 秒（3.75 小时），合计约 **10.95 小时**，不含后续评测。共享负载改变时外推会变化，推理显存也不能用于推断训练 batch 上限。

历史 `RUN_FULL=0` / `RUN_FORMAL=0` 描述的是当时接入阶段的执行边界，不应覆盖后来用户手动启动的事实。PEFT 已完成本轮授权的小批量验收；正式训练由用户在手动验收后执行。
