# ControlAR CSGO Benchmark v2 Seen-10

ControlAR 接入 `GENERATION`，生成 discrete 与 continuous 图像。本机项目环境为 `.venv`（Python 3.11、Torch 2.11 cu128）；官方 Canny-MR、VQ-16、DINO-small 权重及 DINO 配置已下载并通过固定版本校验。共享评测器目录为 `/home/jiahao/task/csgo_benchmark_v2_eval_general/`，使用 `/home/jiahao/miniconda3/envs/UniLIP/bin/python`。环境与资产详情见 [CSGO_SEEN10_ENV.md](CSGO_SEEN10_ENV.md)。新环境首次准备运行 `./scripts/setup_csgo_seen10.sh`。

默认正式训练共 300,000 steps，每 60,000 steps 验证并保存一次，完整周期恰好得到 5 次验证和 5 个等距 checkpoint：`step_060000.pt`、`step_120000.pt`、`step_180000.pt`、`step_240000.pt`、`step_300000.pt`。`best.pt` 保存验证最优权重；`last.pt` 指向最近的完整训练状态。里程碑状态只序列化一次，`last.pt` 使用同文件系统链接，不重复占用 checkpoint 大小。

本次 `RUN_FULL=0`，只运行真实 GPU smoke，不启动全量训练或正式评测。离散与连续单帧 smoke 均已通过。离散 wrapper exit 0：step 1 train loss `9.361092`，pose/map 与 radar adapter 梯度范数 `4.145674` / `30.503811`；validation loss `6.99104166` 连续两次一致，`best.pt` 与训练状态 reload 成功；同一 best checkpoint 在 17.1 秒内生成 `cs_agency/file_num68_frame_421.jpg`（448×448 RGB）。UniLIP evaluator 读取 1/1 张图并计算 paired metrics，报告 `smoke_only=true`、`formal=false`、`official_output_written=false`。连续单帧推理 exit 0，在 18.9 秒内生成 `cs_agency/file_num3_frame_205.jpg`；`smoke continuous --frame-only` evaluator 读取 1/1 张图并计算 PSNR/SSIM/LPIPS，按 frame-only 模式跳过 temporal/FVD。Smoke 不是正式 benchmark 结果；当前无阻塞项。

完整验收记录见 `outputs/csgo_benchmark_v2_smoke/ControlAR/seed_0/smoke_acceptance.json`。另实测从 smoke 的训练状态（step 1）恢复 optimizer/scaler/RNG 后未多跑一步；`infer --smoke --task all` 重跑 exit 0，discrete 与 continuous 各 `generated=0`、`existing=1`，核对文件 mtime 与 size 均未变化。

## 命令

```bash
# 单 batch GPU smoke：训练/验证、checkpoint reload、单图生成、smoke evaluator
bash scripts/run_csgo_seen10.sh smoke --seed 0

# Continuous 单帧 smoke：需要重新生成时换新输出根目录；原目录重跑会跳过已有图片
CSGO_SMOKE_ROOT="$PWD/outputs/csgo_benchmark_v2_smoke/ControlAR/seed_0_continuous_rerun1"
SHARED_EVAL_DIR=/home/jiahao/task/csgo_benchmark_v2_eval_general
UNILIP_PYTHON=/home/jiahao/miniconda3/envs/UniLIP/bin/python
PYTHONDONTWRITEBYTECODE=1 ./.venv/bin/python infer_seen10.py --seed 0 --task continuous --smoke \
  --checkpoint outputs/csgo_benchmark_v2_smoke/ControlAR/seed_0/checkpoints/best.pt \
  --output-root "$CSGO_SMOKE_ROOT"
PYTHONDONTWRITEBYTECODE=1 "$UNILIP_PYTHON" "$SHARED_EVAL_DIR/run_eval.py" \
  smoke continuous --pred-root "$CSGO_SMOKE_ROOT" \
  --data-root /home/jiahao/task/UniLIP/data/csgo_benchmark_v2 --frame-only

# 正式 seed 0 流程（本次 RUN_FULL=0，不执行）
bash scripts/run_csgo_seen10.sh train --seed 0
bash scripts/run_csgo_seen10.sh infer --seed 0 --task all
bash scripts/run_csgo_seen10.sh eval --seed 0 --task discrete
bash scripts/run_csgo_seen10.sh eval --seed 0 --task continuous
```

## 编译批量推理

`infer_seen10.py` 默认继续使用原来的 eager batch=1 路径，因此上述正式推理命令和已经存在的
输出目录都保持兼容。编译路径必须显式指定，并写入独立的输出根目录：

```bash
cd /home/jiahao/task/ControlAR
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=4 \
  .venv/bin/python infer_seen10.py \
  --seed 0 \
  --data-root /home/jiahao/task/UniLIP/data/csgo_benchmark_v2 \
  --checkpoint /home/jiahao/task/ControlAR/outputs/csgo_benchmark_v2_seen10/ControlAR/seed_0/checkpoints/best.pt \
  --output-root /home/jiahao/task/ControlAR/outputs/csgo_benchmark_v2_seen10_compiled_b16/ControlAR/seed_0 \
  --task all \
  --inference-engine compiled \
  --batch-size 16
```

中断后使用同一条命令恢复。编译路径按 manifest 顺序固定分成 16 张一组；只要某组有缺图，
就用同一批次 seed 重算整组并只写入缺失图片。有效的已有 JPEG 会校验并保留，尾批会补齐到
16 张但不写出补齐项。输出 manifest 会记录推理引擎、batch size、编译模式和 seed 策略，并
拒绝将 compiled 结果混入旧 eager 目录。

编译推理完成后，用 `--run-root` 将正式评测指向同一个独立输出根目录。省略此参数时，`eval`
仍默认读取原来的 `outputs/csgo_benchmark_v2_seen10/ControlAR/seed_<seed>/`：

```bash
bash scripts/run_csgo_seen10.sh eval --seed 0 --task all \
  --run-root /home/jiahao/task/ControlAR/outputs/csgo_benchmark_v2_seen10_compiled_b16/ControlAR/seed_0
```

评测预测从该根目录的 `discrete/gen_imgs/` 和 `continuous/gen_imgs/` 读取，正式指标写入其
`evaluation/` 子目录。

共享 GPU 上固定 32 张样本的实测为离散约 1.03 秒/张、连续约 1.02 秒/张；完整 32,800 张
线性外推约 9.34 小时，进程峰值 reserved 显存约 8.92 GiB，首次编译预热约 52 秒。原始结果
见 `outputs/inference_speed_study/REPORT.md`。本次只完成实现和静态验收，不自动启动该命令。

Smoke 输出默认在 `outputs/csgo_benchmark_v2_smoke/ControlAR/seed_<seed>/`，与正式结果隔离。Smoke 不覆盖已有目录；重复运行时指定新的根目录：

```bash
CSGO_SMOKE_ROOT="$PWD/outputs/csgo_benchmark_v2_smoke/ControlAR/seed_0_rerun1" \
  bash scripts/run_csgo_seen10.sh smoke --seed 0
```

## 输出与恢复

正式输出位于 `outputs/csgo_benchmark_v2_seen10/ControlAR/seed_<seed>/`，包含 `checkpoints/{best.pt,last.pt,step_*.pt}`、`discrete/gen_imgs/<map>/<frame>.jpg`、`continuous/gen_imgs/<map>/<frame>.jpg` 和 `evaluation/{discrete,continuous}/` 下的 per-map 与 equal-map 结果。

从最近训练状态续跑使用 `last.pt`，并保持原 seed、world size、batch size 与数据行数：

```bash
bash scripts/run_csgo_seen10.sh train --seed 0 \
  --resume outputs/csgo_benchmark_v2_seen10/ControlAR/seed_0/checkpoints/last.pt
```

`infer` 默认读取 validation 选出的 `best.pt`。中断后对同一 seed/task 重跑 infer，会校验并跳过有效图片，只补生成缺图；`eval` 默认使用共享评测器与 UniLIP Python，并拒绝覆写已有正式评测目录。
