# ControlAR Seen-10：设计决策与 aligned PEFT 实现

本文与 [CSGO_SEEN10.md](CSGO_SEEN10.md) 分工：主文档记录已实现的运行行为、命令、环境和结果；本文记录方案边界、模块职能、后续变更及验收。参考 X-VLA `CSGO_SEEN10*.md` 的内容范围与实验隔离方式，不把其定位 action head、10 步求解或阶段学习率复制到 ControlAR。

**截至 2026-09-25：`csgo_seen10_exp32gen_aligned_peft` 的独立配置、训练、推理和静态验收入口已实现；小批量 GPU 验收已完成，等待用户手动验收。** 正式 19,500-step 训练和全量生成均未启动，须在手动验收后由用户执行。原 aligned 运行和配置独立保留；其 2026-09-24 状态快照见主文档。

## 1. 已实现方案及保留边界

- 任务为 Seen-10 radar/map + 当前 pose → FPV，主对照 UniLIP `exp32_gen`，不增加定位、aux_loc_loss、perception_loss 或其他 UniLIP 创新模块。
- 复用官方 GPT-XL / Canny-MR、DINOv2-small control adapter 和 VQ-16；radar 直接作为 control image，新建数值 pose/map embedding 接入原生 caption 条件路径，不运行 T5。
- `csgo_seen10/data.py` 从发布 manifest/split/calibration 读取样本身份、pose 和 clip/frame 次序；推理不读取 target FPV。448 radar / 448 FPV、原生 AR CE、确定性图像处理是已选定的模型实现。
- legacy 保留首次接入行为；`csgo_seen10_exp32gen_aligned` 增加独立 profile、有效 batch 128、19,500 optimizer updates、2,496,000 曝光量、五次完整验证和 late/final 主报告规则。
- aligned 已实现训练状态/identity 审计、独立 best/late 预测、stateless sample/token 随机流、评测前产物检查；正式指标只调用共享 evaluator。
- 历史首次接入曾创建项目内 evaluator 副本；后续迁移后的唯一正式评测器是 `/home/jiahao/task/csgo_benchmark_v2_eval_general`。不继续复制或修改指标实现。

当前文件职责：

| 文件 | 已实现职责 |
| --- | --- |
| `configs/csgo_seen10.json` | legacy 配置，不改变含义 |
| `configs/csgo_seen10_exp32gen_aligned.json` | 全量训练 aligned 的 canonical 配置 |
| `configs/csgo_seen10_exp32gen_aligned_peft.json` | PEFT 独立 canonical 配置，默认 micro8×累计16 |
| `csgo_seen10/data.py` | manifest 驱动数据、输入隔离与确定性图像处理 |
| `csgo_seen10/model.py` | 数值 pose/map 条件与官方 GPT/VQ 适配 |
| `train_seen10.py` | legacy/aligned 训练、验证、预算、完整恢复与参数审计 |
| `train_seen10_peft.py`、`csgo_seen10/peft.py` | PEFT 独立训练/恢复、LoRA 注入与 merge、参数分组及审计 |
| `infer_seen10.py`、`csgo_seen10/compiled_inference.py`、`csgo_seen10/peft_compiled_inference.py` | 两任务推理；PEFT 的 logits 编译与 CUDA ATen 逆 CDF 分界 |
| `csgo_seen10/artifact_contract.py` | aligned 数据/checkpoint/预测合同与 preflight |
| `csgo_seen10/peft_artifact_contract.py` | PEFT checkpoint、预测和评测前身份合同 |
| `scripts/run_csgo_seen10.sh` | 各 profile 入口与共享 evaluator 调用 |
| `scripts/validate_csgo_seen10_aligned.py` | 不加载模型的 aligned 静态验收 |
| `scripts/validate_csgo_seen10_peft.py` | PEFT 独立静态验收与有效 batch 规则 |
| `scripts/setup_csgo_seen10.sh`、`scripts/download_csgo_seen10_assets.py` | 环境与固定 revision/SHA 权重准备 |

## 2. PEFT 实验的目标与不变项

使用独立名字 `csgo_seen10_exp32gen_aligned_peft`，正式输出到 `outputs/csgo_seen10_exp32gen_aligned_peft/ControlAR/seed_42`。从官方原始 Canny-MR 开始，不从当前 CSGO aligned 权重 warm-start，不停止、覆盖或续写现有 run。

参照 X-VLA aligned-frozen-vl 与 exp32_loc 的关系，比较的是可训练模块的**功能角色**：冻结预训练视觉入口，主要任务 Transformer 使用 LoRA，必要的任务转换模块全量训练。ControlAR 只有一套统一生成 GPT，没有单独的语言 LLM 和 generation DiT，不重复注入两套 LoRA，也不强行建立不存在的模块对应关系。

保持已实现 aligned 的数据、当前样本条件信息、448/448 尺寸、无随机图像增强、AR CE、VQ、有效 batch、总更新数、曝光量、完整验证点、best/late、seed、推理采样和共享评测协议。改变可训练范围及其 LR、weight decay、scheduler；因此与旧 aligned 的比较不是“仅改变 LoRA”单因素消融。

正式 LoRA 训练 recipe 未找到可直接复用的完整官方配置；官方论文有 LoRA 消融，但不能据此称下面的 rank/LR 是官方推荐或已验证稳定。LoRA 强度参考 exp32_gen，优化保留 ControlAR 已有的 AdamW betas、预训练层 LR/正则经验。小批量 GPU 验收确认两步训练 loss/梯度有限，不据此宣称长期稳定或收敛；未使用测试集调参。

## 3. 模块职能与可训练范围（已实现）

| 模块职能 | UniLIP exp32_gen | ControlAR 已实现 aligned | ControlAR PEFT 精确前缀与状态 |
| --- | --- | --- | --- |
| 视觉 encoder | 冻结 vision tower | 全量训练 | `gpt.adapter.model.*` 冻结 |
| 视觉进入主干前的 projector | 冻结 multimodal projector | 全量训练 | `gpt.adapter_mlp.*` 冻结 |
| 主生成网络 | LLM、生成 DiT 各自 LoRA | 统一 GPT 全量训练 | `gpt.layers.*` 的 attention/MLP base 冻结、注入一次 LoRA |
| 独立 generation connector | LoRA r16 | 无同构独立模块 | 不强行对应、不虚构第二个 Transformer connector |
| 新建数值 pose/map 条件 | 无一一对应模块 | 全量训练 | `pose_map_embedder.*` 全量训练 |
| 条件投影与控制注入 | 不强行一一对应；UniLIP 小型生成 projector/query 可训练 | 全量训练 | `gpt.cls_embedding.cap_proj.*`、`gpt.condition_mlp.cap_proj.*`、`gpt.condition_layers.*` 全量训练 |
| VQ token 输出线性层 | 与 flow head 无直接同构关系 | 全量训练 | `gpt.output.weight` 全量训练 |
| token embedding、GPT norms | 不强行对应 | 全量训练 | `gpt.tok_embeddings.*`、`gpt.norm.*`、各层 norm 冻结 |
| 未使用条件 embedding | 不活跃分支冻结 | 冻结 | `gpt.condition_embeddings.*` 冻结 |
| 图像 tokenizer/decoder | VAE 冻结 | VQ 冻结 | 独立 VQ encoder/codebook/decoder 全冻结 |

`gpt.adapter_mlp` 是这里冻结的视觉输入连接器；`condition_mlp` 和 `condition_layers` 属于生成侧控制注入，不应因为名称含 condition 就一起冻结。`gpt.output.weight` 是约 21M 参数的 token 分类线性层，不是 VQ 图像 decoder；本方案是 LoRA + 全量条件/输出模块，不是纯 LoRA，也不是只训练一个很小的 adapter。

冻结视觉权重的时点是加载官方 Canny-MR 之后，不另用 raw DINO 重置。UniLIP 的生成 connector r16 仅是其自身结构的设置，不为了名称相似将 ControlAR 的所有投影改成 r16 LoRA。

### 3.1 LoRA 注入点

36 个 block 的目标层为：

```text
gpt.layers.{0..35}.attention.wqkv
gpt.layers.{0..35}.attention.wo
gpt.layers.{0..35}.feed_forward.w1
gpt.layers.{0..35}.feed_forward.w3
gpt.layers.{0..35}.feed_forward.w2
```

统一 r=32、alpha=64、dropout=0.05、bias=none、标准 scaling=2；A 随机初始化、B 零初始化。保留 fused `wqkv` 原始 base 权重，但 Q/K/V 各使用独立 rank32 增量，按代码真实 `[dim, kv_size, kv_size]` 分块；不是整块 fused QKV 共享一个 rank32。其余线性层各一套 rank32。无需 QLoRA、DoRA 或 rsLoRA。

现有 resid/FFN/token/CFG dropout 0.1 保留；LoRA dropout 0.05 是新增的 LoRA 支路正则。推理在临时 FP32 模型中严格加载未 merge checkpoint，merge LoRA 后转为 BF16，使用 compiled batch16 路径。PEFT 的 `compiled_logits+cuda_aten_fp32_inverse_cdf` 只编译 Transformer、CFG、top-k logits，FP32 softmax/cumsum/逆 CDF 由未编译的 CUDA ATen 执行；原 aligned 与 legacy 默认路径不变。实际非零 LoRA 的 GPU 合并检查与小批量推理/共享评测 smoke 已通过，覆盖范围见第 7 节；可恢复训练 checkpoint 保持未 merge 结构。

### 3.2 参数量合同

下表为 GPT-XL 参数合同。GPU 小批量审计已核对模型总数 866,921,344、trainable 与 optimizer 收录均为 69,316,608（约 8.00%）；480 个冻结参数张量与官方权重逐位一致。分母均不包含独立冻结 VQ。证据见 `outputs/csgo_seen10_exp32gen_aligned_peft_smoke/acceptance_20260925_000701/frozen_parameter_verification.json` 和 `smoke_parameter_contract.json`。

| 部分 | 可训练参数合同 |
| --- | ---: |
| Q/K/V 独立 LoRA，36 层 | 8,847,360 |
| attention output LoRA，36 层 | 2,949,120 |
| MLP 三线性层 LoRA，36 层 | 16,809,984 |
| LoRA 合计 | 28,606,464 |
| pose/map embedder 全量 | 2,371,584 |
| cls caption projection 全量 | 4,259,840 |
| condition projection 全量 | 3,276,800 |
| 3 个 condition injection MLP 全量 | 9,830,400 |
| output token classifier 全量 | 20,971,520 |
| 全量小模块/输出层合计 | 40,710,144 |
| 总可训练 | **69,316,608** |

加入 LoRA 后模型参数为 866,921,344，可训练约 **8.00%**；旧 aligned 可训练 817,343,360。训练时逐参数 `audits/trainable_parameters.json` 与上述 GPU 汇总共同作为 optimizer 对账证据。

## 4. 优化器与学习率（已实现配置）

AdamW，支持时 fused，betas `(0.9, 0.95)`、epsilon 1e-8、gradient clipping 1.0，沿用 BF16 autocast 与现有 FP32 参数/optimizer 状态行为。

| 参数组 | peak LR | cosine 最低 LR | weight decay |
| --- | ---: | ---: | ---: |
| 所有 LoRA A/B | 1e-4 | 1e-5 | 0 |
| 新建 `pose_map_embedder.*` | 1e-4 | 1e-5 | 0 |
| 全量预训练 cls/condition projections、condition layers、output | 5e-5 | 5e-6 | 矩阵 0.05；bias/低维 0 |

warmup 为 195 个 optimizer updates（总预算 1%），随后 cosine 衰减到各组 peak 的 10%。scheduler 只按 optimizer step 更新，不按 microstep。所有指定 trainable 同时开始 warmup，不复制 X-VLA 前 1,000 steps 只训练 head 的阶段。

选择依据：预训练全量层维持 ControlAR 已有 5e-5 和矩阵正则；LoRA/新条件层使用 1e-4、无 weight decay，配合短 warmup 和长周期末端降 LR。相对原 aligned 的 constant LR，这是新 PEFT 实验的优化差异，不声称已证明更优。

## 5. micro batch、预算和显存

PEFT 默认配置为 **world_size=1、micro batch=8、gradient accumulation=16、effective batch=128**。8×16 是默认值，不是强制组合；原 aligned 配置和代码行为保持独立。

PEFT 正式训练的 batch 组合只校验：

```text
world_size × micro_batch_per_device × gradient_accumulation_steps = 128
```

三个计数必须为正整数；不要求 micro batch 固定为 8，也不要求 accumulation 固定为 16，不将实际值与配置默认值作硬性相等检查。单卡 1×128、4×32、8×16、16×8 等组合均应通过。显式传参可以覆盖默认值，但乘积不为 128 时必须拒绝。此规则仅放宽 batch 的分解方式，不放宽 19,500 updates、曝光量、验证点或数据协议。

原 aligned 分支及静态校验器仍硬性要求 1×128。PEFT 的独立训练入口和静态校验器允许 CLI 覆盖默认 micro/accumulation，按实际组合计算 epoch microstep 数，并以全局源样本数 49,920 和每 epoch 390 optimizer updates 校验。

| 单卡 micro batch | 累计次数 | 每 update 源样本 | PEFT 预计训练峰值显存 |
| ---: | ---: | ---: | --- |
| 1 | 128 | 128 | 7–11 GiB |
| 8（默认） | 16 | 128 | 20–30 GiB |
| 16（待实测） | 8 | 128 | 35–55 GiB |

上表是结构估算。实际单卡 micro8×累计16 的 2-update 小批量训练（含反向与 Adam 状态初始化）峰值 allocated **26.64 GiB**、reserved **29.88 GiB**，位于估算区间，证据见验收根下 `uninterrupted/train_summary.json`。该数据不能推断 micro16 的显存，也不能用 batch16 推理约 9 GiB 推断训练上限；共享负载余量仍需在正式启动前检查。

```text
micro8:  6,240 microsteps/epoch ÷ 16 = 390 updates/epoch
micro16: 3,120 microsteps/epoch ÷  8 = 390 updates/epoch
每 epoch 49,920 条，尾部 80 条丢弃
19,500 updates × 128 = 2,496,000 曝光 = 49.92 等价完整 epoch
```

不因 micro batch 增大提高 LR，不改变 19,500 updates 或五个验证点。不同 batch 的 dropout 和浮点计算顺序可能不同，预算一致不等于逐位同一训练轨迹。当前只有两步训练短测，不能据此承诺正式训练 ETA，也不能复用旧 aligned 或推理吞吐。

## 6. 已实施文件边界

| 文件 | 已实施内容 |
| --- | --- |
| `configs/csgo_seen10_exp32gen_aligned_peft.json` | 独立 profile、LoRA、分组 LR/scheduler、默认 micro8/accum16 |
| `csgo_seen10/peft.py` | fused QKV 独立增量、Linear LoRA、注入和推理 merge、参数角色审计 |
| `train_seen10_peft.py` | 独立训练、有效 batch=128 校验、动态 epoch microstep、完整里程碑恢复 |
| `infer_seen10.py` | PEFT checkpoint 严格加载、临时 FP32 merge、compiled b16 两任务推理 |
| `csgo_seen10/peft_compiled_inference.py`、`csgo_seen10/compiled_inference.py` | PEFT 专用 logits 编译入口与共享生成器 opt-in；原 legacy/aligned 默认分支不变 |
| `csgo_seen10/peft_artifact_contract.py` | PEFT checkpoint、best/late、预测身份和评测前检查 |
| `scripts/run_csgo_seen10.sh` | 显式 PEFT experiment 路由到独立训练与校验入口 |
| `scripts/validate_csgo_seen10_peft.py` | 独立静态验收；micro 与累计为正整数，仅按乘积检查有效 batch=128 |

PEFT 不复用原 aligned 的 `_freeze_aligned_inactive_parameters`，以免意外解冻 LoRA base。训练入口的实际参数审计已确认冻结 vision/projector/VQ/base 没进入 optimizer，指定 trainable 没漏收。

此前 PEFT 实现未修改 X-VLA、UniLIP 或共享 evaluator。当时 aligned 恢复所绑定的八个文件（`train_seen10.py`、`csgo_seen10/model.py`、`csgo_seen10/data.py`、`csgo_seen10/artifact_contract.py`、`autoregressive/models/gpt_t2i.py`、`autoregressive/train/train_c2i.py`、两个旧配置 JSON）的 SHA 均保持不变，见验收根下 `original_source_preservation.json`。`csgo_seen10/compiled_inference.py` 已增加 PEFT opt-in，不能列作保持旧 SHA 的文件。新实验有独立 run、checkpoint、预测、evaluation 和日志目录。

## 7. 小批量验收标准与结果

1. 配置解析和旧命令兼容；新旧 run/config/checkpoint/预测/evaluation/log 明确隔离。
2. 10 地图、50,000/5,000/20,000/12,800 计数与 manifest/selection/calibration 身份一致；推理 target FPV 隔离。
3. 完整 trainable/frozen/optimizer membership audit；36 层每个目标层覆盖，Q/K/V rank 独立，bias/scaling/dropout 正确，零增量时与 base 前向一致。
4. 检查默认 micro8×16，以及 1×128、4×32、16×8 等合法组合的解析、loss 缩放和 epoch 计数；拒绝 8×8 等有效 batch 不为 128 的组合。每 update 必须为 128 条源记录、每 epoch 390 updates；scheduler/global step 按 optimizer update 更新，并验证 warmup 起止和最终 LR。
5. 独立 smoke 目录中少量训练、完整验证流程缩小版、保存后新进程恢复；核对模型、LoRA、小模块、optimizer、scheduler、AMP scaler、RNG、sampler/dataloader 与累计边界，不以单纯重载成功代替恢复一致性。
6. GPU 测量至少完成有反向与 Adam 状态初始化的完整 optimizer update，记录 allocated/reserved 峰值和短测耗时；正式训练采用的 batch 组合须在启动前固定，不能在 run 途中随意改身份。
7. 离散/连续各少量推理、同一 checkpoint、448 RGB JPEG、样本/clip identity、恢复保留已有图；若 merge LoRA，单独检查 merge 前后 logits/采样数值和 compiled 路径。
8. 共享 evaluator smoke 检查，明确 frame-only 与 temporal/FVD 的覆盖边界；完整正式评测仍要求全部样本，smoke 不报告成正式结果。

已完成的小批量训练证据位于 `outputs/csgo_seen10_exp32gen_aligned_peft_smoke/acceptance_20260925_000701`：默认 micro8×累计16 完成 2 updates，模型/optimizer 参数审计和冻结权重逐位核对通过。严格恢复在同一 step 1 checkpoint 分叉、测试专用 deterministic math/SDPA 设置下，step 2 的模型、optimizer、scheduler、scaler、RNG 与 sampler 等状态逐位一致，见 `deterministic_resume_comparison.json`。默认高性能 GPU 后端恢复的下一步 forward loss 完全相同，但 backward 有非确定性；不宣称默认后端逐位恢复。离散 64 张和连续完整 64 帧 clip 的编译推理、448 RGB/样本身份、已有图片跳过不变检查以及共享 evaluator smoke 均通过。连续 smoke 覆盖 TWE/TDE，FID/FVD 按共享 smoke 规则跳过；micro16、多卡和正式全量运行没有执行。完整结果、编译修复边界和证据见 [PEFT 验收报告](outputs/csgo_seen10_exp32gen_aligned_peft_smoke/acceptance_20260925_000701/ACCEPTANCE.md)。

完整训练仍只在 3,900、7,800、11,700、15,600、19,500 验证保存；late/final 是主结果，best 为补充。小批量验收即使通过，也不自动启动正式 19,500-step 训练或 20,000/12,800 全量推理，由用户验收后手动执行。具体命令见 [CSGO_SEEN10.md](CSGO_SEEN10.md)。

## 8. 跨服务器迁移接入（2026-09-25）

参考 X-VLA、OpenVLA-OFT、RDT 的机器路径与实验配置分离方式，继续支持旧命令，并增加独立环境准备、路径打印和本地资产检查。定位项目的模型/head/优化配置没有移植到本项目。ControlAR 的生成评测依赖单独准备 `.venv-eval`，仅使用共享 evaluator，不回退到项目内旧指标副本。

| 文件 | 本轮作用 |
| --- | --- |
| `csgo_seen10/paths.py` | CLI/env/config 路径优先级、旧机器默认路径缺失时回退、项目相对路径 |
| `scripts/csgo_runtime_paths.py`、`scripts/run_csgo_seen10.sh` | 无模型导入的 `--print-paths`、三 profile 的统一数据/evaluator/Python 参数传递 |
| `train_seen10.py`、`train_seen10_peft.py`、`infer_seen10.py` | 使用解析后的机器数据路径；保留其他预算/模型/checkpoint/预测身份约束 |
| `scripts/validate_csgo_seen10_{aligned,peft}.py` | 对实际数据位置验证 manifest、split、calibration，不再把物理位置当作实验配方 |
| `csgo_seen10/source_compat.py`、`csgo_seen10/legacy_source_resume.json` | 默认严格源码身份；显式、完整 SHA 白名单的旧同路径 checkpoint 恢复兼容审计 |
| `scripts/setup_csgo_seen10.sh`、两个 `requirements-csgo-*.txt` | 独立训练/评测环境、已有环境保留、只读 CPU 导入检查和用户可选 BF16 CUDA 检查 |
| `scripts/download_csgo_seen10_assets.py` | `--check` 完整本地哈希、`HF_ENDPOINT`，官方 revision/SHA 不变 |
| `scripts/test_csgo_seen10_{paths,source_compat,entry_paths}.py`、`tests/test_csgo_{setup,runner_paths}.py` | 路径覆盖链路、恢复负例、临时环境 mock、无任务 runner 路由测试 |

三个 canonical 配置 JSON、数据读取/预处理、模型、LoRA、采样公式和评测指标实现均未改动。新训练保存实际源码哈希和解析后的真实数据路径。旧同路径恢复的兼容操作需要 `--allow-legacy-source-resume`；跨路径旧 checkpoint/旧预测续写仍不支持，禁止改写其 identity 来绕过检查。

本轮验收：18 项 CPU/脚本用例、实际复制 metadata 后的 aligned/PEFT 检查、真实历史 aligned/PEFT identity 的显式源码转换核验、当前环境 CPU 导入、官方权重完整哈希和脚本语法检查。未重新安装环境或执行训练/推理/评测任务，未运行 GPU smoke。新服务器稳定版依赖的实际安装、独立 `.venv-eval`、BF16/compile 硬件兼容和完整指标缓存仍需在目标服务器验证。证据见 [迁移验收报告](outputs/csgo_seen10_portability_checks/20260925_005013/ACCEPTANCE.md)，使用命令见主文档第 4 节。
