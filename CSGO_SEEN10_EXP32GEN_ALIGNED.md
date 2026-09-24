# ControlAR aligned 说明已合并

已实现的 `csgo_seen10_exp32gen_aligned` 统一记录在 [CSGO_SEEN10.md](CSGO_SEEN10.md)：

- 第 1–3 节：比较口径、数据、可训练模块、训练预算和 best/late 规则。
- 第 4–5 节：环境权重、训练与恢复、两个 checkpoint role 的推理和评测命令。
- 第 6–8 节：独立输出、共享评测协议、验收、现有结果和带日期的运行状态。

本文件保留为旧链接入口，不再重复维护参数或命令。原文“尚未启动正式训练”是接入当时的历史状态；当前运行快照以主文档为准。

新 `csgo_seen10_exp32gen_aligned_peft` 已实现独立配置、`train_seen10_peft.py` 训练、LoRA 与预测身份合同；小批量 GPU 验收已完成，正式训练未启动，等待用户手动验收。默认 micro8×累计16，允许其他正整数组合，只要求有效 batch=128；与原 aligned 固定 micro1×累计128 分离。手动训练、恢复、best/late 推理和评测命令见 [CSGO_SEEN10.md](CSGO_SEEN10.md)，设计与验收标准见 [CSGO_SEEN10_PLAN.md](CSGO_SEEN10_PLAN.md)。

已核对的训练小批量证据位于 `outputs/csgo_seen10_exp32gen_aligned_peft_smoke/acceptance_20260925_000701`：默认 8×16 完成 2 updates、参数审计和测试专用确定性恢复检查。PEFT 推理使用 `compiled_logits+cuda_aten_fp32_inverse_cdf` 分界；共享 `compiled_inference.py` 增加 PEFT opt-in，原 aligned/legacy 默认路径不变。离散 64 张与连续 64 帧推理、已有图片跳过和共享 evaluator smoke 已通过（不含 FID/FVD），详见 [验收报告](outputs/csgo_seen10_exp32gen_aligned_peft_smoke/acceptance_20260925_000701/ACCEPTANCE.md)。
