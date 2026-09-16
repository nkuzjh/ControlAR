# ControlAR 接入 CSGO Benchmark v2 Seen-10

本次按 `csgo_benchmark_v2_start.md` 的项目参数实施：GENERATION、seed 0、BUILD_SHARED_EVALUATOR=1、RUN_FULL=0。文档中的迁移历史、其他模型和定位训练不属于本项目工作范围。

## 接入方案

- 复用 `autoregressive/models/gpt_t2i.py`、DINOv2 control adapter、`autoregressive/models/generate.py` 与原生 VQ-16 tokenizer。采用官方 GPT-XL / DINOv2-small 任意分辨率权重，原生 448 分辨率。
- radar 直接作为 control image；归一化数值 `[x,y,z,pitch,yaw]` 经可训练 MLP 接入原生 caption conditioning，并加入固定地图条件；不需要运行 T5，也不将 pose 仅编码成自然语言。
- 新增 manifest-driven 数据适配器：从 minimal report 解析图片/radar，从 manifest 引用的 split 读取 identity、pose、clip/frame 顺序；Z 使用已发布 calibration。训练/验证读取目标，推理只读取 radar 与 pose。
- 轻量训练入口沿用原生 teacher-forcing CE、AdamW 参数分组、AMP/DDP 和 model/steps/args checkpoint 字段，补齐 optimizer/RNG 恢复与 validation 选优。推理两任务锁定同一 checkpoint，保存 UniLIP 一致的 448 RGB JPEG 和 identity。
- 新建独立 `csgo_benchmark_v2_eval/`：仅接收标准预测、读取数据合同，移植 UniLIP 保留指标的实际算法及默认参数，不 import UniLIP/ControlAR 模型代码。正式评测要求完整覆盖；smoke 单独标记且不生成正式汇总。

## 预计文件

- `csgo_seen10/`：数据、数值条件模型和公用配置/保存加载代码。
- `configs/csgo_seen10*.json`、`train_seen10.py`、`infer_seen10.py`。
- `scripts/run_csgo_seen10.sh`、环境/权重准备入口与最小依赖清单。
- `csgo_benchmark_v2_eval/`：独立数据合同、metric 实现、`run_eval.py`、唯一 requirements/config。
- `CSGO_SEEN10.md`：实际环境、运行命令和 smoke 证据。

## 运行与验收边界

预期统一入口：`bash scripts/run_csgo_seen10.sh {smoke,train,infer,eval} --seed 0`；以后直接更换为 seed 1/2。支持原生 checkpoint 续训；已有预测不覆盖，缺失预测可补齐。

先用独立模型环境完成真实数据 batch、原生模型 forward/backward、checkpoint 保存重载、标准 448 RGB 生成和统一评测读取。主代理审查输入隔离、split/clip identity、相同 checkpoint、metric 对齐，并自行执行核心验收。完整训练与 Table 1 数字留给 RUN_FULL=1；smoke 不作为正式结果。
