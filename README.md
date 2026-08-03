# Injected Bug Debugging Benchmark

注入式 Bug 的模型调试基准与归因可信度报告（研究项目）。完整设计与术语见
[injected-bug-debugging-benchmark-design.md](injected-bug-debugging-benchmark-design.md)。

**一句话**：通过可控微调向小模型注入已知 bug 来构造内部机制真值，统一评估各类归因方法
（激活修补、消融、EAP、ACDC、SAE 等）的定位正确性，并交付自动化的“可信度报告”工具。

## 仓库结构

| 目录 | 职责 | 对应设计章节 |
|---|---|---|
| `inject_bugs/` | bug 注入：数据生成器与 LoRA 微调脚本 | §5 |
| `ground_truth/` | 修复性真值搜索、DNF 表达、修复判定协议 | §5.5 |
| `evaluate/` | 统一归因方法与指标实现 | §6 |
| `credibility_report/` | 可信度报告工具库与 CLI | §7 |
| `common/` | 共享基础设施（设备、配置、日志、路径、随机种子） | 扩展 |
| `configs/` | YAML 实验配置示例 | — |
| `tests/` | 冒烟测试 | — |

## 环境与设备切换（本机 CPU ↔ 腾讯云 GPU）

依赖由 `uv` + `pyproject.toml` 管理（`uv.lock` 锁定版本）。本机（Windows）PyPI 的 torch
轮子即 CPU 版；腾讯云 Studio（Linux + GPU）的 torch 轮子自带 CUDA，同一套 `uv sync`
即可，无需改代码。

```bash
# 本机 / 腾讯云 Studio 通用
uv sync
uv run pytest
```

设备按以下顺序解析（`common/device.py`）：

1. 环境变量 `IBB_DEVICE`（如 `cpu`、`cuda:0`）；
2. 配置文件 `device` 字段（`--config configs/cloud_gpu.yaml`）；
3. 自动检测：有 CUDA 则 `cuda`，否则 `cpu`。

```bash
# 本机：自动解析为 cpu
uv run python -c "from common.device import resolve_device; print(resolve_device())"

# 云端：显式指定（二选一即可）
export IBB_DEVICE=cuda:0
uv run python -m credibility_report.cli --config configs/cloud_gpu.yaml --model gpt2 --task ioi
```

云端存储路径可用环境变量覆盖：`IBB_DATA_ROOT`、`IBB_CHECKPOINT_DIR`、`IBB_RESULTS_DIR`、
`IBB_LOGS_DIR`。如需指定 CUDA 版本，请按
[PyTorch 官方指引](https://pytorch.org/get-started/locally/) 用对应 index 重装 torch。

> 注意：本仓库自带 `evaluate` 包，与 HuggingFace 的 `evaluate` 库同名；本项目的指标为
> 自研实现（设计文档 §6.2），请勿混用。

## 快速验证

```bash
uv run pytest                # 冒烟测试
uv run python -m credibility_report.cli --model gpt2 --task ioi   # CLI 骨架演示
```

## Phase A（toy 真值管线）

`configs/phase_a_toy.yaml` 定义 2 层 × 4 head toy transformer（组件空间 = 8 heads + 2 MLPs）、
三类 bug（`trigger_backdoor` / `compositional_logic` / `knowledge_conflict`）的植入机制（S*）
与期望 DNF，以及数据量与训练超参。运行完整管线（3 类 bug × 3 seeds：数据生成 → 基座训练 →
掩码微调注入 → 质量门槛 → 穷举真值 → 贪心搜索 → DNF 恢复 → 指标 sanity → 汇总）：

```bash
uv run python scripts/run_phase_a.py --config configs/phase_a_toy.yaml
```

报告输出到 `results/phase_a_report.json`（含每类 bug 的找回率、3 seeds IoU、贪心 vs 穷举 F1、
DNF 合取项、指标 sanity 结果与实测耗时）。相关测试见 `tests/test_phase_a.py`。

## 阶段规划（设计文档 §8）

- Phase A：toy transformer 上验证真值管线（穷举可恢复、贪心近似、指标 sanity check）。
- Phase B：GPT-2 Small + 5 类 bug，首批归因方法排名与校准版报告。
- Phase C：跨模型（Pythia、Gemma-2-2B）与自然锚点（IOI、induction heads）。
- Phase D：下游应用（知识编辑、SAE steering）演示。

## 许可证

MIT，见 [LICENSE](LICENSE)。