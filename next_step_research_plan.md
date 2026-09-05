# 下一步研究方案（修订 v3）：修复导向真值的可信度强化与成文

> 状态：2026-09-02 v3 修订稿（v1 2026-09-01 定稿；v2 2026-09-02 复核修订）。v3 在 v2 基础上吸收第二轮独立评审，新增 A″（容量-几何消融）、B2 前置检查与判读修正、negative 端跨 seed 复核，并把方法主表成员规则改为"按 bug 核心集定义的 core recall + sham gap"（含逐 seed 证据）。
> 依据：`results/step1_report.json`、`results/step1_failure_modes.json`、`results/phase_b_step3_report.json`、`results/phase_b_step3_full_summary.json`、`configs/phase_b_step1.yaml`、`ground_truth/repair_search.py`。
> 本文件作为论文的预注册材料发布（原为内部过程文档；日期为自述日期 2026-09-02，未经第三方注册；执行前快照，见下方英文摘要）。

## English pre-registration summary

This document is the project's internal research plan for the Phase-1 study, dated 2026-09-02 (v3). It was written before the A′→A″→A→B→C experiments were run, and it fixes the protocol axes, judgment gates, and main-table membership rule that the paper reports as pre-specified. The date is self-declared (the document was not third-party registered); the same protocol axes are encoded in the pipeline's configuration fingerprint (`common/fingerprint.py` and the config YAMLs), which are committed to the repository, so the timing of the protocol relative to the results is verifiable from repository history. The sections marked "待开工" reflect the pre-execution state; the final implementation lives in the code repository.

Key pre-specified decisions:

- **Main-table membership rule**: for compositional-logic (core `{head(11,2), mlp(11)}`) a method is a member only if both components rank in its top-10 on every seed; for numeric-rule (core `{mlp(9)}`) a method is a member only if `mlp(9)` ranks top-1 on every seed; every member must additionally clear a sham gap ≥ 0.10.
- **Frozen judgment gates**: trigger rate ≥ 0.90, retention ≥ 0.95, core IoU ≥ 0.6, sham gap ≥ 0.10.
- **Interventions (Section A)**: mean ablation (the definitional protocol), zero ablation, and patch-base (per-sample replacement of a component's activation with the clean model's activation on the same input).
- **De-circularity checks (Section B)**: B1 template gate-0 (unablated trigger on a new surface template ≥ 0.90) then gate-1 (trigger < 0.10 after core ablation); B2a patch-sanity threshold 0.30; B2b steering emergence = some α with trigger ≥ 0.5 and retention ≥ 0.9, partial = max trigger ≥ 3× control.
- **C2 random-truth null**: 100 stratified fake-union sets, split by whether they include `mlp(9)`.

## 摘要与定位

- 论文定位不变：单架构（GPT-2 Small）机理可行性 + 失败模式研究，SCI 二区及以上期刊（非顶会冲刺），全文撤下"基准/内部机制真值/可信度工具交付"三处声称。
- **v2 → v3 修订点**：

| v2 条目 | v3 修订 |
|---|---|
| A′（negative 点 pair 闭环，seed 1） | 保留并扩展：新增 TB/KC/FR × `ll8-9-10-11` × seeds {2,3} 的**单组件扫描复核**（抑制器模式跨 seed 一致性，≈1.5–3 GPU·h）；预算允许时加 TB seed 2 pair 复核（≈5 GPU·h）。论文口径：negative 端 = seed 1 全量 pair 枚举 + seeds 2–3 单组件扫描（+ TB seed 2 pair） |
| 无容量-几何对照实验 | **新增 A″**（CL × seed 1 × 4 点：all-layer r16、all-layer r32、`ll8-9-10-11` × `tm=[c_attn,c_proj]`、`[7,8,9,10]` 窗口），≈2–4 GPU·h，排在 A′ 之后、A/B 之前；先用手头同模块数窗口对比作第一层反驳，A″ 结果决定贡献①措辞（几何 vs 容量/几何共同决定） |
| B2 steering 直接上预注册判据 | **新增 B2a**（α=1.0 整段激活替换 sanity，patch 非 Δ 加性）与 **α 对数粗扫 {0.1…8} → 细扫**；B2 全负不再自动触发 ①③ 降级，改为边界小节（权重型注入未必可被激活内容表达） |
| 方法主表：sham gap≥0.10 + AUPRC 门槛（union 口径） | **改为按 bug 核心集的 core recall 成员规则**（预注册）：CL = 两核心组件逐 seed 同入 top-10；numeric = `mlp(9)` 逐 seed top-1；加 sham gap≥0.10。`mean_ablation` 保留为协议参照行（不作"定位者"声称）。叙事③按此重写（见"实现改动与交付物"） |
| numeric 过度分解证据："seed 1 DNF 含 5 项合取" | 订正为准确表述：**三个 seed 的 DNF 各含一个长度为 5 的合取项；`max_conjuncts=5` 是合取项数量上限，实际项数 seed 1/2/3 = 4/3/3，未顶满**（seed 1：union 12、合取长度 1/2/4/5；seed 2/3：union 8、合取长度 1/2/5）——过度分解证据不变，无需收窄 |
| C2 随机假真值对照 | 保留"按含/不含 `mlp(9)` 分层 + 参考方法同算"，补一句 rationale：含 `mlp(9)` 组回答"0.57 是否仅由命中唯一真核解释"（degraded SAE hit@1=1.0），不含组回答纯随机外圈的虚高程度 |
| 预算 Phase 1 ≈25–35 GPU·h | 重估为 **≈30–40 GPU·h**（A′ 13–18 含跨 seed 复核、A″ 2–4、A 8–14、B 3–5、C ~1）；可选 D +8–12、TB seed 2 pair 复核 +5 |

- 核心叙事四条（每条与证据边界绑定，写作不可越界）：
  1. **行为注入成功 ≠ 因果可定位，定位性由注入几何决定**——边界：mean-ablation 修复协议内（CL 全层空 vs `ll8-9-10-11` 稳定 `head(11,2)+mlp(11)`，跨 rank/seed IoU=1.0）；容量混淆由 A″ 与手头证据（同 12 模块窗口位置不同结果不同、`all.r4` 空而 `ll8.r8` strict、`ll8.r4`≡`ll8.r8`）共同排除。
  2. **destructive_suppressor 机制的系统刻画**——边界：单组件扫描层面（156 组件全量判定，TB/KC/FR 仅 `mlp(0/1/2)` 为抑制器、无单组件非破坏性杠杆）；"无最小修复集"的强声称由 A′（seed 1 全量 pair + seeds 2–3 单扫复核）闭环后作出；此前表述一律写"single_scan 有界结果"。
  3. **方法比较（按核心召回）**——**预注册成员规则 + 事前预测**（基于 d5bc64c 报告，C1 重算后按同一规则执行、不事后调阈值）：CL 主表预计仅 **EAP**（`head(11,2)`/`mlp(11)` 逐 seed 排名 6/5/5 与 3/2/3，max=6 ≤10；sham gap 0.176）；numeric 主表预计 **zero_ablation 与 activation_patching**（`mlp(9)` 逐 seed top-1；gap 0.342/0.363）；消融族在 CL 上方向相反：全部漏掉 `head(11,2)`（rank 136–141）、命中 `mlp(11)`（rank 1–4；仅 `activation_patching` seed 2 双漏：head 136 / mlp 156）——"部分定位 vs 完整合取恢复"本身写成发现句；任务依赖只在"每 bug 的胜者"层面陈述，不做跨 bug 泛化声称。
  4. **修复协议噪声的受控证据**——numeric union 随搜索预算膨胀（seed 1 同配置 600s→3 组件、1800s→12 组件）且三 seed DNF 均含一个长度为 5 的合取项（项数 4/3/3，未顶 `max_conjuncts=5` 数量上限），支撑 C1 的核心/union 分层口径。
- 措辞纪律：全文 "behavior-verified repair-oriented truth"，不出现 benchmark / ground-truth 声称；hit@k 等指标语义必须在方法节显式声明（现有实现为"top-k 内命中 ≥1 正例"，**不作为主表成员判据**，主表用 core recall 全含判据）。
## Phase 0 — 文献与术语基线（E；零 GPU；最先启动，与 Phase 1 并行）

- 检索并固定四组对照口径：定位性与容量/摊薄之争（CausalGym、Tracr、causal abstraction、backdoor localization）、冗余/多重实现（hydra effect）、复制抑制（copy suppression）、"定位在哪一层"与"机制如何计算"的 where-vs-how 区分。
- 交付物：`notes/lit_notes.md`（≤10 条核心文献注释书目）、术语表（localized/circuit/mechanism/repair-oriented truth 的允许与禁用表述）、四条核心叙事各 1–2 句 related-work 预写句、以及把"容量过剩摊薄 vs 几何决定"之争映射到文献充分性/互换性口径（interchange/compression 基准）的对照表。
- 验收：术语表覆盖 A″/A 所需的全部几何与容量变量命名；E 不阻塞 Phase 1 启动，但必须早于动笔完成。

## Phase 1 — GPT-2 Small 主实验（预算 ≈30–40 GPU·h，不含 D）

- 执行顺序：A′ → A″ → A → B → C；预算冲突时的让位顺序：D → A 部分协议组合 → B 细扫；A′/A″ 不让位。

### A′ — negative 闭环（最高优先；主枚举 12–15 GPU·h + 跨 seed 单扫 1.5–3 GPU·h；可选 TB seed 2 pair 复核 +5 GPU·h）

- 主枚举（seed 1）：TB/KC/FR × `ll8-9-10-11`，逐 bug 排除该 bug 的 general_suppressors（`mlp(0/1/2)`）后约 153 个组件两两配对 ≈1.16–1.18 万对；逐对 `judge_repair`，`max_evals:15000`、`step_wall_budget_s:21600`、early_stop；≈4.5–5 GPU·h/点。
- 跨 seed 复核（新增，采纳外部意见 4）：同三 bug × seeds {2,3} 单组件全量扫描，检验抑制器模式（仅 `mlp(0/1/2)`）与"无单组件杠杆"跨 seed 一致（≈1.5–3 GPU·h）；预算允许时加 TB × seed 2 的 pair 复核（+5 GPU·h）。
- 判读门：seed 1 全量 pair 全空 且 seeds 2–3 单扫一致 → 叙事②"该注入几何下无最小修复集"强声称闭环；论文口径固定为"seed 1 全量 pair + seeds 2–3 单扫一致"（若含 TB seed 2 pair 复核则升级为跨 seed pair 口径）；找到任一 pair → ② 收窄为"局部 pair 可修复"，该 pair 扩 seeds 2–3 复核；预算兜底 = 仅 TB 一点全量 pair，其余两点按"single_scan 有界结果"表述。

### A″ — 容量-几何消融（新增，采纳外部意见 1；≈2–4 GPU·h；排在 A′ 后、A/B 前）

- 四点（CL × seed 1；沿用 step1 同协议：LoRA 重训注入 + mean-ablation 修复协议内真值搜索；未注明者 r8、`tm=[c_attn,c_proj,c_fc]`）：
  1. all-layer × r16（固定全层几何、容量 ↑）；
  2. all-layer × r32（容量 ↑↑）；
  3. `ll8-9-10-11` × `tm=[c_attn,c_proj]`（局部几何 × attention-only 新组合；勿与 step1 矩阵第 8 点 all-layer × attention-only——CL 的 destructive_suppressor——混淆）；
  4. `[7,8,9,10]` × r8（窗口平移一格；step1 已有 `[6,7,8,9]`=effect_available、`[8,9,10,11]`=strict 两个锚点，该点补窗口边界特异度）。
- 判读：all-layer r32 仍空 → "容量摊薄"解释弱化、几何解释增强——配合手头第一层证据（同 12 模块窗口位置不同结果不同；`all.r4`（144 秩单元）空而 `ll8.r8`（96 秩单元）strict；`ll8.r4`≡`ll8.r8`）；all-layer r16 即可定位 → 贡献①措辞必须收窄为"容量/几何共同决定"；第 3 点空 → ① 边界收窄为"窗口内注意力+MLP 组件集共同决定"，strict → 组件类型几何进一步支持几何解释；第 4 点结果决定窗口特异度表述（strict/effect_available/空三态）。
- 该步是动笔前必须知道的答案：成本低于 A′、优先级高于 A/B/D；若实耗 >4 GPU·h，从 A 的 seed/协议组合让位。

### A — 协议消融（8–14 GPU·h；定位为构造稳定性检查，非循环性终结者）

- 矩阵：`intervention ∈ {mean, zero, patch_base}` × {CL、numeric} × seeds {1,2,3}；CL 在 `ll8-9-10-11` 窗口内搜索、numeric 在注入层段全量搜索（与原报告几何一致）；每点一次真值搜索，≈0.4–0.8 GPU·h/点。
- 关键执行约束：intervention/rank/target matrices/窗口/seed 全量进入配置指纹（见"实现改动与交付物"），防止静默复用既有 mean 协议 analysis 造成协议串扰。
- 判据：主判据 core IoU≥0.6（CL 核心 `{head(11,2), mlp(11)}`；numeric 核心 `{mlp(9)}`）；不套 `n_strict_necessary`（CL 在该口径 = 0，数值病态）。
- 判读：三协议 core IoU 一致通过 → "定位性对修复协议稳健"；mean vs zero 显著分歧 → ① 收窄为协议依赖并写入边界小节。

**patch_base 语义（2026-09-04 定）**：逐样本 counterfactual patch——用 clean(base) 模型在「同一输入」上该组件的逐样本激活替换注入模型该组件激活，直接隔离注入 delta（不是 base 模型 mean，后者与 mean 消融近乎冗余）。

**实现（代码已落地，2026-09-04）**：
- `ground_truth/judgment.py`：`SEARCH_MODES = (mean, zero, patch_base)` + `INTERVENTION_TO_MODE`（协议名 `mean_ablation/zero_ablation/patch_base` → 搜索模式 `mean/zero/patch_base`）。
- `inject_bugs/hooked_utils.py`：新增 `build_patch_base_hooks`；`last_position_logits`/`trigger_rate`/`normal_accuracy`/`joint_trigger_normal_rates` 加 `mode`/`base_model` 透传——`mode="patch_base"` 每 chunk 先跑 base 模型抓激活再 patch，`base_model=None` 时 raise。
- `ground_truth/repair_search.py`：`judge_repair`/`single_component_judgments`/`greedy_search`/`exhaustive_search`/`pair_repair_search`/`recover_dnf` 加 `intervention`（默认 `"mean"`）+ `base_model` 并透传；默认值保证 Phase A / negpair / step1 零回归。
- 测试 `tests/test_phase_b_protocol.py`（7 用例）：patch_base 自源恒等（锁 chunk 对齐）、异源替换、无 base_model raise、三模式 judge_repair 结构、默认=mean 回归、指纹三干预漂移、`INTERVENTION_TO_MODE` 映射。全量 98 用例 + ruff 0 错误。

**待开工（云端，driver 未写）**：`scripts/run_phase_b_protocol.py` + `configs/phase_b_truth_protocol.yaml` + `cloud/run_phase_b_protocol.sh` + `scripts/analyze_protocol.py`。关键约束：intervention **不改变训练**，训练按 `(bug, seed, point)` 隔离、三干预共享一次训练，仅搜索按 intervention 三跑；指纹的 `intervention` 只进搜索缓存键、不进训练 checkpoint 身份（否则三干预冗余重训 3 次）。
### B — 去循环性：新模板 gate + steering（3–5 GPU·h）

- B1（新模板 gate 链）：gate-0 = 在未消融注入模型上验证新模板 trig≥0.90（排除模板特异/检索伪影）；通过 → gate-1 = 消融目标核心组件后新模板 trig 回落 <0.10（与 gate-0 镜像的失效判据）；gate-0 失败 → 写"注入浅层性"独立小节，B 系列不再作为去循环性证据。
- B2a（新增前置 sanity，≈0 GPU·h 量级；采纳外部意见 2.1）：把注入模型 trigger 行在 `head(11,2)+mlp(11)`（及 CL 对照组件组）的激活整段替换 patch 进 clean 模型（patch 检查，非 Δ 加性 steering）；完全无效应 → 方向构造失败（行为不可由组件输出线性方向表达），steering 负结果不可解释，B2 停止并记入边界小节；有效应 → 进入 B2b。
- B2b（α 扫；采纳外部意见 2.2——α=2.0 上限无数据支撑）：方向 = 注入模型 trigger 行组件输出与 clean 之差；先 α 对数粗扫 {0.1, 0.3, 1.0, 3.0, 8.0} 定位涌现量级，再在邻域细扫；负对照必备：随机晚期组件方向 + 另一 bug 真值组件方向，对照 trig 上限 0.15。
- B2 判读（预注册，全计划唯一新增预注册判据）：涌现 = 存在某 α 使 trig≥0.5 且 ret≥0.9；部分涌现 = 最高 trig ≥3×对照；全负 → 边界小节（权重型 LoRA 注入未必可被 clean 激活内容线性表达），不自动降级叙事①③。
- B3：纯写作（method 节 steering 协议 + 上述判读口径），无 GPU。

### C — 真值口径重算与随机对照（≈1 GPU·h，以 CPU 重算为主）

- C1（主表重算）：按"主表成员规则"（见"实现改动与交付物"）对既有逐 seed json 重算 core recall/sham gap，输出主表与附录分类，不重训；AUPRC 仅进 union 附录（CL 2 正例下病态；numeric union 8–12 与核心集口径不一致——采纳外部意见 3(4) 后不再作统一门槛）。
- union 预算膨胀证据（写作素材，数字已核验）：seed 1 同配置 600s→3 组件、1800s→12 组件；三个 seed 的 DNF 各含一个长度为 5 的合取项（`max_conjuncts=5` 为合取项**数量**上限，实际项数 4/3/3、未顶满；seed 1 合取长度 1/2/4/5，seeds 2/3 为 1/2/5）——"核心 vs union 分层"口径由此支撑。
- C2（随机假真值，100 组，CPU）：按"含/不含 `mlp(9)`"分层抽样（纯随机 156 选 8–12 会低估随机命中难度）；zero_ablation/EAP 参考方法用既有逐组件分数同算；补 rationale 句——含 `mlp(9)` 组回答"SAE 的 0.57 是否仅由命中唯一真核解释"（degraded SAE hit@1=1.0），不含组回答纯随机外圈上方法的虚高程度。

### D — 可选跨 seed 扩展（seeds 4–5；+8–12 GPU·h；优先级最低）

- 结构与 A′ 相同（pair 枚举 + 单扫），扩到 seeds 4–5；预算冲突时最先让位；D 与 A″ 冲突时保 A″（外部意见：A″ 优先于 D）。

## Phase 2 — 跨模型迁移（Pythia；F′）

- 目标模型改为 Pythia-160M（12 层 / 768 维 / 156 组件，与 GPT-2 Small 同构；Pythia-70M 仅 6 层、无 `ll8-9-10-11` 等价窗口，弃作主迁移对象）。
- 冒烟门（先行两点，通过才扩矩阵）：CL `ll8-9-10-11` × seed 1；TB all-layer × seed 1。失败 → 二区降级为"同构架构上的探索性对照"，论文按单架构（GPT-2 Small）成文。
- 移植清单：tokenizer 守卫按模型分支（模板 token 化/词表不同）、SAE 管线关闭（SAE 在 GPT-2 残差流上训练、不可移植，方法节注明）、checkpoint/loader 按模型分支、LoRA 目标矩阵与超参按 160M 同构映射（层名前缀不同）。
- 70M 保留仅作非平行对照：层段 `[2,3,4,5]`，方法节标注结构不匹配，不与 GPT-2/160M 结果并表比较。
- 预算：Phase 2 全矩阵在冒烟门通过后单列核定；不影响 Phase 1 的执行顺序与决策。
## 实现改动与交付物

### 代码（入库）
- 配置指纹扩展：真值搜索/analysis 的缓存键与输出 json 指纹字段 = {bug, seed, intervention, rank, target_matrices, window, 协议版本}；任一字段变更 → 指纹漂移、缓存失效（新增回归测试：改配置必改指纹）。
- A′ pair 检查入口：`judge_repair` 支持 pair 候选枚举 CLI（general_suppressors 排除名单按 bug 注入、逐对 early-stop、`max_evals`/`step_wall_budget_s` 参数化）。
- 判据实现：core recall（top-k 全含语义：CL 双组件同入 top-10；numeric 单组件入 top-1）与逐 seed 汇总、sham gap 计算；与既有 hit@5（top-k 命中 ≥1 正例）在接口上显式分离。
- 回归测试：指纹漂移测试、core recall 单测（双组件/单组件边界）、hit@k 语义测试、pair 枚举 smoke（2 组件子集）。

### 配置（新增 yaml，随代码入库）
- `phase_b_truth_protocol.yaml`：A 矩阵（intervention × bug × seed）与判据预注册。
- `phase_b_negpair.yaml`：A′ pair 枚举、排除名单、预算参数。
- A″ 配置：四点容量-几何矩阵（r16/r32/attention-only/窗口平移）。
- `phase_b_step3_ext.yaml`：C1 core recall 主表重算 + C2 100 组分层假真值（分层种子固定）。

### 脚本/驱动
- `truth_validation_suite`：core recall / sham gap / AUPRC（仅附录）/ 逐 seed 表输出。
- `steering_scan`：B2a patch sanity → B2b α 对数粗扫与细扫 + 负对照输出。
- `sae_pollution_check`：C2 分层假真值与 rationale 句。
- `negpair_loop`（A′）与 A″ 驱动器：按配置矩阵提交，输出 json 至 `results/`。

### 主表成员规则（预注册；采纳外部意见——不设统一 Hit@5，按 bug 核心集定义门槛）
- CL：`{head(11,2), mlp(11)}` 逐 seed 同入 top-10（事前预测仅 EAP 通过：`head(11,2)` 排名 6/5/5、`mlp(11)` 排名 3/2/3；sham gap 0.176）。
- numeric：`mlp(9)` 逐 seed 入 top-1（事前预测 zero_ablation 与 activation_patching 通过）。
- 附加 sham gap≥0.10；任一 seed 不满足 → 该行移附录并注明 seed 级数值与不稳定性（例：CL activation_patching seed 2 负例 −0.009/rank 136）。
- 特殊行：`mean_ablation`（含 ACDC-lite，acdc 并入该行并标注）保留为协议参照行、标注循环性，不作"定位者"声称（确认专家此处置正确）；logit_lens 与 SAE 作混淆/降级案例行（logit_lens CL `mlp(11)` rank 155、numeric sham gap −0.004；SAE 受唯一真核/退化解影响）。
- 方法计数口径：独立方法 6 个（EAP、zero_ablation、activation_patching、mean_ablation/ACDC-lite、logit_lens、SAE）；`grad_x_act` 列为 null case 不计方法；SAE 结果只进附录。
- 叙事③按上述主表落地（预注册于"核心叙事"第 3 条）：CL "消融族部分定位（仅 `mlp(11)`，`head(11,2)` rank≥136；`activation_patching` seed 2 双漏）vs EAP 完整合取恢复"写成发现句；numeric zero/act 定位 `mlp(9)`；"任务依赖"仅在每 bug 胜者不同层面陈述（3-seed CI：EAP [0.243,0.625] vs zero [0.173,0.506]，区间重叠，不声称统计显著差异）。

## 验收与决策门

- 贡献①（几何决定定位性）：A″ all-layer r32 空 且 A 三协议 core IoU≥0.6 → 保留强句；A″ r16 可定位 → 收窄为"容量/几何共同决定"；A 协议分歧 → "协议依赖"措辞；所有 negative 结果连同配置（600s 墙钟截断、budget_exceeded 语义）如实入正文/附录。
- 贡献②（destructive_suppressor 系统刻画）：A′ seed 1 全量 pair 全空 + seeds 2–3 单扫一致 → 强声称闭环（含 TB seed 2 pair 复核则升级口径）；找到 pair → 收窄并扩 seeds。
- 贡献③（方法比较）：C1 重算只做"主表/附录"分类，不事后调成员规则；与事前预测不一致时以结果为准并如实报告。
- 去循环性（B）：B1 gate-0 trig≥0.90 → gate-1（消融后 <0.10）；gate-0 失败 → "注入浅层性"小节；B2 涌现 = 某 α trig≥0.5 且 ret≥0.9，部分涌现 = 最高 trig ≥3×对照（对照上限 0.15）；全负 → 边界小节，不自动降级①③。
- C2 判读：zero/EAP 对随机假真值高分（虚高）→ union 口径声称收窄、该对照入附录；预算膨胀证据（600s→3、1800s→12 组件）与 DNF 合取结构证据不受影响。
- 通用阈值冻结（沿用 d5bc64c 报告预注册）：trigger≥0.90、ret≥0.95、core IoU≥0.6、sham gap≥0.10；任何门槛调整须记录为方案修订，不允许静默执行。
- 报告纪律：所有 negative 点与 seed 级不稳定数值（CL activation_patching −0.009/0.398/0.489、logit_lens AUPRC 0.067 等）一律如实入附录；"behavior-verified repair-oriented truth" 措辞贯穿全文。

## 假设

- 云端 T4 单卡可用；时长估算：A′ 4.5–5 GPU·h/点、A″/A 真值搜索 0.4–1 GPU·h/点；排队时间不计入决策顺序。
- 基线 checkpoint（commit d5bc64c）可复用；若环境清理需重训基线 → 预算 ×1.3 并先报备。
- 二区（Pythia-160M）仅作同构性探索；跨架构/跨规模泛化不作声称；70M 仅非平行对照。
- 判据与主表成员规则在 Phase 1 首跑前冻结（见"验收与决策门"）。
