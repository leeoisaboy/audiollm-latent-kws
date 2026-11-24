<p align="center">
  <a href="https://github.com/leeoisaboy/audiollm-latent-kws">
    <img src="docs/assets/title_banner.zh.png" alt="AudioLLM-Latent-KWS 语言无关的冻结音频大模型五步关键词检索引擎" />
  </a>
</p>

<p align="center">
  <a href="README.en.md"><img alt="English" src="https://img.shields.io/badge/English-README-2f81f7?style=flat-square"></a>
  <a href="https://leeoisaboy.github.io/audiollm-latent-kws/"><img alt="Live Studio" src="https://img.shields.io/badge/Live_Studio-Latent--KWS_Audit_Studio-0071e3?style=flat-square"></a>
  <a href="https://huggingface.co/datasets/Leeoisaboy/audiollm-latent-kws-bench"><img alt="HF Dataset" src="https://img.shields.io/badge/HuggingFace-L1%2FL2%2FL3_Benchmark-ff9d00?style=flat-square"></a>
  <a href="https://leeoisaboy.github.io/disentangle-acoustic-print/"><img alt="Sister Project" src="https://img.shields.io/badge/姐妹项目-Acoustic_Print_Studio-2b9c94?style=flat-square"></a>
  <img alt="Tests" src="https://img.shields.io/badge/tests-26%20passing-2ea44f?style=flat-square">
  <img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-555555?style=flat-square">
</p>

<h1 align="center">AudioLLM-Latent-KWS</h1>

<p align="center"><strong>零外部阅读器、零拼音依赖、不改一行权重：把冻结音频大模型收敛成跨方言、跨语种的五步纯函数开放词表 KWS 引擎。</strong></p>

## 方法流程图 · Architecture Pipeline

<p align="center">
  <img src="docs/assets/pipeline_architecture.jpg" alt="AudioLLM-Latent-KWS 语言无关五步流水线与双后端公开基准验收架构图" width="1000" />
</p>

## 正在运行的审计台

<p align="center">
  <a href="https://leeoisaboy.github.io/audiollm-latent-kws/">
    <img src="docs/assets/audit-studio-demo.gif" alt="Latent-KWS Audit Studio 动态白盒审计台" width="960" />
  </a>
</p>

<p align="center"><strong><a href="https://leeoisaboy.github.io/audiollm-latent-kws/">▶ 进入在线 Latent-KWS Audit Studio</a></strong></p>

审计台采用无边框浅色长卷轴设计，随滚动自动展开。除了保留完整的历史双网格泄漏对比与五折森林裁判台，新版工作台同步接入了 **L1 沪语 / L2 多方言 / L3 英文开放集** 三级公开基准矩阵、`Step-Audio-2` 与 `Qwen2-Audio` 双后端层位扫描结果，以及 **3.07× / 7.93× Prefix KV-Cache** 实测竞赛视图。页面右上角仍支持拖入本地 `pool.jsonl`、`lm.jsonl` 在浏览器内存里零上传白盒审阅。

| 模块 | 页面里你能看到什么 | 对应代码 / 证据 |
| --- | --- | --- |
| 00 三级公开矩阵 | L1 沪语 / L2 KeSpeech / L3 LibriPhrase × 双冻结后端 + 6 项硬性门禁看板 | `analysis/gates.py`、`docs/CONVERGENCE_RESULTS.md` |
| 01 减法消融与零泄漏 | 外部 40 ms 阅读器硬切（35.2% 泄漏）→ 磁吸（3.4%）→ 无阅读器 Listwise 多选（**0.00% 泄漏**） | `core/pipeline.py`、`scripts/run_prefix_ablation.py` |
| 02 两阶段重排与开放集 | Stage-1 浅层粗定位 → Stage-2 共享音频 KV-Cache 重排（含 `NONE` 开放集拒识） | `scripts/test_listwise_mcq.py`、`analysis/open_set.py` |
| 03 双后端 KV-Cache 竞赛 | `Step-Audio-2`（**3.07×**, `r=0.99993`）与 `Qwen2-Audio`（**7.93×**, `r=1.00000`） | `integration/step_audio2_backend.py`、`qwen2_audio_backend.py` |
| 04 五折裁判台 | 按说话人分组五折交叉验证 95% 置信区间森林图，诚实归档 7 项阴性实验 | `experiments/ARCHIVED.md`、`analysis/open_set.py` |

## 为什么做这次通用化收敛

早期原型虽然在私有吴语语料上跑通了，但背着四处强耦合：依赖外部 40 ms 字符阅读器生成 `readings.jsonl`、绑定普通话 `pypinyin` 拼音库、写死 `Step-Audio-2` 第 2 层与 8 帧滑窗、且没有带负例的公开开放集验收。

这一版的核心不是继续往上叠模块，而是**做架构减法、依赖解耦与跨语种公开验收**：

1. **砍掉外部 40 ms 阅读器与普通话拼音**：核心包 `core/` 经 AST 静态审计 **0 语种/拼音库依赖**，用「共享音频 KV-Cache + Listwise 多选题（含 `NONE` 选项）」或「二元 Yes/No Log-Odds」替代外部字符轨，前缀泄漏直接压到 **0.00%**。
2. **极简 3 方法后端接口**：将 [integration/backend.py](integration/backend.py) 抽象为 `extract_shallow`、`prefill_prefix_kv`、`score_suffixes` 三个方法，同时跑通 **`Step-Audio-2-mini`（80 ms/帧）** 与 **`Qwen2-Audio-7B-Instruct`（40 ms/帧）** 两套异构开源冻结 Audio-LLM。
3. **五步纯函数核心链路**：`enroll`（按注册中位时长自动换算窗口帧宽）→ `locate`（浅层余弦 / 3 段时序金字塔）→ `safe_prefix_cut` → `kv_rerank`（池内 z-score 融合 + 正逆序去偏）→ `threshold`（开放集拒识阈值）。
4. **三级全公开基准验收**：在 Hugging Face 公开预切分的三级测试集 **[Leeoisaboy/audiollm-latent-kws-bench](https://huggingface.co/datasets/Leeoisaboy/audiollm-latent-kws-bench)**（含严格说话人隔离与 1:1 真实/难易负例），本仓库只公开非对齐核心算法与全自动 6 项门禁评估代码。

## 三级公开基准与 H20 双后端实测

公开预切分评测集地址：**[huggingface.co/datasets/Leeoisaboy/audiollm-latent-kws-bench](https://huggingface.co/datasets/Leeoisaboy/audiollm-latent-kws-bench)**（每级 40 词 / 120 注册 / 180 正例 + 180 负例，全部说话人隔离）：

- **L1 同语族复现（沪语 / 吴语）**：`Shanghai_Dialect_Conversational_Speech_Corpus`（14 说话人）
- **L2 跨汉语方言**：`KeSpeech` 多方言测试集（96 说话人）
- **L3 跨语种开放集**：`LibriPhrase` 英文短语（160 说话人，50% 难负例 + 50% 易负例）

### 1. 双冻结后端 vs Whisper ASR 模糊检索（NVIDIA H20 · GPU0 · fp16 · 0.00% 前缀泄漏）

对 `Qwen2-Audio` 做 11 层扫描后发现，其双向 `audio_tower` 输出（`layer=-2`，无因果文本左文混合）相比原硬编码 `layer=2` 将三级 `R@20` 直接拉升 **+22.8 至 +42.5 pp**，使**双后端在全部三级公开集上均通过 ≥ 85% 召回底线**：

| 后端 / 基线 | 公开层级 | 单句耗时 | Top-20 召回 | S1 Top-1 | 最终 Top-1 | 相对 S1 净提升 (95% CI) | 开放集 AUC | Recall@5%FAR |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Whisper-tiny ASR + 模糊检索 | L1 沪语 | 0.697 s | — | — | 3.3% | — | 0.583 | 4.4% |
| **Step-Audio-2 (`layer=2` + Listwise)** | **L1 沪语** | **0.124 s** | **96.2%** | 34.9% | **40.4%** | **+5.48 pp `[+2.63, +8.43]`** | **0.847** | **31.7%** |
| **Qwen2-Audio (`audio_tower` + Listwise)** | **L1 沪语** | **0.168 s** | **87.7%** | 20.5% | **26.7%** | **+6.16 pp `[+2.01, +10.36]`** | 0.700 | 16.7% |
| Whisper-tiny ASR + 模糊检索 | L2 多方言 | 0.125 s | — | — | 44.4% | — | 0.703 | 40.0% |
| **Step-Audio-2 (`layer=2` + Listwise)** | **L2 多方言** | **0.121 s** | **99.1%** | **72.4%** | **72.4%** | — | **0.973** | **90.6%** |
| **Qwen2-Audio (`audio_tower` + Listwise)** | **L2 多方言** | **0.174 s** | **94.4%** | 55.1% | **61.7%** | **+6.54 pp `[+0.00, +12.86]`** | **0.828** | **50.6%** |
| Whisper-tiny ASR + 模糊检索 | L3 英文开放集 | 0.160 s | — | — | 73.9% | — | 0.885 | 73.3% |
| **Step-Audio-2 (`layer=2` + Listwise)** | **L3 英文开放集** | **0.114 s** | **100.0%** | 94.4% | **97.8%** | **+3.33 pp `[+0.00, +6.86]`** | **0.987** | **91.7%** |
| **Qwen2-Audio (`audio_tower` + Listwise)** | **L3 英文开放集** | **0.151 s** | **97.8%** | 36.7% | **90.6%** | **+53.89 pp `[+45.76, +62.01]`** | **0.942** | **57.2%** |

> **关于 L1 多词句的句级与词级升级口径**：L1 有 35% 的口语句子同时包含 2–7 个目标词，导致单胜者出现级 Top-1 的数学神谕上限仅为 `180/292 = 61.6%`。改用句级 `Utt-Hit@1` 与词级 `mAP / F1` 后，Base 模型 3 段时序金字塔 + Pairwise `A2A+Text` 达 **`61.7%` 句级命中**；接入公开吴语适配器（`Wu-ASR` 浅层 + `Wu-Und` 顶层 Listwise）进一步达到 **`99.3% R@20`、`72.2% Utt-Hit@1`、`93.9% Utt-Hit@5`、`73.9% Macro-F1` 与 `0.896 AUC`**（详见 [docs/CONVERGENCE_RESULTS.md](docs/CONVERGENCE_RESULTS.md)）。

### 2. 双后端 Prefix KV-Cache 一致性与加速比（NVIDIA H20 · fp16 · 20 候选）

| 冻结后端 | Pearson `r` | Spearman `r` | 最大误差 | KV 复用耗时 | 全序列重算 | H20 实测加速 | 验收门禁 (`r≥0.995, >1.3×`) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `Step-Audio-2-mini` | **0.99993** | **0.99992** | 0.032 nats | 0.140 s | 0.429 s | **3.07×** | **PASS** |
| `Qwen2-Audio-7B-Instruct` | **1.00000** | **1.00000** | 0.011 nats | 0.202 s | 1.605 s | **7.93×** | **PASS** |

## 7 项被诚实砍掉的阴性实验与减法记录

所有未通过五折按说话人交叉验证的模块均已从 `pyproject.toml` 生产主包剥离，归档于 [experiments/ARCHIVED.md](experiments/ARCHIVED.md) 供同行避坑：

- **外部 40 ms 字符阅读器 + 拼音近似吸附**：在公开 L1 上不仅引入异构系统与普通话强绑定，收益还显著劣于零泄漏的共享音频 KV Listwise 多选题（`-3.42 pp` vs `+10.62 pp`）。
- **时序形状加权（一阶差分）**：95% CI `[-0.31, +0.40]` 跨零（独立声学轨迹解耦研究见姐妹项目 [disentangle-acoustic-print](https://github.com/leeoisaboy/disentangle-acoustic-print)）。
- **多原型 K-means、PCA 白化、PMI 文本先验校准、自适应扩展窗、多层分数融合**：均在严格说话人隔离验证中掉分或跨零，全部默认关闭。

## 快速开始

### 1. 零后端体验 Audit Studio

在线打开 **[Latent-KWS Audit Studio](https://leeoisaboy.github.io/audiollm-latent-kws/)**，或本地静态启动：

```bash
python -m http.server 8766 --directory studio
```

### 2. 安装核心包并运行五步纯函数 API（零语种依赖）

```bash
git clone https://github.com/leeoisaboy/audiollm-latent-kws.git
cd audiollm-latent-kws
python -m venv .venv
python -m pip install -e ".[dev]"
pytest -q
```

```python
from core import enroll, locate, safe_prefix_cut, kv_rerank, threshold

# 1. 自动按注册中位时长换算滑窗帧宽，构建 L2 归一化词模板
bank = enroll(enroll_items, feature_getter=lambda cid: feats[cid], frame_s=backend.frame_s)
# 2. 浅层隐空间 Top-20 粗定位
pool = locate(utt_id, query_feats, bank, pool_top=20)
# 3. 无外部阅读器、0% 泄漏安全前缀切分
cuts = safe_prefix_cut(pool, mode="empty")
# 4. 共享音频 Prefix KV-Cache 重排 + 池内 z-score 融合
reranked = kv_rerank(pool, wav_path, backend, prefix_cuts=cuts, lambda_lm=1.5)
# 5. 开放集阈值判定（正例触发 / 负例拒识）
decision = threshold(reranked, threshold_value=0.35)
```

### 3. 一键公开基准复现与 6 项门禁校验

下载公开预切分数据集 [`Leeoisaboy/audiollm-latent-kws-bench`](https://huggingface.co/datasets/Leeoisaboy/audiollm-latent-kws-bench) 后，一条命令即可在双冻结后端上运行全量 L1/L2/L3 评测与 6 项硬性门禁断言：

```bash
python scripts/run_convergence_eval.py \
  --benchmarks-root /path/to/audiollm-latent-kws-bench \
  --feat-cache-root /path/to/feature_cache \
  --step-audio2-dir /path/to/Step-Audio-2-mini \
  --qwen2-audio-dir /path/to/Qwen2-Audio-7B-Instruct \
  --output reports/convergence_report.json
```

<details>
<summary>展开目录结构与文档索引</summary>

```text
audiollm-latent-kws/
├── core/                  语言无关 5 步纯函数核心引擎（enroll / locate / safe_prefix_cut / kv_rerank / threshold）
├── integration/           极简 3 方法 AudioLLMBackend 接口 + Step-Audio-2 / Qwen2-Audio 双冻结后端
├── analysis/              开放集 ROC-AUC / EER / Recall@FAR、说话人五折 CV 与 6 项 Release Gates
├── locator/               Stage-1 模板构建、去均值校准与滑窗粗定位
├── reranker/              传统对齐对照与打分适配器
├── studio/                零后端 Latent-KWS Audit Studio 静态网页（GitHub Pages）
├── scripts/               三级公开评测、前缀消融、层位扫描、Listwise 多选与 A2A 实验脚本
├── docs/                  架构说明、模型契约与 CONVERGENCE_RESULTS.md 完整实测报告
├── experiments/           7 项阴性实验归档（已从生产主包剥离）
└── tests/                 核心 5 步链路、AST 零语种依赖与 Release Gates 单元测试（26 passed）
```

</details>
