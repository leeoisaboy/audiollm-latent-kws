<p align="center">
  <a href="https://github.com/leeoisaboy/audiollm-latent-kws">
    <img src="docs/assets/title_banner.zh.png" alt="AudioLLM-Latent-KWS 冻结 Audio-LLM 的方言开放词表关键词检索" />
  </a>
</p>

<p align="center">
  <a href="README.en.md"><img alt="English" src="https://img.shields.io/badge/English-README-2f81f7?style=flat-square"></a>
  <img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-555555?style=flat-square">
  <img alt="Status" src="https://img.shields.io/badge/status-research%20prototype-b8860b?style=flat-square">
</p>

# AudioLLM-Latent-KWS

**只录几段方言指令，就能把新词加入关键词检索，不需要重新训练大模型。**

很多语音设备已经能听懂普通话，却很难低成本覆盖吴语等方言的细分说法。每增加一个词，如果都要补数据、标时间、再微调模型，成本会迅速落到长尾词上。这个项目研究一个更基础的问题，**一个已经训练好、参数不再更新的音频大模型，内部是否已经保存了足够的声学信息，让新词只靠少量注册录音就能被找到？**

项目采用两阶段检索流程，全程不更新模型参数。

1. 第一步先找候选。把每条语音切成小段，与每个词的平均声学模板比较，保留最像的 20 个词。
2. 第二步再看语境。把候选词放回冻结的大模型中，结合它前面的文字重新打分，选出最终结果。

项目定位为可复现的研究基线。它量化了两个容易被忽略的问题，一个是两套时间网格的对齐，另一个是时序形状信号为什么没有转化为最终收益。

## 主要结果

下面的数字来自 2031 条吴语语音、3244 次目标词出现、293 个词的真实语料实验。

| 我们关心的问题 | 结果 | 怎么读 |
|---|---:|---|
| 正确词有没有进入前 20 个候选？ | **94.7%** | 3244 次目标词中，有 3071 次至少进入前 20；这决定后级重排有没有机会看到正确答案 |
| 只靠声学，目标词排第一的比例？ | **47.4%** | 3244 次目标词中，1538 次直接排第一 |
| 加上语境重排后的默认结果？ | **54.2%** | 3244 次中 1757 次排第一，不需要额外安装拼音库 |
| 再加拼音近似匹配的结果？ | **55.0%** | 3244 次中 1783 次排第一，是推荐组合；语言模型权重调到 5.5 时，单看这一指标可到 55.1% |

94.7% 是当前候选池的上限。剩下的 5.3% 没有进入前 20，后级重排无法把它们找回来。排第一按每一次目标词出现计算，同一个词在同一句里出现两次就计两次。

文中还会出现词级指标。它先对同一个词的所有出现求平均，再让每个词等权，避免高频词把总体数字带偏。

拼音近似匹配是可选分支。语音阅读器可能把关窗识成官窗。程序会用普通话拼音做近似匹配，避免把目标词的首字误留进上下文。

最终推荐组合如下。

```text
模型第 2 层声学表示
  + 8 帧（约 640 毫秒）平均模板
  + 注册集全局均值扣除
  + 前 20 个候选
  + 按 40 毫秒文字网格对齐前文
  + 声学分 + 4.0 × 语言模型分
  + 可选普通话拼音近似匹配
  + 时序形状关闭
```

其中 `4.0` 是语言模型项相对于声学项的权重，声学项本身的权重始终是 1。这个权重并非越大越好。实测曲线在 4.0 到 5.5 之间很平。只看排第一这一个指标，5.5 的 55.1% 是最高点；4.0 为 55.0%，但词级指标更高（64.5% 对 64.1%）。两者差 0.1 个百分点，落在噪声范围内，因此仓库保留更保守、也更均衡的 4.0。

## 两套时间网格容易出错

音频模型每 **80 毫秒**一帧，外部文字阅读器每 **40 毫秒**一个字。两套网格并不重合。如果直接按最近帧硬切为上下文边界，目标词的第一个字很容易被留在前缀里，相当于提前把答案泄露给重排模型。

我们在 3244 次真实目标词上检查了上下文切分。

| 上下文切法 | 前缀中出现目标词首字的比例 | 含义 |
|---|---:|---|
| 按最近帧硬切 | **35.2%** | 超过三分之一的目标词把首字泄露进了上下文 |
| 先对齐到文字字符网格，再切开 | **3.4%** | 泄漏下降到约十分之一 |

这一步只改变上下文从哪里开始，不改变模型、候选池或融合权重，却能决定重排是否在利用答案。完整检查见 [reports/prefix_audit_real.json](reports/prefix_audit_real.json)。

时序形状的实验结果需要单独说明。一阶差分表示相邻帧的变化方向。它单独使用时确实有排序能力，但以固定权重加到声学分数和语言模型分数上，没有带来稳定收益。

| 验证方式 | 五折交叉验证平均增益 | 95% 区间 | 结论 |
|---|---:|---:|---|
| 按语音分组 | **+0.17 个百分点** | [-0.20, +0.56] | 区间跨零，不能算稳定收益 |
| 按说话人分组 | **+0.03 个百分点** | [-0.31, +0.40] | 同样跨零 |

因此仓库保留实验代码，默认路径不启用这个分支。这个结果说明，一个信号单独有效，并不代表固定线性相加后仍能带来收益。

## 验证方式

### 8 帧均值模板的定位结果更稳

早期实现固定使用 5 帧（400 毫秒）。在完整语料上改成 8 帧（640 毫秒），再扣除注册集的全局均值后，前 20 个候选的召回从 92.5% 升到 **94.7%**。

为了避免在测试集上挑参数，我们采用五折交叉验证。每折只用训练部分选配置，再在没见过的语音或说话人上评估。

| 比较 | 目标词排第一的变化 | 95% 区间 | 说明 |
|---|---:|---:|---|
| 8 帧 + 去均值 对比 原 5 帧方案 | **+3.70 个百分点** | [+2.80, +4.61] | 按说话人分组，五折均选择 8 帧 + 去均值 |
| 同一比较，按语音分组 | **+3.51 个百分点** | [+2.62, +4.42] | 5 折中有 4 折选择去均值 |

说话人分组来自语料文件名中的 `rec_sXX` 规则，尚未使用官方发布的说话人编号；正式论文仍应在官方划分或第二套独立语料上复核。

### 重排权重的默认值改为 4.0

旧版本使用 `1.5 × 语言模型分`。它来自更早的实验，并没有在当前语料上重新选择。重新扫描后得到下面的结果。

| 比较 | 目标词排第一的变化 | 95% 区间 |
|---|---:|---:|
| 默认网格对齐，权重 4.0 对比 1.5 | **+1.05 个百分点** | [+0.47, +1.64] |
| 网格对齐加拼音近似匹配，权重 4.0 对比仅网格对齐，权重 4.0 | **+0.80 个百分点** | [+0.37, +1.26] |

这也是仓库把默认值改成 `4.0` 的原因。拼音近似分支需要额外安装 `pypinyin`；不安装时，默认的精确网格对齐仍然可用。

## 没有进入默认路径的实验方案

下面这些方向都真实跑过。保留代码是为了复现实验结果。

| 方案 | 真实语料上的观察 | 当前决定 |
|---|---|---|
| 按词长自动调窗口、在峰值附近扩展窗口 | 目标词排第一 46.5%，候选召回 93.4%，低于 8 帧固定窗 | 不进入默认路径 |
| 多个子模板（K 均值） | 2 个子模板约 45.0%，候选召回 91.7%，更弱 | 保留在实验代码 |
| 去主成分、主成分白化 | 分别约 39.5% / 43.2%，候选召回明显下降 | 保留作对照实验 |
| 多层分数融合 | 单用第 1 层的排第一略高，但候选召回从 94.7% 降到 93.4% | 默认仍用第 2 层 |
| 同一候选保留两个起点 | 权重 4.0 下 +0.68 点，弱于拼音近似 | 不进入默认路径 |
| 文本先验校准 | 旧权重 1.5 下 `gamma=1` 为 **-0.96 点**；权重 4.0 下为 **-3.45 点** | 保留作结果记录 |
| 时序形状加权 | 最终融合分数上 +0.17 / +0.03 点，区间均跨零 | 默认不启用 |

这些结果表明，当前系统的主要瓶颈在候选池上限、两套时间网格的对齐，以及声学分数与语言模型分数的融合方式。继续叠加模块，容易只在单一指标上得到小幅提升，却缺乏统计支持。

## 缓存带来的实际加速

同一句语音的系统提示词和音频前缀在给 20 个候选打分时完全相同。因此先算一次音频前缀并缓存中间状态，再让每个候选只增量计算自己的短文本，是合理的工程优化。

在 H20、GPU0、20 个候选、半精度（fp16）条件下，实测结果如下。

| 音频长度 | 全序列重算 | 复用缓存 | 加速 | 额外峰值显存 |
|---|---:|---:|---:|---:|
| 1.0 秒 | 0.283 秒 | 0.160 秒 | 1.77 倍 | 174 → 154 MB |
| 3.6 秒 | 0.239 秒 | 0.183 秒 | 1.30 倍 | 127 → 123 MB |
| 6.4 秒 | 0.301 秒 | 0.141 秒 | 2.14 倍 | 192 → 163 MB |

在 200 条对照语音上，缓存与全序列的分数最大差约 0.04 个对数概率单位，排序相关系数不低于 0.997，最终答案在 199/200 条上一致。实测加速为 1.3 到 2.1 倍。候选文本只有 1 到 4 个词元，原来的实现已经会把长度相同的候选放在一起计算，因此缓存带来的提升有限。

## 快速运行

合成数据只用于检查接口能否串起来，不代表真实准确率。

```bash
git clone https://github.com/leeoisaboy/audiollm-latent-kws.git
cd audiollm-latent-kws
python -m venv .venv
python -m pip install -e ".[dev]"
python scripts/make_fixtures.py --output data/synthetic
```

```bash
python cli.py locate \
  --templates data/synthetic/templates.jsonl \
  --enroll-dir data/synthetic/enroll \
  --search-manifest data/synthetic/search_manifest.jsonl \
  --search-dir data/synthetic/search \
  --latent-width 16 \
  --window 8 \
  --calibration center \
  --pool-top 20 \
  --include-gold \
  --output reports/synthetic/pool.jsonl

python cli.py evaluate \
  --pool reports/synthetic/pool.jsonl \
  --pool-top 20 \
  --output reports/synthetic/stage1.json

python cli.py rerank \
  --pool reports/synthetic/pool.jsonl \
  --reading data/synthetic/readings.jsonl \
  --wav-dir data/synthetic \
  --scorer examples.dummy_scorer:create_scorer \
  --output reports/synthetic/lm.jsonl

python cli.py fuse \
  --pool reports/synthetic/pool.jsonl \
  --lm reports/synthetic/lm.jsonl \
  --lambda-lm 4.0 \
  --output reports/synthetic/metrics.json

python cli.py audit \
  --pool reports/synthetic/pool.jsonl \
  --lm reports/synthetic/lm.jsonl \
  --reading data/synthetic/readings.jsonl \
  --output reports/synthetic/prefix_audit.json
```

真实实验需要私有音频、注册清单、文字阅读器和冻结模型。特征导出仍使用：

```bash
python integration/step_audio2.py \
  --model /path/to/Step-Audio-2-mini \
  --manifest /private/path/all_clips.jsonl \
  --enroll-manifest /private/path/enroll.jsonl \
  --search-manifest /private/path/search.jsonl \
  --output-root features \
  --layers 0,1,2,3,4,8 \
  --device cuda:0 \
  --quantization 4bit \
  --dtype float16
```

仓库不包含模型权重、受限音频、说话人身份或识别器输出。详见 [docs/MODELS.md](docs/MODELS.md) 与 [docs/REPRODUCE.md](docs/REPRODUCE.md)。

## 代码结构

`v3` 将最终方案和实验过程分开，读者可以先看主流程，再看实验分支。

| 目录 | 用途 |
|---|---|
| `locator/` | 最终版第一阶段，均值模板、去均值、固定 8 帧滑窗、前 20 个候选 |
| `reranker/` | 最终版第二阶段，文字网格对齐、可选拼音近似、单分支打分 |
| `analysis/` | 候选池评估与前缀首字泄漏检查 |
| `integration/` | Step-Audio-2 特征导出、中间状态缓存（KV Cache）和低显存加载 |
| `experiments/` | 旧实验、时序形状实验、层分析和已淘汰方案 |
| `scripts/` | 真实语料上的对比实验、交叉验证、权重扫描与缓存基准 |
| `tests/` | 最终路径的单元测试；`tests/research/` 保留实验分支测试 |

关键证据文件如下。

- [reports/README.md](reports/README.md)：每个实验报告对应哪一条结论。
- [reports/stage1_cv_speaker.json](reports/stage1_cv_speaker.json)：第一阶段按说话人分组的五折验证。
- [reports/stage2_lambda_sweep.json](reports/stage2_lambda_sweep.json)：重排权重扫描与配对置信区间。
- [reports/lambda_cv_speaker.json](reports/lambda_cv_speaker.json)：先选权重、再在未见说话人上评估的五折验证。
- [reports/prefix_audit_real.json](reports/prefix_audit_real.json)：3244 次真实目标词的前缀首字泄漏检查。
- [reports/shape_gate_cv_fused_l4.json](reports/shape_gate_cv_fused_l4.json)：时序形状在权重 4.0 下未带来稳定收益的结果。
- [reports/stage2_cache_parity.json](reports/stage2_cache_parity.json)：缓存与全序列的数值和排序一致性。
- [reports/cache_benchmark.json](reports/cache_benchmark.json)：延迟与显存基准。

完整结果与适用边界见 [docs/REAL_CORPUS_RESULTS.md](docs/REAL_CORPUS_RESULTS.md)，架构说明见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

## 测试与复现

```bash
python -m compileall -q .
python -m pytest -q
python scripts/lambda_sweep.py --help
```

当前本地检查结果：22 个测试通过，1 个依赖 `torch` 的模型加载守卫测试在未安装运行时的环境中自动跳过。

## 适用范围与限制

- 结果只来自一个 2031 句的私有吴语语料，不能直接外推到所有吴语小片。
- 说话人划分使用文件名规则，正式论文应换用官方说话人或独立语料验证。
- 拼音近似使用普通话转写，并非严格吴语音素模型；它只是一个成本较低的近似保护，不能当作方言音系学结论。
- 合成样例只验证接口，不能作为准确率证据。
- 本项目没有测量误唤醒、远场噪声、功耗和实时端到端延迟，目前仍是研究原型，尚未达到生产级唤醒词系统的要求。
- 中间状态缓存的实测加速为 1.3 到 2.1 倍，具体数值见上表。
- 文本先验校准、去主成分、主成分白化、多原型、现有层融合和固定权重形状融合都没有进入默认路径。

项目采用 Apache-2.0 许可。模型、音频和派生特征遵循各自原始许可。如果这些未带来稳定收益的实验和对齐检查对你有帮助，欢迎引用或复现。
