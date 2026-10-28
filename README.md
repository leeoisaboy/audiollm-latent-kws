<p align="center">
  <a href="https://github.com/leeoisaboy/audiollm-latent-kws">
    <img src="docs/assets/title_banner.zh.png" alt="AudioLLM-Latent-KWS：冻结 Audio-LLM 的方言开放词表关键词检索" />
  </a>
</p>

<p align="center">
  <a href="README.en.md"><img alt="English" src="https://img.shields.io/badge/English-README-2f81f7?style=flat-square"></a>
  <img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-555555?style=flat-square">
  <img alt="Status" src="https://img.shields.io/badge/status-research%20prototype-b8860b?style=flat-square">
</p>

# AudioLLM-Latent-KWS

**只录几段方言指令，就能把新词加入关键词检索，不需要重新训练大模型。**

很多语音设备已经能听懂普通话，却很难低成本覆盖吴语等方言的细分说法。每增加一个词，如果都要补数据、标时间、再微调模型，成本会迅速落到长尾词上。这个项目研究一个更基础的问题：**一个已经训练好、参数不再更新的音频大模型，内部是否已经保存了足够的声学信息，让新词只靠少量注册录音就能被找到？**

项目给出一条两阶段、全程不更新模型参数的检索流程：

1. **先找候选**：把每条语音切成小段，与每个词的平均“声学模板”比较，保留最像的 20 个词。
2. **再看语境**：把候选词放回冻结的大模型中，结合它前面的文字重新打分，选出最终结果。

这不是一个可以直接上线的唤醒词系统，而是一个可复现的研究基线。它的价值不只在于成绩，还在于把两个容易被忽略的问题量化了：**两套时间网格的对齐陷阱**，以及**看起来有用的时序形状信号为什么没有转成最终收益**。

## 先说结论

下面的数字来自 2031 条吴语语音、3244 次目标词出现、293 个词的真实语料实验。

| 我们关心的问题 | 结果 | 怎么读 |
|---|---:|---|
| 正确词有没有进入前 20 个候选？ | **94.7%** | 3244 次目标词中，有 3071 次至少进入前 20；这决定后级重排有没有机会看到正确答案 |
| 只靠声学，目标词排第一的比例？ | **47.4%** | 3244 次目标词中，1538 次直接排第一 |
| 加上语境重排后的默认结果？ | **54.2%** | 3244 次中 1757 次排第一，不需要额外安装拼音库 |
| 再加同音字近似吸附的结果？ | **55.0%** | 3244 次中 1783 次排第一，是推荐组合；5.5 权重下单看这一指标可到 55.1% |

94.7% 是当前候选池的硬上限：剩下的 5.3% 根本没有进入前 20，后级重排再强也无法把它们找回来。“排第一”则按每一次目标词出现计算，同一个词在同一句里出现两次就计两次。

文中还会出现“词级指标”。它先对同一个词的所有出现求平均，再让每个词等权，避免高频词把总体数字带偏。

“同音字近似吸附”是可选分支：语音阅读器可能把“关窗”识成“官窗”。程序会用普通话拼音做一个近似匹配，避免把目标词的首字误留进上下文。

最终推荐组合是：

```text
模型第 2 层声学表示
  + 8 帧（约 640 毫秒）平均模板
  + 注册集全局均值扣除
  + 前 20 个候选
  + 按 40 毫秒文字网格对齐前文
  + 声学分 + 4.0 × 语言模型分
  + 可选：普通话拼音近似吸附
  + 时序形状关闭
```

其中 `4.0` 是语言模型项相对于声学项的权重，声学项本身的权重始终是 1。它不是“越大越好”的魔法数：实测曲线在 4.0 到 5.5 之间很平。只按“排第一”一个指标挑，5.5 的 55.1% 是最高点；4.0 为 55.0%，但词级指标更高（64.5% 对 64.1%）。两者差 0.1 个百分点，落在噪声范围内，因此仓库保留更保守、也更均衡的 4.0。

## 关键发现：两套时间网格会悄悄出错

音频模型每 **80 毫秒**一帧，外部文字阅读器每 **40 毫秒**一个字。两套网格并不重合。若直接把“最近帧”硬切为上下文边界，目标词的第一个字很容易被留在前缀里，相当于提前把答案泄露给重排模型。

我们在 3244 次真实目标词上做了前缀审计：

| 上下文切法 | 前缀中出现目标词首字的比例 | 含义 |
|---|---:|---|
| 按最近帧硬切 | **35.2%** | 超过三分之一的目标词把首字泄露进了上下文 |
| 先吸附到文字字符网格，再切开 | **3.4%** | 泄漏下降到约十分之一 |

这一步只改变“上下文从哪里开始”，不改变模型、候选池或融合权重，却能决定重排是否在利用答案。完整审计见 [reports/prefix_audit_real.json](reports/prefix_audit_real.json)。

另一个值得保留的阴性结果来自“时序形状”。一阶差分（相邻帧的变化方向）单独使用时确实有排序能力，但把它以固定权重加到“声学 + 语言模型”的分数上，没有带来稳定收益：

| 验证方式 | 五折交叉验证平均增益 | 95% 区间 | 结论 |
|---|---:|---:|---|
| 按语音分组 | **+0.17 个百分点** | [-0.20, +0.56] | 区间跨零，不能算稳定收益 |
| 按说话人分组 | **+0.03 个百分点** | [-0.31, +0.40] | 同样跨零 |

所以我们没有把“形状”包装成一个漂亮的新模块，而是保留诊断代码并把默认路径关闭。这个结论对后续工作更有价值：信号存在，不等于固定的线性相加就能兑现。

## 这些选择是怎么验证的

### 候选定位：8 帧均值模板更稳

早期实现固定使用 5 帧（400 毫秒）。在完整语料上改成 8 帧（640 毫秒），再扣除注册集的全局均值后，候选池召回从 92.5% 升到 **94.7%**。

为了避免在测试集上挑参数，我们做了嵌套五折：每折只用训练部分选配置，再在没见过的语音或说话人上评估。

| 比较 | 目标词排第一的变化 | 95% 区间 | 说明 |
|---|---:|---:|---|
| 8 帧 + 去均值 vs. 原 5 帧方案 | **+3.70 个百分点** | [+2.80, +4.61] | 按说话人分组，五折均选择 8 帧 + 去均值 |
| 同一比较，按语音分组 | **+3.51 个百分点** | [+2.62, +4.42] | 5 折中有 4 折选择去均值 |

需要说明：这里的“说话人分组”来自语料文件名中的 `rec_sXX` 规则，不是官方发布的说话人编号；正式论文仍应在官方划分或第二套独立语料上复核。

### 重排权重：1.5 是继承值，不是最优值

旧版本使用 `1.5 × 语言模型分`。它来自更早的实验，并没有在当前语料上重新选择。重新扫描后：

| 比较 | 目标词排第一的变化 | 95% 区间 |
|---|---:|---:|
| 默认 C，权重 4.0 vs. 1.5 | **+1.05 个百分点** | [+0.47, +1.64] |
| C + 拼音近似，权重 4.0 vs. C，权重 4.0 | **+0.80 个百分点** | [+0.37, +1.26] |

这也是仓库把默认值改成 `4.0` 的原因。音素近似分支需要额外安装 `pypinyin`；不安装时，默认的精确网格吸附仍然可用。

## 为什么没有采用那些“看起来更强”的方案

下面这些方向都真实跑过。保留它们的代码，是为了让失败可复现，而不是为了堆功能。

| 方案 | 真实语料上的观察 | 当前决定 |
|---|---|---|
| 按词长自动调窗口、峰值邻域生长 | 目标词排第一 46.5%，池召回 93.4%，低于 8 帧固定窗 | 不进入默认路径 |
| 多原型 / K 均值子模板 | 2 子原型约 45.0%，池召回 91.7%，更弱 | 保留在实验代码 |
| 去主成分（All-but-the-Top）、PCA 白化 | 分别约 39.5% / 43.2%，池召回明显下降 | 保留为阴性对照 |
| 多层分数融合 | 单层 `llm1` 的排第一略高，但池召回从 94.7% 降到 93.4% | 默认仍用第 2 层 |
| 同一候选保留两个起点 | 权重 4.0 下 +0.68 点，弱于拼音近似 | 不进入默认路径 |
| 点互信息（PMI）文本先验校准 | 旧权重 1.5 下 `gamma=1` 为 **-0.96 点**；权重 4.0 下为 **-3.45 点** | 保留为阴性结果 |
| 时序形状门控 | 可部署融合上 +0.17 / +0.03 点，区间均跨零 | 默认关闭 |

这些结果说明：当前系统的瓶颈主要不是“模板不够复杂”，而是**候选池上限、两套时间网格的对齐、以及声学与语义分数的融合方式**。盲目叠加模块很容易在单一指标上得到小幅正数，却失去统计支持。

## 工程加速：缓存有效，但没有宣传得那么夸张

同一句语音的“系统提示 + 音频”前缀在给 20 个候选打分时完全相同。因此先算一次音频前缀并缓存中间状态，再让每个候选只增量计算自己的短文本，是合理的工程优化。

在 H20、GPU0、20 个候选、半精度（fp16）条件下实测：

| 音频长度 | 全序列重算 | 复用缓存 | 加速 | 额外峰值显存 |
|---|---:|---:|---:|---:|
| 1.0 秒 | 0.283 秒 | 0.160 秒 | 1.77 倍 | 174 → 154 MB |
| 3.6 秒 | 0.239 秒 | 0.183 秒 | 1.30 倍 | 127 → 123 MB |
| 6.4 秒 | 0.301 秒 | 0.141 秒 | 2.14 倍 | 192 → 163 MB |

在 200 条对照语音上，缓存与全序列的分数最大差约 0.04 个对数概率单位，排序相关系数不低于 0.997，最终答案在 199/200 条上完全一致。**因此这里只主张 1.3 到 2.1 倍加速，不主张 5 到 10 倍，也不主张显存下降 80%。** 候选只有 1 到 4 个 token，原实现又已经按长度批处理，剩余空间没有想象中大。

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

## 代码怎么读

`v3` 把“最终方案”和“实验过程”分开，避免读者一打开项目就被大量淘汰分支淹没。

| 目录 | 用途 |
|---|---|
| `locator/` | 最终版第一阶段：均值模板、去均值、固定 8 帧滑窗、前 20 个候选 |
| `reranker/` | 最终版第二阶段：文字网格吸附、可选拼音近似、单分支打分 |
| `analysis/` | 候选池评估与前缀泄漏审计 |
| `integration/` | Step-Audio-2 特征导出、中间状态缓存（KV Cache）和低显存加载 |
| `experiments/` | 冻结的旧分支、形状算子、层探针和已淘汰方案 |
| `scripts/` | 真实语料上的消融、交叉验证、权重扫描与缓存基准 |
| `tests/` | 最终路径的单元测试；`tests/research/` 保留实验分支测试 |

关键证据文件：

- [reports/README.md](reports/README.md)：每个实验报告对应哪一条结论。
- [reports/stage1_cv_speaker.json](reports/stage1_cv_speaker.json)：第一阶段按说话人分组的嵌套五折。
- [reports/stage2_lambda_sweep.json](reports/stage2_lambda_sweep.json)：重排权重扫描与配对置信区间。
- [reports/lambda_cv_speaker.json](reports/lambda_cv_speaker.json)：先选权重、再在未见说话人上评估的嵌套五折。
- [reports/prefix_audit_real.json](reports/prefix_audit_real.json)：3244 次真实目标词的前缀首字泄漏审计。
- [reports/shape_gate_cv_fused_l4.json](reports/shape_gate_cv_fused_l4.json)：时序形状在权重 4.0 下的阴性结果。
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

## 边界与不能声称的事情

- 结果只来自一个 2031 句的私有吴语语料，不能直接外推到所有吴语小片。
- 说话人划分使用文件名规则，正式论文应换用官方说话人或独立语料验证。
- 拼音近似使用普通话转写，并非严格吴语音素模型；它是一个便宜的近似保护，不是方言音系学结论。
- 合成样例只验证接口，不能作为准确率证据。
- 本项目没有测量误唤醒、远场噪声、功耗和实时端到端延迟，不是生产级唤醒词系统。
- 中间状态缓存只带来 1.3 到 2.1 倍实测加速，不能写成 5 到 10 倍或显存下降 80%。
- 点互信息校准、去主成分（All-but-the-Top）、PCA 白化、多原型、现有层融合和固定权重形状融合都没有进入默认路径。

项目采用 Apache-2.0 许可。模型、音频和派生特征遵循各自原始许可。若这个负结果与对齐审计对你的工作有帮助，欢迎引用或复现。
