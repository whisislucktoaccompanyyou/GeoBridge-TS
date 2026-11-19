# GeoBridge-TS

依据本目录的《MEDIA_CDARL_大模型伪标签驱动ABC训练方案_v1.md》实现一套系统：中转站离线分割 → 伪标签质检 → 原 B 初始生成器 G0 → 同条件熵正则 OT / 布朗桥终点回归 → GPG → 缓存合成图 → 同一学生的 A/C/U 训练和单 EMA 教师。最终只部署学生。

代码全部位于本目录，可单独复制到 Linux GPU 服务器；不依赖相邻 MEDIA_CDARL 目录。原项目文件没有修改。复用模块及许可证见 [THIRD_PARTY.md](THIRD_PARTY.md)。这里实现的是方案定义的数值桥近似，不是对论文全部求解器的精确复现。

## 先看这里：现在应该改哪个文件

**正式使用以 `config/geobridge_ts.toml` 为主配置。** 数据目录、输出目录、提示词路径、中转站地址、模型、训练设备和训练步数都在这里修改。本说明中的正式命令统一显式指定这个文件。

| 文件 | 用途 | 什么时候修改 | 是否自动读取 |
| --- | --- | --- | --- |
| `config/geobridge_ts.toml` | 正式主配置，控制真实数据、API、模型、训练和推理 | 正式运行前填写服务器路径、设备、预算及API设置 | `pipeline.py` 不写 `--config` 时默认读取它；其他入口要求显式指定 |
| `config/geobridge_ts.example.toml` | 公开仓库使用的完整正式配置模板，密钥为空 | 新检出代码后复制为geobridge_ts.toml，再填写本机路径与密钥 | 不会自动覆盖已有主配置 |
| `config/geobridge_ts.smoke.toml` | CPU模拟检查配置，使用小网络、模拟图和本机模拟API | 通常无需修改，只用来检查代码能否跑通 | 仅在 `--config` 指定它时读取；不会覆盖或合并进正式配置 |
| `config/geobridge_ts.local.toml` | 可选的个人配置副本，当前没有创建，也不是启动必需文件 | 想保留主配置原样时，可自行复制后修改 | 不会自动发现或合并；必须写 `--config config/geobridge_ts.local.toml` |

**正式运行只需修改 `config/geobridge_ts.toml`：地址填写在 `gateway.request_url`，密钥填写在 `gateway.api_key`。** 不需要额外的密钥文件，也不必创建 `.local.toml`。所选TOML中的值就是实际使用的值，不受系统环境变量覆盖。

## 项目文件树与各目录的作用

以下是本机项目文件结构。含真实密钥的geobridge_ts.toml只在本机保留，不提交公开仓库；从GitHub检出时先按第12节从example创建它。`manifests/`、`runs/` 均按使用步骤生成或由你创建，未生成时看不到是正常情况。

```text
GeoBridge-TS/
├── README.md                         # 本使用说明：目录、配置、启动、输出和排错
├── MEDIA_CDARL_大模型伪标签驱动ABC训练方案_v1.md
│                                     # 你的原始方法与实施方案
├── IMPLEMENTATION.md                 # 实现范围、已有离线测试证据和未验证部分
├── THIRD_PARTY.md                    # 复用代码来源及许可证说明
├── requirements.txt                  # Python依赖列表
├── configuration.py                  # 读取TOML、解析路径、读取密钥、校验与脱敏
├── .gitignore                        # 忽略真实配置、运行产物和Python缓存
├── config/                           # 参数配置目录
│   ├── geobridge_ts.toml              # 正式主配置，主要修改这个
│   ├── geobridge_ts.example.toml      # 公开仓库中的空密钥正式模板
│   └── geobridge_ts.smoke.toml        # CPU小规模模拟检查配置
├── prompts/                          # 发送给外部大模型的提示词
│   ├── vessel_segmentation.user.txt   # 你提供的原文存档，保留原十图实验内容
│   └── vessel_segmentation.api.txt    # 已授权改成的逐张API版本，默认实际发送这个
├── scripts/                          # 从命令行启动的入口
│   ├── build_manifest.py             # 索引原图、读取人工确认的映射表并建立清单
│   ├── pipeline.py                   # API、质检、数据准备、训练及合成的统一入口
│   ├── evaluate_student.py           # 在真实val/test标注图上评估学生
│   └── predict_student.py            # 用训练好的学生分割一张新图
├── pseudo_labels/                    # 外部大模型伪标签模块
│   ├── __init__.py                   # Python包初始化文件
│   ├── gateway.py                    # 请求体、认证、HTTP、限流、重试和批量调用
│   ├── parsers.py                    # 把明确格式的响应解码为真正的二值掩膜
│   ├── quality.py                    # 检查尺寸/像素/FOV/前景比例，保存掩膜和叠加图
│   ├── jobs.py                       # SQLite任务账本和进程锁，记录状态并支持续跑
│   └── mock_server.py                # 仅本机使用的模拟API，供功能测试使用
├── data/                             # 数据读取、配对和几何条件
│   ├── __init__.py
│   ├── fundus_manifest.py            # 清单契约、文件哈希、受试者划分和泄漏检查
│   ├── paired_pseudo.py              # 同步缩放/填充/翻转、权重、外观增强和采样器
│   ├── geometry.py                   # 血管掩膜的边界、骨架和有符号距离图
│   └── synthetic_pairs.py            # 核验合成图Y始终对应输入B的同一份P
├── model/                            # 模型结构与生成算法
│   ├── __init__.py
│   ├── segmentation_core.py          # 可微学生S、G0、判别器构建和参数独立性检查
│   ├── ema_teacher.py                # EMA教师、缓冲复制及对齐后的不确定性目标
│   ├── ot_bridge/                    # 新增OT/SB/GPG模块
│   │   ├── __init__.py
│   │   ├── transport.py              # 同条件代价矩阵、log-Sinkhorn和端点配对采样
│   │   ├── bridge.py                 # 清晰终点U-Net、桥状态、漂移和逐步SDE生成
│   │   └── gpg.py                    # 固定教师，通过图像梯度纠正血管几何
│   └── legacy/                       # 从原MEDIA_CDARL复用的必要骨干，本地副本
│       ├── __init__.py
│       ├── unet.py                   # 原去噪特征U-Net
│       ├── discriminators.py         # 原判别器
│       └── spade/                    # 原SPADE生成/分割骨干
│           ├── __init__.py
│           ├── generator.py          # SPADE生成器主体
│           ├── architecture.py       # SPADE残差块
│           ├── normalization.py      # 条件归一化
│           └── base_network.py       # 基础网络类
├── training/                         # 模型怎样训练、保存和评估
│   ├── __init__.py
│   ├── stages.py                     # 各阶段调度、优化、缓存合成和验证选模
│   ├── losses.py                     # 有效区域内BCE/Dice、一致性与几何目标
│   ├── checkpoints.py               # 保存/恢复模型、优化器、AMP、随机数及采样状态
│   ├── evaluation.py                # 原图坐标指标、CSV、汇总和对比图
│   └── smoke.py                     # 创建模拟数据并跑通隔离的小流程
└── tests/                            # 离线验收代码，通常不需要手工修改
    ├── conftest.py                   # 测试公共配置和临时模拟数据
    ├── test_gateway.py               # API请求、响应解析、重试、续跑和密钥脱敏
    ├── test_math_and_gradients.py    # OT、桥噪声、GPG、EMA、坐标和数据划分
    ├── test_end_to_end.py            # 完整小流程及中断恢复一致性
    ├── test_legacy_and_metrics.py    # 原骨干梯度与评估指标
    └── fixtures/
        └── binary_png_response.json # 静态模拟响应样例，不是真实中转站结果
```

| 目录 | 里面存什么 | 你通常需要做什么 | 是否直接启动里面的代码 |
| --- | --- | --- | --- |
| `config/` | 本机正式配置、空密钥example模板与模拟配置 | 主要编辑本机正式TOML | 由入口的 `--config` 读取 |
| `prompts/` | 原提示词和实际API提示词 | 需要改变提示词时编辑API版，并记录新数据版本 | 不直接执行文本 |
| `scripts/` | 四个命令行入口 | 按下面顺序执行命令 | **从这里启动** |
| `pseudo_labels/` | 中转站调用、掩膜解析与质检代码 | 通常不修改；服务协议特殊时才改适配器 | 通过pipeline调用 |
| `data/` | 数据处理代码 | 原始眼底图无需放进这个目录，路径在TOML中配置 | 通过入口调用 |
| `model/` | 网络定义、EMA和桥/GPG算法 | 通常不修改，结构参数在TOML中设置 | 通过训练或预测入口调用 |
| `training/` | 训练循环、保存恢复、评估与模拟流程 | 通常不修改 | 通过入口调用 |
| `tests/` | 验收测试和模拟响应样例 | 改动代码后可运行测试 | 使用pytest统一运行 |
| `manifests/`（生成） | 输入清单、样本/受试者/划分对应表 | 填写和核实CSV映射，不把眼底原图复制进这里 | 清单由build_manifest生成 |
| `runs/`（生成） | API结果、检查点、合成图、日志与评估 | 查看结果；正式结果不能当作临时缓存随意删除 | 由程序写入 |

所有 `__init__.py` 都是Python包文件，部分为空，方便代码互相导入；保留即可。大部分 `.py` 是内部模块，平时只需要操作主配置、提示词和 `scripts/` 的入口。

## 第一次启动的顺序

本项目按阶段启动，没有“直接执行一个main.py就自动付费分割并完成所有训练”的模式。正式流程如下：

```text
填写正式TOML和密钥
    → 建立原图清单并确认受试者划分
    → 单张API检查 → 核实响应格式 → 批量分割和伪标签质检
    → prepare-data
    → train-g0
    → warmup-student → init-teacher
    → train-bridge
    → synthesize-cache → 查看合成质量
    → train-student-teacher
    → val/test评估
    → predict_student.py分割新图
```

其中A是学生的人工真值监督路径，C是**同一个学生**的合成图监督路径，U是未标注真实图一致性路径；不是三个独立分割模型。T是该学生的EMA教师。完整B包含G0、OT/SB和GPG。

如果只是先确认安装是否正常，可以跳到第2节运行CPU模拟检查，不需要真实密钥、数据目录或正式训练预算。模拟检查会再次生成明确标记的测试产物，阅读README不会自动生成这些文件。

## 1. 当前状态

- 已实现全部阶段入口、配置、请求账本、严格解析、几何、G0、OT/SB、GPG、缓存、A/C/U、EMA、恢复、评估和学生推理。
- 已运行 28 项离线测试，以及独立完整模拟小流程。模拟流程只访问本机 HTTP 服务，产物带 `mock=true`，正式流程禁止混用。
- 已于2026-11-19对 `21_training.png` 完成1次真实识图请求：HTTP200，约5.6秒，响应模型名为gpt-6.1-sol，能描述眼底、血管、左侧视盘和黑色背景。没有生成真实伪标签，也没有进行正式训练或验证医学效果。
- 本机 Python 3.12.10、PyTorch 2.11.0+cpu；未验证 CUDA / AMP、服务器吞吐、384 画布显存及长训收敛。
- 用户已在主配置填写密钥。根地址返回网页，真实Responses端点已确认是 `https://sub2api.aiproxyapp.top/v1/responses`，主配置已修正；代码仍不自动拼接路由。
- 此次只验证图片可见性和描述能力，没有测试完整分割提示词及像素级掩膜。`response_profile` 仍未核实，`profile_verified` 保持false，不能据此直接批量生成训练伪标签。测试报告和原始响应已按用户要求清理；验证摘要保留在IMPLEMENTATION.md中。

## 2. 安装

要求 Python 3.11+（使用标准库 `tomllib`）。先按服务器 CUDA 版本从 PyTorch 官方安装对应的 torch，然后在本目录执行：

GitHub首次检出的目录没有真实主配置，先按第12节把example复制为主配置；已有主配置时直接保留，不用再次复制。离线测试也可以直接使用仓库中的example，不要求填写真实密钥。

```bash
python -B -m pip install -r requirements.txt
python -B -m pytest -p no:cacheprovider -q
python -B scripts/pipeline.py smoke-run --config config/geobridge_ts.smoke.toml
```

最后一条会在独立 `runs/geobridge_ts/smoke_时间_随机ID/` 下创建少量模拟图、模拟伪标签、每阶段 2 次更新、合成缓存与测试输出。它不会使用配置中的远程中转站或真实数据，不能作为模型效果结果。也可以对任一阶段传 `--smoke-run`，其含义仍是运行这套完整隔离模拟流程。

上述命令需要在项目根目录执行。Windows PowerShell先运行：

```powershell
Set-Location -LiteralPath 'D:\program\CTA大模型_分割\分割图像——医学图像\GeoBridge-TS'
```

Linux服务器先 `cd` 到你上传后的GeoBridge-TS目录，再执行同样的Python命令。如果服务器使用 `python3`，将命令开头的 `python` 换成 `python3`。

本说明统一加 `-B`，用于避免本项目生成 `.pyc` / `__pycache__`；pytest加 `-p no:cacheprovider` 避免生成 `.pytest_cache`。这些参数不改变训练算法。此前清理掉的模拟结果属于已完成检查的产物；新的模拟检查仍会生成新的 `runs/geobridge_ts/smoke_*/`，其中 `smoke_report.json` 是这次功能检查报告。

## 3. 配置和密钥

### 3.1 正式主配置与模拟配置怎么选择

主要修改 [config/geobridge_ts.toml](config/geobridge_ts.toml)，每个参数都附有中文注释。`--config` 指定哪一个TOML，这一次运行就读取哪一个，**不会把config目录的多个文件一起读取，也不会自动合并它们**。

`pipeline.py` 不传 `--config` 时，默认读取正式主配置。清单构建、评估和预测入口要求明确传 `--config`。为避免用错，本说明的命令全部显式指定。

模拟配置的 `mock=true`、`backend="tiny"`、32画布和少量更新是检查代码路径用的；不能拿它训练真实模型。`smoke-run` 还会在内存中覆盖若干参数，强制使用临时模拟数据、本机HTTP、虚拟认证和独立输出，因此模拟TOML不是正式训练参数调优入口。

### 3.2 请求地址和密钥直接填在TOML中

打开 `config/geobridge_ts.toml`，找到已有的 `[gateway]`，修改这些已有字段：

```toml
[gateway]
request_url = "https://sub2api.aiproxyapp.top/v1/responses"
api_key = "这里替换为你的真实密钥"
model = "gpt-6.1-sol"
protocol = "openai_responses"
```

上面只展示需要关注的字段，直接修改文件中已有的值，不要在文件末尾重复追加第二个 `[gateway]`。截图里的密钥被隐藏，原实现未杜撰密钥；之后用户已自行填写。真实值仅保存在你本机所选配置中。

地址和密钥仅从命令行 `--config` 选中的TOML读取，不再读取额外密钥文件，也不再使用系统环境变量覆盖。`gateway.base_url` 只是地址记录，程序实际发请求使用 `gateway.request_url`。

实际主配置的 `request_url` 已改为测试成功的完整端点 `https://sub2api.aiproxyapp.top/v1/responses`，而根地址返回网页。程序不自动追加路径。

运行快照、响应元数据和检查点仍会脱敏，不改变主配置中的真实密钥。按用户最终要求，真实配置不上传，公开仓库只包含密钥为空的example。纯本地训练如果已有合格伪标签，不要求API密钥。

### 3.3 路径怎么填写

所有相对配置路径都相对 TOML 文件，不依赖启动命令时的工作目录。清单中的相对路径统一相对 `paths.data_root`。服务器上的数据根目录、所有源目录、清单、提示词、输出目录均可在配置中修改。

例如主配置位于 `GeoBridge-TS/config/`：`prompt_file="../prompts/vessel_segmentation.api.txt"` 对应项目根目录下的prompts文件。Linux服务器可把数据源配置为 `/data/retina/...`；Windows的TOML路径可使用 `D:/dataset/...`，或单引号字面量路径，避免双引号字符串内反斜杠的转义问题。

| 字段 | 应该指向什么 | 是你填写还是程序生成 |
| --- | --- | --- |
| `sources.unlabeled_images` | Kaggle优质训练候选原图目录 | 填已有目录 |
| `sources.unlabeled_fov` | 这些原图对应的有效眼底视野掩膜目录 | 填已有目录 |
| `sources.labeled_images` | 有人工标注的原图目录 | 填已有目录 |
| `sources.labeled_masks` | 人工血管标注目录 | 填已有目录 |
| `sources.labeled_fov` | 人工标注数据对应的FOV目录 | 填已有目录 |
| `sources.index_output_csv` | 自动索引生成的空映射模板位置 | 程序生成，之后填写内容 |
| `sources.metadata_csv` | 已核实受试者、划分和配对关系的CSV | 你填写后保存 |
| `paths.input_manifest` | 后续各阶段读取的原图JSONL清单 | 清单构建器生成 |
| `paths.accepted_manifest` | 仅包含合格外部伪标签的JSONL清单 | API/质检阶段生成 |
| `paths.run_dir` | 本次实验所有训练与生成产物的运行目录 | 填输出位置，程序创建目录 |
| `paths.prompt_file` | 实际发给外部模型的UTF-8提示词 | 默认逐张API版，可明确修改 |
| `inference.input_image` | 将来需要分割的一张新真实图 | 推理前填写 |
| `inference.output_dir` | 新图预测结果输出目录 | 填位置，推理时创建 |

FOV表示“有效眼底视野”，血管掩膜表示“哪里是血管”，两者不是同一文件。当前默认要求提供FOV，尚未实现从原图自动提取FOV。

### 3.4 其他配置段与首次必填项

| 配置段 | 内容及必填项 |
| --- | --- |
| `paths` | 数据根目录、原图清单、accepted 清单、运行输出和提示词 |
| `sources` | Kaggle 原图/FOV、人工原图/标注/FOV目录、受试者及文件映射 CSV；清单构建器使用 |
| `gateway` | 地址、协议、模型、密钥、明确响应字段、极性、限流、总请求预算和重试策略 |
| `quality` | 前景占比和连通分量筛查；不修改血管形状 |
| `data` | 本地画布、FOV要求、模拟数据准入；正式流程不得设置 allow_mock |
| `model` | 原 UNet/SPADE 参数、独立桥网络参数、原 G0 扩散时间和 schedule |
| `training` | 设备、各阶段步数和学习率、AMP、检查点频率、A/C/U权重、EMA与不确定性筛选 |
| `bridge` | 同条件候选数、ε、代价归约、Sinkhorn容差/迭代、δ、SDE步数 |
| `gpg` | 调度、投影迭代、步长、近端权重、变化上限和κ上限 |
| `synthesis` | 种子、参考图策略、人工查看后的图像质量准入开关 |
| `evaluation` | 从验证阶段固定的二值阈值、使用的学生检查点 |
| `inference` | 新真实图、预测输出目录、可独立搬迁的学生检查点 |

`g0_steps / warmup_steps / bridge_steps / student_steps` 默认都是 0，必须填正整数才会开始对应训练。没有替用户编造训练预算。`ε=0.01`、384画布、50桥步等都是待标定研究起点，不是已验证最优值。

首版每类 batch=1、num_workers=0。A/C/U分别反向累计到同一个学生，然后一次更新；同步采样器便于完整恢复。正式网络 backend=legacy，tiny 仅用于模拟测试。独立的 `config/geobridge_ts.smoke.toml` 提供CPU小规模配置，mock模式禁止访问远程中转站。各训练阶段生成 `阶段.models.json`，记录模型参数量、可训练参数量、张量约定与设备；实际峰值显存与耗时在training.jsonl中记录。

在 `prepare-data` 前确定本次运行的科学配置。修改数据、模型、预算或桥参数应使用新的 `run_dir`。`quality_approved` 人工准入开关与 `inference` 路径不改变科学指纹，可以稍后调整。训练恢复只适用于原运行配置；部署推理允许搬迁路径，但检查模型结构和画布。

正式开始前至少确认：源目录与清单路径、可靠的subject_id及train/val/test划分、可用设备、四个阶段的正整数训练步数。API批量处理前还须确认真实密钥、完整请求地址、实际响应字段和有限请求预算。模拟配置里的2次更新不能当作正式训练预算。

### 3.5 可选的个人配置副本

如果你想保留主配置原样，也可以复制为 `config/geobridge_ts.local.toml`，之后所有命令都改用 `--config config/geobridge_ts.local.toml`。该副本要包含完整配置，不是只写差异的补丁。程序不会自动把local文件与主配置合并。当前没有替你创建个人副本；直接使用主配置即可。

## 4. 提示词

- `prompts/vessel_segmentation.user.txt`：用户原文件逐字节保留，含原十张 DRIVE 图实验要求。
- `prompts/vessel_segmentation.api.txt`：经用户授权的逐张 API 版本。保留原视觉判断、细血管、禁止自动识别和标注隔离原则；替换固定十图、Windows路径和调用端评分要求。
- API版要求只返回完整 JSON 的非压缩列优先 RLE：`width/height/order="F"/counts`。从背景游程开始，第一像素是血管时第一个计数为0。

代码按文件原文发送，不运行模型返回的代码，不根据文本自行造掩膜，也不会在分割阶段把人工 GT 发给中转站。API版的输出契约不表示该模型必然有能力执行；如果只得到文字或不完整RLE，将停止准入并保留失败状态。

## 5. 建立清单和划分

manifest就是“样本与文件对应清单”，JSONL表示每一行是一个样本的JSON对象。它把原图、血管标签、FOV、受试者和划分绑定起来，后续代码按这个对应关系读取，不靠两个目录各自排序来凑配对。

先填写 `sources` 的原图目录。索引只读取图像及文件哈希，不猜受试者或划分：

```bash
python -B scripts/build_manifest.py --config config/geobridge_ts.toml --index
```

生成 `sources.index_output_csv`。填写 `sample_id, subject_id, split, image_pool, image_path, gt_mask_path, fov_path, annotation_source`，另存到 `sources.metadata_csv`。`image_pool` 为 `labeled/unlabeled`，文件相对对应 sources 目录，或直接填绝对路径。`sample_id` 为唯一ASCII字母、数字、下划线、短横线、点组合。`split` 必须显式写 train/val/test，双眼等属于同一受试者时必须同划分。未知受试者映射需要先核实，不能把每张图自动当作不同病人。

CSV列含义与例子：

| 列 | 含义 | 示例 |
| --- | --- | --- |
| `sample_id` | 这张图的唯一标识 | `patient01_left` |
| `subject_id` | 真实受试者标识，同一人的多张图使用相同值 | `patient01` |
| `split` | 该图用于训练、验证或最终测试 | `train`、`val`、`test`之一 |
| `image_pool` | 原图目录属于哪一类 | Kaggle图写`unlabeled`，人工标注图写`labeled` |
| `image_path` | 相对对应 `sources.*_images` 的原图文件名，或绝对路径 | `21_training.png` |
| `gt_mask_path` | 相对 `sources.labeled_masks` 的人工血管标签文件名，或绝对路径 | `21_manual1.gif`；无人工标注时留空 |
| `fov_path` | 相对该pool的 `sources.*_fov` 的视野掩膜文件名，或绝对路径 | `21_training_mask.gif` |
| `annotation_source` | 人工标注来自哪个数据集/标注版本 | `DRIVE_1st_manual`；无人工标注时可留空 |

索引命令只帮你列图像，不会自动填写subject_id、划分或猜掩膜文件名。确认CSV后，再执行下一条构建命令。原图尺寸不需要手工填CSV，程序读取图像后自动记录。

```bash
python -B scripts/build_manifest.py --config config/geobridge_ts.toml
```

清单构建器填入原图哈希、EXIF统一方向后的宽高及坐标系，检查跨划分受试者/重复图泄漏。它不会自行采用20/10/10，也不覆盖已有清单。可自行提供满足同一契约的JSONL，每行示例：

```json
{"sample_id":"patient01_left","subject_id":"patient01","split":"train","image_path":"images/p01.png","image_sha256":"实际SHA256","width":565,"height":584,"coordinate_frame":"exif_transposed_original","fov_path":"fov/p01.png","gt_mask_path":"manual/p01.png","annotation_source":"人工标注来源"}
```

无人工标注的图不写 gt 字段。FOV及血管掩膜应在统一方向的原图坐标上，是单通道0/255图。没有FOV时不会以血管掩膜代替FOV；默认要求提供。标签权重图可通过 `label_weight_path` 指定，0代表未知像素、255代表权重1；未提供时使用明确的统一有效视野权重。

验证和测试图只能用于各自评估。B 的源端、目标端与参考池，以及 U 分支都只使用 train。

## 6. 中转站单张核查与批量运行

```bash
python -B scripts/pipeline.py api-probe --config config/geobridge_ts.toml --dry-run
python -B scripts/pipeline.py api-probe --config config/geobridge_ts.toml --live --limit 1
```

第一条校验提示词、请求结构、PNG MIME、原尺寸和URL，不联网。第二条显式发送一张训练图，受 max_requests 限制。若地址返回网页或不支持的响应，会报告不准入；根据服务文档修改完整路由，不自动试探多个收费端点。

在 `run_dir/pseudo_labels/<job_key>/response.json` 查看单张脱敏真实响应。确认能获得真正的像素结果后，设置固定解析配置，例如**仅当实际响应符合时**：

```toml
response_profile = "rle"
response_field = "output.0.content.0.text"
profile_verified = true
```

其他明确支持的profile：`binary_png_base64`、`binary_png_url`、`pixel_array`。字段是精确的点分路径，不递归搜图。PNG只接受单通道二值无损文件，拒绝叠加图、任意拉伸、未知坐标、错误尺寸、非二值和损坏文件。COCO压缩RLE、多边形及专有文件引用还未实现，会显式失败；不伪装成已支持。

核实后可以离线重解析刚才的probe，不为同一请求再付费：

```bash
python -B scripts/pipeline.py pseudo-qc --config config/geobridge_ts.toml
```

该命令仅用已保存响应/掩膜，按请求签名核对并导出 accepted。URL下载格式须在live probe中取回产物；QC不联网。下载只允许配置的主机，禁止自动重定向，不转发API认证。

确认少量结果的坐标、极性、细血管和叠加图后，设置有限 `gateway.max_requests`（包含每次重试），显式给出批量样本数：

```bash
python -B scripts/pipeline.py api-segment --config config/geobridge_ts.toml --limit 10 --resume
python -B scripts/pipeline.py pseudo-qc --config config/geobridge_ts.toml
```

SQLite账本区分pending/inflight/received/decoded/accepted/needs_review/failed/ambiguous。成功产物按哈希复用；重复执行默认也复用，`--resume`用于明确表达续跑。429与临时服务错误有上限退避；401/403/参数错误停止。超时/崩溃后默认ambiguous，不自动重发，`retry_ambiguous=true` 才允许后续显式运行时再试；未确认幂等支持时重发可能重复计费。认证修正后重试失败任务可显式设置 `retry_failed=true`。费用缺少计价依据时为unknown，usage有返回就记录。

accepted只表示程序格式/质量筛查通过，不等于医学标注准确。待核查结果不会进入B/C，不自动补线、闭运算、删支或平滑。变更模型、提示词或预处理会形成新版本；不同accepted版本应放在新的run_dir，不覆盖旧P。

## 7. 正式本地阶段

| 阶段命令 | 做什么 | 开始前需要 | 主要输出 |
| --- | --- | --- | --- |
| `prepare-data` | 固定本次数据/配置版本，核验配对并生成几何条件 | 原图清单、accepted伪标签、FOV | prepared.json、清单快照、geometry/ |
| `train-g0` | 用真实图/P训练初始图生成器和D | 训练划分的原图/P配对，正整数g0_steps | checkpoints/g0_last.pt |
| `warmup-student` | 用人工训练标注预热唯一学生S | 人工train、独立人工val、正整数warmup_steps | checkpoints/warmup_last.pt |
| `init-teacher` | 从当前预热学生复制初始化EMA教师T | 完成的warmup_last.pt | checkpoints/teacher_init.pt |
| `train-bridge` | 固定G0，训练同条件OT/SB终点预测网络 | 完成的G0、teacher_init、原图/P配对、正整数bridge_steps | checkpoints/bridge_last.pt |
| `synthesize-cache` | 固定G0/桥/T，逐步生成并执行GPG，保存最终Y/P对 | 完成的G0/桥/T、不同训练受试者的参考图 | synthetic/图像、逐图元数据和manifest |
| `train-student-teacher` | 以A/C/U三路共同更新一个S，再EMA更新T | 完整合成缓存、人工train/val、真实训练图、质量准入和正整数student_steps | student_last.pt、student_best.pt |

每条命令只完成对应阶段；上一条结束后再运行下一条。G0与学生预热本身可以分别准备，桥阶段须等待两者及教师初始化完成。正式命令不会因有缺失检查点而用随机模型代替。

配置齐备且已有accepted清单后，按顺序执行。各阶段不要求API密钥：

```bash
python -B scripts/pipeline.py prepare-data --config config/geobridge_ts.toml
python -B scripts/pipeline.py train-g0 --config config/geobridge_ts.toml
python -B scripts/pipeline.py warmup-student --config config/geobridge_ts.toml
python -B scripts/pipeline.py init-teacher --config config/geobridge_ts.toml
python -B scripts/pipeline.py train-bridge --config config/geobridge_ts.toml
python -B scripts/pipeline.py synthesize-cache --config config/geobridge_ts.toml
```

G0复用原去噪UNet和SPADE初始合成路径，采用配对重建、LSGAN与去噪项；独立判别器D。S是参数独立的原UNet+SPADE实例，在t=0提取特征，输出转换为[0,1]概率。没有自动加载任何未知旧权重，没有把旧refinement网络改名当桥。

同一个P内至少2个源候选、2个保结构外观目标候选。OT用float64 log-Sinkhorn、0.5×sum平方代价和ε；选择mean代价时熵系数同步除以坐标数。按软计划采样一对真实端点，不把终点加权平均成模糊图。非收敛明确失败，不静默换匹配策略。若正式ε导致极低熵、收敛缓慢，应标定ε/候选而非把失败输出当训练数据。

桥以x_t、x0、4通道几何和独立时间嵌入预测清晰终点；训练采用方案的布朗桥中间状态与终点MSE。完整B推理只有x0/几何/噪声，不读取目标x1。正式缓存参考照片来自**不同训练受试者**；若缺少合格参考，停止而不用当前目标X顶替。

GPG在清晰终点上通过固定教师优化独立图像变量，有近端项、逐像素变化上限、有效区域约束与回溯；反馈到后续SDE，末端再次纠正。无需保存整条采样反向图。缓存记录P、G0/桥/教师指纹、参考样本、种子、letterbox、参数和GPG诊断。保存到原图画布后，C仍配对同一份原始P。

检查真实生成外观、旧血管残留、细支、GPG前后变化等，充分训练并确认后，将 `synthesis.quality_approved=true` 再执行：

```bash
python -B scripts/pipeline.py train-student-teacher --config config/geobridge_ts.toml
```

这只是人工图像质量准入开关，不改变缓存版本。它不能自动证明G0/SB已经收敛。教师只在学生真正更新后EMA参数，缓冲直接复制；AMP溢出跳步不计EMA。A/C/U逐路累积同一学生梯度，B固定。教师通过不同弱增强和逆翻转对齐估计目标与方差，学生用适度外观强增强；无可靠像素时U为零。

中断后对相应命令增加 `--resume`，例如：

```bash
python -B scripts/pipeline.py train-bridge --config config/geobridge_ts.toml --resume
python -B scripts/pipeline.py synthesize-cache --config config/geobridge_ts.toml --resume
python -B scripts/pipeline.py train-student-teacher --config config/geobridge_ts.toml --resume
```

同步采样器、Python/NumPy/Torch随机数、模型、优化器、常数学习率调度状态、AMP、EMA历史和依赖指纹会恢复。CPU小网络的中断恢复已验证一致；不承诺未测CUDA内核跨设备逐位相同。检查点仅加载可信本地文件；当前新流程没有通用旧权重迁移入口。

## 8. 真实图评估与部署

```bash
python -B scripts/evaluate_student.py --config config/geobridge_ts.toml --split val
python -B scripts/evaluate_student.py --config config/geobridge_ts.toml --split test
```

学生按独立验证集Dice选择 `student_best.pt`。阈值默认0.5，需在正式训练配置冻结前确定；本实现不自动在测试集调阈值。若研究要求额外阈值搜索，应在独立验证流程确定后记录，不修改测试结果来追求更高分。

评估将概率映射回原尺寸，在明确FOV内计算Dice/IoU/Recall/Precision/FNR/FDR/FPR、TP/FP/FN/TN、梯形积分PR-AUC、clDice及连通分量。主分100×Dice，逐图等权均值；零分母是JSON null/CSV空值，并报告各均值的有效样本数。clDice和连通分量不等于完整临床拓扑验证。输出概率、二值PNG、逐图CSV、汇总JSON和原图/预测/标注/叠加/误差五联图。

只需学生即可部署。配置 `inference.input_image/output_dir/student_checkpoint` 与推理设备后：

```bash
python -B scripts/predict_student.py --config config/geobridge_ts.toml
```

不读取训练manifest，不依赖B、D、T或API。输出原图坐标的 `.mask.png` 和 `.prob.npy`。

## 9. 实现地图与验收

| 目录/文件 | 作用 |
| --- | --- |
| `configuration.py`、`config/` | TOML、中文注释、路径解析、秘密脱敏、配置指纹 |
| `pseudo_labels/` | Responses/Chat/custom适配、HTTP、SQLite、解析/QC、离线模拟服务 |
| `data/` | manifest、划分/哈希、统一letterbox、四通道几何、严格合成对 |
| `model/segmentation_core.py` | 可微S、原B源端G0、模型独立性检查 |
| `model/legacy/` | 原项目必要模块的本地副本，保留许可 |
| `model/ot_bridge/` | 条件OT、终点U-Net、布朗桥/SDE、GPG |
| `model/ema_teacher.py` | 独立教师、参数EMA/缓冲复制、对齐不确定性 |
| `training/` | 全阶段、保存恢复、损失、评估、隔离模拟 |
| `scripts/` | 清单构建、流水线、评估、学生部署入口 |
| `tests/` | 28项离线验收，含静态PNG响应fixture和动态本机Responses fixture |

详细测试证据与边界见 [IMPLEMENTATION.md](IMPLEMENTATION.md)。

## 10. 运行后到哪里找结果

下面以主配置默认的 `paths.run_dir="../runs/geobridge_ts/default"` 为例。路径相对config目录解析，因此实际在项目根目录的 `runs/geobridge_ts/default/`。你修改run_dir后，输出会跟随改变。此树是**运行后的示意**，不是当前目录已经拥有的正式训练结果。

```text
runs/geobridge_ts/default/
├── api_jobs.sqlite3                  # API任务账本：成功、失败、不确定状态及用量等
├── api.lock                          # API进程锁文件
├── training.lock                     # 训练进程锁文件
├── pseudo_labels/
│   └── <job_key>/                    # 一个明确请求与解析版本对应一个目录
│       ├── response.json             # 已脱敏的服务响应，真实probe先检查这个
│       ├── transform.json            # 方向、原尺寸和请求画布信息
│       ├── mask.png                  # 成功解析后才产生的候选二值血管掩膜
│       └── overlay.png               # 候选血管叠加原图，供检查坐标与形状
├── accepted.jsonl                    # 只包含合格伪标签；位置由accepted_manifest控制
├── prepared.json                     # 当前数据/配置版本及各类样本数量
├── config.sanitized.json             # 脱敏配置快照，用于追溯，不是下次启动主配置
├── manifest.snapshot.jsonl           # 本次实际使用的样本清单快照
├── geometry/
│   └── <sample_id>.npy               # P、边界、骨架、有符号距离四通道数组
├── checkpoints/
│   ├── g0_last.pt                    # G0/D训练恢复状态
│   ├── warmup_last.pt                # 预热学生恢复状态
│   ├── teacher_init.pt               # 初始化学生/教师及来源
│   ├── bridge_last.pt                # 桥训练恢复状态与固定依赖模型
│   ├── student_last.pt               # 最近保存的正式学生教师训练状态，续跑用
│   └── student_best.pt               # 验证Dice最优的正式学生教师检查点，默认评估/推理用
├── train-g0.models.json              # 对应阶段参数量、设备、画布和张量说明
├── warmup-student.models.json        # 其他训练阶段也有自己的 *.models.json
├── train-bridge.models.json
├── train-student-teacher.models.json
├── training.jsonl                    # 各阶段逐步损失、OT/可靠区诊断、耗时与显存
├── synthetic/
│   ├── manifest.jsonl                # 严格绑定最终Y、原P、参考样本、种子与版本
│   ├── <sample_id>.png               # 完整B的最终合成图Y，不是血管二值掩膜
│   └── <sample_id>.json              # 生成参数、模型指纹、坐标变换、GPG前后诊断
└── evaluation/
    ├── val/                          # 验证集评估命令的输出
    └── test/                         # 测试集评估命令的输出
        ├── metrics.csv               # 每张图的指标和混淆矩阵
        ├── summary.json              # 等权均值、阈值、有效样本数和零分母说明
        ├── <sample_id>.png           # 学生预测二值掩膜，回到原图尺寸
        ├── <sample_id>.prob.npy      # 原图坐标的浮点血管概率
        └── <sample_id>.comparison.png # 原图/预测/人工标注/叠加/像素误差五联图
```

新图预测的输出由 `inference.output_dir` 单独控制，默认是 `runs/geobridge_ts/predictions/`，每张输出 `<原图名>.mask.png` 和 `<原图名>.prob.npy`。模拟检查输出在另一个 `smoke_时间_随机ID/` 目录，不与正式default目录混用。

文件类型说明：

| 类型 | 含义 | 是否应保留 |
| --- | --- | --- |
| `.py`、`__init__.py` | Python源码和包初始化 | 保留，多个模块互相依赖 |
| `.toml` | 参数配置 | 保留，正式与模拟分别使用 |
| `.txt`（提示词）、`.md` | 提示词、方案与说明 | 保留需要的版本，许可证说明同时保留 |
| `.jsonl`、`.json`、`.csv` | 数据配对清单、元数据、诊断或指标表 | 正式运行产物需保留以便追溯和续跑 |
| `.png` | 原图、血管掩膜、合成图或对比图，按目录区分 | 正式数据和结果需保留；不要仅按扩展名删除 |
| `.npy` | NumPy数值数组，例如几何条件或概率图 | 运行数据，不是Python缓存 |
| `.pt` | PyTorch模型检查点，可能含优化器、教师与随机状态 | 正式模型需保留，恢复训练依赖它 |
| `.sqlite3` | API任务账本 | 正式API任务需保留，用于续跑和避免重复请求 |
| `.pyc`、`__pycache__/`、`.pytest_cache/` | 可再生的Python/测试缓存 | 可清理；README命令使用-B避免再生成Python字节码 |

## 11. 常见问题与启动前检查

| 现象/问题 | 应该检查什么 |
| --- | --- |
| “到底改哪个配置？” | 正式改config/geobridge_ts.toml；只在模拟命令里使用smoke.toml；两份不会自动合并 |
| “API密钥缺失？” | 在命令实际选中的TOML的[gateway]中填写api_key，确认不是改了另一份配置 |
| “改了地址却还用旧地址？” | 检查命令--config是否指向你修改的TOML，实际使用gateway.request_url而不是base_url |
| “为什么没有manifests或runs目录？” | 尚未生成是正常情况。按清单构建和对应阶段命令启动后创建 |
| “dry-run为什么仍提示缺少清单？” | dry-run需要读取真实清单中的一张训练图来验证请求结构，只是不联网；完全无数据时先用smoke-run |
| “缺少FOV/人工验证集？” | 核实CSV对应关系，FOV与人工血管标注不同；学生训练须有独立人工val数据 |
| “训练预算尚未填写？” | 四个*_steps默认0，分别改成你确定的正整数；不是改smoke.toml里的2 |
| “要求CUDA，但没有CUDA？” | 核实服务器torch构建和GPU；只有CPU时可先做CPU模拟检查 |
| “HTTP200却无法解析掩膜？” | 检查response.json，可能是网页、纯文字或格式不符；核实完整端点和真实响应字段 |
| “profile未核实，不能批量？” | 先live单张探测，确认解析字段与极性，再设置profile_verified=true并进行本地QC |
| “检查点已存在/缺少依赖？” | 续跑同一阶段用--resume；缺少上游则先完成上游阶段，不要删正式模型来绕过检查 |
| “数据/配置版本变化，不能恢复？” | 同一run_dir绑定原数据和科学参数。新实验使用新run_dir，保留旧结果 |
| “能删.pt或.sqlite3让目录更少吗？” | 它们是模型和任务记录。正式结果保留；仅已确认是模拟检查的整次产物可清理 |

如果只是要完成日常使用，重点看：本页开头的配置表、项目文件树、第2节安装、第3节配置、第5–8节正式操作及第10节输出位置。内部模型文件不需要逐个手工运行。

## 12. 上传服务器与提交GitHub

公开仓库只提交 `config/geobridge_ts.example.toml` 和模拟配置。真实的 `config/geobridge_ts.toml` 已加入Git忽略规则，保留在本机，内容与密钥均不修改；它不会进入提交或Git历史。

从GitHub检出项目后，在新目录第一次使用前复制模板：

```bash
cp config/geobridge_ts.example.toml config/geobridge_ts.toml
```

Windows PowerShell：

```powershell
Copy-Item -LiteralPath 'config/geobridge_ts.example.toml' -Destination 'config/geobridge_ts.toml'
```

然后在新生成的主配置中填写服务器路径、密钥、设备和训练预算。已有主配置时不要用模板覆盖。程序实际运行仍使用 `geobridge_ts.toml`，不会自动读取example覆盖你的配置。

直接上传自己的服务器时，可以单独安全传输真实配置，与代码放在同一个config目录下。`.gitignore`不限制普通文件上传。`runs/`、Python字节码和测试缓存也不作为源代码提交。
