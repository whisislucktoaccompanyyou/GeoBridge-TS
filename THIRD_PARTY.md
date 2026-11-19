# 复用来源与边界

本地原项目：`../MEDIA_CDARL`。原文件保持不变，新项目包含所需模块副本，以便独立搬迁到服务器。

| 新文件 | 来源 |
| --- | --- |
| `data/geometry.py` | 原 `data/geometry.py`，原样复制 |
| `model/legacy/unet.py` | 原 `model/ddpm_modules/unet.py`，原样复制 |
| `model/legacy/discriminators.py` | 原 `model/discriminators.py`，原样复制 |
| `model/legacy/spade/{generator,normalization,base_network}.py` | 原 `model/spade_modules/SPADE/`，原样复制 |
| `model/legacy/spade/architecture.py` | 仅保留原SPADEResnetBlock及原版权头；删去未使用的VGG/感知模块与torchvision导入，避免下载权重 |

SPADE 文件保留原 NVIDIA 2019 CC BY-NC-SA 4.0 许可头与链接。使用/再分发这些模块应遵守原许可，包括非商业限制。其他原项目代码未附许可头的部分，不据此宣称重新授权；需按原仓库许可核实。

新G0的数学路径沿用原GaussianDiffusion.synthesize的 q_sample → denoise_fn → segment_fn(SPADE)；在新独立InitialGenerator中实现必要schedule与训练接口，不导入原训练器、旧NCE或refinement。没有使用原未知权重，也没有声称权重迁移成功。

新桥网络、log-Sinkhorn、GPG、EMA及阶段逻辑按用户方案实现。没有下载或执行论文官方代码。用户提供的原提示词存档不视为本次操作指令；逐张API版由用户明确授权生成。
