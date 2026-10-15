# VOICE-25M 使用说明

本版对应 scGPT 对齐的 25M 模型：k-means 细胞聚类 bank，每簇约 100 个细胞，K=50、tau=0.03，使用为该 bank 重拟合的 V3 beta。代码结构沿用原仓库的 `voice/`、`train/`、`predict/`、`configs/`、`weights/`，不包含下游分析。

安装 `requirements.txt` 后，准备独立模型包（见 [weights/README.md](../weights/README.md)）：

```bash
python predict/predict.py --image /data/slide.svs \
  --release /models/voice25m --out /results/pred.h5ad
```

只提供 H&E 即可开始：自动分割核、按 55 µm 视野裁剪、构造掩膜、直接预测、cluster bank 召回、V3 融合。普通 PNG/JPEG 通常缺少物理分辨率，需要另传实际 `--mpp`；程序不会猜测。原始输入不需要 RNA、已算好的特征或训练数据。预训练权重和 bank 是运行依赖，代码仓库没有它们的公开下载地址，发布者还需提供。

输出 `X` 是融合分数，`layers['A']` 是直接分支 `log1p(mu)`，`layers['R']` 是召回的 z-score；融合值不能当绝对分子计数。支持已有 `--cells` 或 `--prepared` 输入；大矩阵逐批写盘。

模型、bank 和 beta 必须配套。仓库内已保存真实 V3 beta 与校验信息，拒绝把旧 K200 beta 或不同指纹 bank 混用。所有运行路径通过命令行或相对模型配置提供，没有个人机器上的固定数据地址。

训练、scGPT 预处理、建库、V3 拟合和打包说明见 [training.md](training.md)。完整 WSI 质量仍取决于分辨率、染色和分割；自动核分割与训练时的细胞边界不完全相同，有配套细胞边界时可直接使用。
