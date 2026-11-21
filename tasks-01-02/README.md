# 任务 1 和任务 2 · 工业图像数据挖掘

本目录对应小组分工中的**任务 1（图像无监督聚类）**与**任务 2（图像异常检测）**。

| 文件 | 用途 |
| --- | --- |
| vision.py | 用冻结的 ResNet-50 提取两项任务所需的整图与图像块特征 |
| cluster.py | 运行近邻图谱聚类和对照方法，并用课程标签评估 |
| anomaly.py | 运行正常图像块近邻、整图 SVM 和融合模型，并在独立测试集计分 |
| audit.py | 核对图像数量、分辨率和训练／测试集间的精确重复 |
| figures.py | 根据实验结果重建报告图表 |
| requirements.txt | Python 依赖 |

## 实验报告

[任务 1 与任务 2 的实验报告 PDF](report/course-report.pdf)涵盖问题定义、特征表示、模型设计、对照实验和结果分析。
LaTeX 源文件位于 report/，任务章节分别是 [task1.tex](report/sections/task1.tex) 和 [task2.tex](report/sections/task2.tex)。


## 环境和数据

使用 Python 3.12。运行需要课程发放的原始数据，数据根目录结构如下。

```text
数据根目录/
├── Cluster/Cluster/
│   ├── dataset/*.png
│   └── cluster_labels.json
└── Image_Anomaly_Detection/Image_Anomaly_Detection/
    ├── hazelnut/{train/{good,bad},test}/
    ├── zipper/{train/{good,bad},test}/
    └── image_anomaly_labels.json
```

从本目录执行下列命令。将 "数据根目录" 改成包含 Cluster 和 Image_Anomaly_Detection 的文件夹路径。

```sh
python -m pip install -r requirements.txt
python audit.py --root "数据根目录"
python vision.py --root "数据根目录"
python cluster.py --root "数据根目录" --stage fit
python cluster.py --root "数据根目录" --stage evaluate
python anomaly.py --root "数据根目录" --stage fit
python anomaly.py --root "数据根目录" --stage evaluate
```

首次提取特征会从 PyTorch 官方地址下载 ResNet-50 ImageNet-1K V2 权重并核验 SHA-256。离线环境可向 vision.py 传入 --weights 指定已有文件。只提取单项任务可传入 --task cluster 或 --task anomaly。程序在本地生成 _work/ 特征缓存和 results/ 评估结果。

聚类拟合阶段只读取图像，不读取类别标签。异常检测使用课程训练集做分层五折验证和阈值选择，测试标签仅由评估阶段读取。

## 重画图表与编译 LaTeX

使用学校发放的 LaTeX 模板，将其完整的字体资源目录复制到 report/fonts/。运行下面的程序会使用同一份 Times New Roman 字体重画图表。报告源文件已随附对应图表，只改正文或作者信息时可直接使用这些图表编译。

```sh
python figures.py --root "数据根目录"
```

在具备课程模板所需宏包的 TeX Live 环境中执行下列命令。合并小组报告时，可先在 report/authors.tex 中填写组员信息。

```sh
cd report
latexmk -pdf -outdir=build-course-report course-report.tex
```
