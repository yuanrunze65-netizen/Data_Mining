# 任务 1 和任务 2 · 工业图像数据挖掘

本目录对应小组分工中的**任务 1（图像无监督聚类）**与**任务 2（图像异常检测）**。

| 文件 | 用途 |
| --- | --- |
| vision.py | 用冻结的 ResNet-50 提取两项任务所需的整图与图像块特征 |
| cluster.py | 运行近邻图谱聚类和对照方法，并用课程标签评估 |
| anomaly.py | 运行正常图像块近邻、整图 SVM 和融合模型，并在独立测试集计分 |
| requirements.txt | Python 依赖 |

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
python vision.py --root "数据根目录"
python cluster.py --root "数据根目录" --stage fit
python cluster.py --root "数据根目录" --stage evaluate
python anomaly.py --root "数据根目录" --stage fit
python anomaly.py --root "数据根目录" --stage evaluate
```

首次提取特征会从 PyTorch 官方地址下载 ResNet-50 ImageNet-1K V2 权重并核验 SHA-256。离线环境可向 vision.py 传入 --weights 指定已有文件。只提取单项任务可传入 --task cluster 或 --task anomaly。程序在本地生成 _work/ 特征缓存和 results/ 评估结果。

聚类拟合阶段只读取图像，不读取类别标签。异常检测使用课程训练集做分层五折验证和阈值选择，测试标签仅由评估阶段读取。
