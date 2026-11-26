# 数据挖掘导论 2026 大作业

本仓库是小组实践作业的工作目录。[作业说明](作业说明_2026.pdf)列出了题目、评分项和提交要求。

## 目录结构与数据准备

从[课程组北航云盘](https://bhpan.buaa.edu.cn/link/AA29E06E1DF21C419D9A1F56FFB409B665)下载课程数据和报告模板，按原目录解压到仓库根目录的 `DM_2026_Dataset/`。若下载的是完整的 `DM_2026_Dataset` 压缩包，解压到仓库根目录即可；若分开下载，将各项放到同一个 `DM_2026_Dataset/` 下。`weather.csv` 直接放入，图像、财务和甲状腺数据保留下面的两层同名目录。数据目录仅在本地使用，Git 不追踪。

```text
Data_Mining/
├── 作业说明_2026.pdf
├── README.md
├── DM_2026_Dataset/                           # 本地下载解压
│   ├── Cluster/Cluster/                      # dataset/、cluster_labels.json
│   ├── Image_Anomaly_Detection/Image_Anomaly_Detection/
│   │                                         # hazelnut/、zipper/、异常标签
│   ├── weather.csv
│   ├── thyroid/thyroid/                      # train-set.csv、test-set.csv
│   ├── FinanceQA/FinanceQA/                  # 数据、题目和评测材料
│   └── course-report-template/               # 课程模板，含 fonts/
├── src/
│   ├── T1-cluster/                           # 图像无监督聚类
│   ├── T2-Image_Anomaly_Detection/           # 图像异常检测
│   ├── T3-weather/                           # 时间序列预测
│   ├── T4-thyroid/                           # 无监督疾病判断
│   └── T5-FinanceQA/                         # 选做：企业财务问答
└── report/                                   # LaTeX 源码、图表和实验报告 PDF
```

## 复现任务 1 和任务 2

在仓库根目录使用 Python 3.12 执行下列命令。两个任务共享图像特征抽取、数据核验和制图代码。默认数据根目录为 DM_2026_Dataset/，也可使用 --root 指定同结构的外部数据目录。

```sh
python -m pip install -r src/requirements.txt
python src/audit.py
python src/vision.py
python src/T1-cluster/cluster.py
python src/T2-Image_Anomaly_Detection/anomaly.py
python src/figures.py
```

首次抽取特征时，程序会从 PyTorch 官方地址下载 ResNet-50 ImageNet-1K V2 权重并核验 SHA-256。离线可给 vision.py 传入 --weights。只抽取一项任务的特征时，可用 --task cluster 或 --task anomaly。特征缓存写入 _work/，逐图结果和评价指标写入 results/。

任务 1 拟合时仅使用图像，类别标签只在评价时读取。任务 2 只使用课程训练集做分层五折验证与阈值选择，测试标签仅在评价时读取。

## 选做二：企业财务问答

[选做二运行说明](src/T5-FinanceQA/README.md)包含数据准备、Flash 作答、Pro 独立评测和实验输出位置。正式测试 19 题均已作答，17 题获得课程检查器支持（89.5%）。查看[报告 PDF](report/financeqa-report.pdf)和 [LaTeX 源码](report/financeqa-report.tex)，正文可合入小组总报告。准备好 report/fonts 后，在 report/ 下执行：

~~~sh
latexmk -pdf -outdir=build-course-report financeqa-report.tex
~~~

## 报告与小组协作

报告正文入口为 [report/course-report.tex](report/course-report.tex)。小组成员信息在 [report/authors.tex](report/authors.tex) 统一填写。编译时，将本地课程模板的 `DM_2026_Dataset/course-report-template/fonts/` 复制为 `report/fonts/`，进入 report/ 后使用 pdfLaTeX 和 BibTeX。安装了 TeX Live 和 latexmk 时，可以运行以下命令。

```sh
cd report
latexmk -pdf -outdir=build-course-report course-report.tex
```

组员在 main 上协作。先同步远程最新提交，将各自的实验源码放入 src/ 中对应的任务目录，再更新本 README 和 report/ 中相应的章节、图表及参考文献。

提交作业源码、实验报告和复现所需的材料。个人实施记录、本地启动脚本、运行环境和生成缓存留在本地。
