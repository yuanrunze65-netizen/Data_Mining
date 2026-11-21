# 数据挖掘导论 2026 大作业

本仓库是小组实践作业的工作目录。[作业说明](作业说明_2026.pdf)列出了题目、评分项和提交要求。

| 目录 | 内容 |
| --- | --- |
| [DM_2026_Dataset/](DM_2026_Dataset/) | 课程发放的六项材料。保留 Cluster、FinanceQA、Image_Anomaly_Detection、course-report-template、thyroid 的解压目录及 weather.csv |
| [src/T1-cluster/](src/T1-cluster/) | 任务 1，图像无监督聚类 |
| [src/T2-Image_Anomaly_Detection/](src/T2-Image_Anomaly_Detection/) | 任务 2，图像异常检测 |
| [src/T3-weather/](src/T3-weather/) | 任务 3，时间序列预测 |
| [src/T4-thyroid/](src/T4-thyroid/) | 任务 4，无监督疾病判断 |
| [src/T5-FinanceQA/](src/T5-FinanceQA/) | 选做任务，企业财务问答 |
| [report/](report/) | LaTeX 报告工程、图表与编译后的 [任务 1 和任务 2 实验报告 PDF](report/course-report.pdf) |

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

## 报告与小组协作

报告正文入口为 [report/course-report.tex](report/course-report.tex)。小组成员信息在 [report/authors.tex](report/authors.tex) 统一填写。编译时，将 [课程模板的字体目录](DM_2026_Dataset/course-report-template/fonts/) 复制为 report/fonts/，进入 report/ 后使用 pdfLaTeX 和 BibTeX。安装了 TeX Live 和 latexmk 时，可以运行以下命令。

```sh
cd report
latexmk -pdf -outdir=build-course-report course-report.tex
```

组员在 main 上协作。先同步远程最新提交，将各自的实验源码放入 src/ 中对应的任务目录，再更新本 README 和 report/ 中相应的章节、图表及参考文献。

提交作业源码、实验报告和复现所需的材料。个人实施记录、本地启动脚本、运行环境和生成缓存留在本地。
