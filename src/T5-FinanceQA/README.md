
# 选做二：企业财务问答 Agent

接收课程包给定的 CIK 和问题。模型根据现有证据选择查财务表、找年报、展开上下文或计算，最后输出英文答案。无按题号编写的答案或 SQL。基础方案是 BM25 与标签匹配；增强方案增加财务词扩展、重排和年度来源优先规则。

## 复现

从 [课程组云盘](https://bhpan.buaa.edu.cn/link/AA29E06E1DF21C419D9A1F56FFB409B665)下载课程包。仓库根目录下应有 `DM_2026_Dataset/FinanceQA/FinanceQA/agent_workspace/data/finance.db`，另有同级的 `evaluation_only/`。无需修改课程包。运行环境为 Python 3.11+，SQLite 启用 FTS5；仅使用标准库。下面使用 PowerShell，命令在仓库根目录执行。

在本机设置 `FINANCEQA_API_KEY`（作答）和 `LLM_API_KEY`（课程评测），使用同一服务时可从前者拷贝到后者。密钥不出现在代码、配置快照或模型日志中。

```powershell
$agent = 'src/T5-FinanceQA/run.py'
$eval = 'DM_2026_Dataset/FinanceQA/FinanceQA/evaluation_only/evaluate.py'
$judge = 'https://api.deepseek.com/v1/chat/completions'
$out = 'results/financeqa/reproduce'
$env:LLM_API_KEY = $env:FINANCEQA_API_KEY

python $agent prepare
python -m unittest discover -s src/T5-FinanceQA/tests -v

foreach ($variant in @('basic', 'hybrid')) {
    $run = "$out/train-$variant"
    python $agent run --split train --model deepseek-flash --variant $variant --workers 8 --output $run
    python $eval --split train --predictions "$run/predictions.json" --judge-url $judge --model deepseek-v4-pro --output "$run/evaluation.json"
}
```

先看开发集结果，选择方法，再冻结并运行测试。`--variant` 填开发集选定的配置；报告记录本次实际入选方案。输出路径每次须新建。

```powershell
$variant = Read-Host 'Selected variant (basic or hybrid)'
$cfg = @('--model', 'deepseek-flash', '--variant', $variant, '--workers', '8')
python $agent freeze @cfg --output "$out/frozen.json"
python $agent run @cfg --split test --frozen "$out/frozen.json" --output "$out/test"
python $eval --split test --predictions "$out/test/predictions.json" --check-only --output "$out/test/check.json"
python $eval --split test --predictions "$out/test/predictions.json" --judge-url $judge --model deepseek-v4-pro --output "$out/test/evaluation.json"
```

作答脚本只从指定划分读取问题，并只向模型提供当前题和工具结果。它不会打开参考答案或评分文件。官方检查器是单独进程；整批测试作答完成后才运行。即使不同处理方式可能提高分数，也不要据测试答案或反馈修改正式方法。

## 输出和证据

| 文件 | 内容 |
| --- | --- |
| `_work/financeqa/index.json` | 报告清单、哈希、分块数量和索引构建耗时 |
| `predictions.json` | 全量课程协议输出：`question_id`、`answer` |
| `trace.jsonl` | 每题一条，记录真实工具名、参数、结果、错误、证据编号与每轮 API 用量 |
| `config.json`、`summary.json` | 模型、配置、数据和源码哈希、题目数、token 数、耗时 |
| `frozen.json` | 测试前冻结的方法配置，运行时须精确匹配 |
| `check.json`、`evaluation.json` | 官方格式检查和独立实评分，含分公司统计 |

索引、记录与答案保存在 Git 忽略的本地目录。提交实验材料时将报告、预测、日志、配置与评测结果一起发给课程组。不要把课程原始数据、密钥或测试评测请求中的参考答案加入 Git。

如果评分发生网络超时，可以用 [retry_failed.py](evaluation/retry_failed.py) 在同一份预测上补判失败请求，沿用课程原检查器、提示和模型，另存新文件。原始评分和已成功的逐题评分照常保留。参数见该脚本的 --help。
