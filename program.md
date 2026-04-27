# autoresearch-trio

这是一个用于改进 TRIO 训练版 text-to-SQL 模型的自主研究协议。它刻意参考
`karpathy/autoresearch` 的工作流：一个稳定的基准文件、一个可编辑的实验文件、
以及一份本地结果日志。

## 初始化

开始一次新的实验前：

1. 先和用户确认 run tag，例如 `apr27-sql`。
2. 从 `main` 创建专用分支：

   ```bash
   git checkout -b autoresearch/<tag>
   ```

3. 阅读本轮实验范围内的文件：

   - `README.md`：项目背景。
   - `prepare.py`：固定的数据准备和评估框架；实验过程中不要修改。
   - `train.py`：TRIO 训练脚本；这是主要可编辑文件。
   - `program.md`：当前自主研究协议。

4. 检查本地 BIRD 数据：

   ```bash
   uv run prepare.py --check
   ```

5. 如有需要，生成 processed 数据：

   ```bash
   uv run prepare.py
   ```

6. 建立 baseline：

   ```bash
   uv run train.py > run.log 2>&1
   ```

baseline 应该创建 SwanLab 日志，并向 `results.csv` 追加一行记录。
默认 `prepare.py` 会从 train 数据里按 `db_id` 整库隔离出 50 条 eval 样本；
这些 holdout 数据库不会进入训练集。这样可以不依赖 Mini-Dev 的 Google Drive
数据库包。需要官方 Mini-Dev 时，显式使用 `uv run prepare.py --eval-source mini-dev`。

## 你可以做什么

- 修改 `train.py`。
- 调整 prompt 格式、evidence 使用方式、数据采样策略、LoRA rank、学习率、batch size、采样参数和 SFT 调度。
- 如果能保持实验逻辑清晰，可以在 `train.py` 中加入少量辅助函数。

## 你不能做什么

- 自主实验期间不要修改 `prepare.py`。
- 不要修改 eval 数据或 SQLite 数据库。
- 不要用 `bird_mini_dev` 训练；它是 eval/dev set。
- 不要硬编码 eval question、DB id 或 gold SQL。
- 不要手动改 `results.csv`，除非是在修复明显写坏的本地日志行。
- 不要新增依赖，除非用户明确批准。

## 目标

在 `prepare.py` 固定的 `TIME_BUDGET` 内，最大化 BIRD Mini-Dev SQLite
text-to-SQL 表现。

指标优先级：

1. 更高的 `ex`
2. 更高的 `soft_f1`
3. 更高的 `r_ves`
4. 更低的 `unsafe_sql_rate`
5. 更低的 `latency_ms`

## 实验循环

不断循环，直到用户中断：

1. 检查当前 git 状态：

   ```bash
   git status --short
   git rev-parse --short HEAD
   ```

2. 选择一个清晰的实验想法，并修改 `train.py`。
3. 提交变更：

   ```bash
   git add train.py
   git commit -m "experiment: <short description>"
   ```

4. 重定向日志运行实验：

   ```bash
   uv run train.py --description "<short description>" > run.log 2>&1
   ```

5. 提取 summary：

   ```bash
   grep "^ex:\|^soft_f1:\|^r_ves:\|^valid_sql_rate:\|^unsafe_sql_rate:" run.log
   ```

6. 确认 `results.csv` 已追加。
7. 如果该行状态是 `keep` 或 `baseline`，保留这个 commit。
8. 如果该行状态是 `discard`，只允许回退本轮 agent 自己创建的实验 commit。
   回退前必须先检查：

   ```bash
   git status --short
   ```

   不要丢弃用户未提交的改动，也不要回退不属于本轮实验的文件。
9. 如果该行状态是 `crash`，查看最后的日志：

   ```bash
   tail -n 80 run.log
   ```

   如果只是明显 typo 或小 bug，可以修复后重跑；如果实验想法本身不可行，就放弃并继续下一轮。

## 记录规则

每次正式实验都必须同时记录到：

- SwanLab 项目 `autoresearch-trio`
- 本地 `results.csv`

`--swanlab-mode disabled` 只用于 dry-run 或本地调试：

```bash
uv run train.py --dry-run --swanlab-mode disabled
```

`results.csv`、`run.log`、`outputs/`、`data/` 和 `swanlog/` 都是本地实验产物，
不要提交到 git。

## 简洁性准则

其他条件相同的时候，越简单越好。为了极小指标提升加入脆弱 prompt hack 通常不值得保留。
如果某个改动能简化代码并保持性能不变，也可以视为有效改进。
