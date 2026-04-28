## Start Prompt

```
你现在在 autoresearch-trio 项目中。请严格阅读 README.md、program.md、prepare.py、train.py。

目标：参考 karpathy/autoresearch 的循环，连续做至多 100 轮 TRIO text-to-SQL 实验，目标是最大化 eval/ex。若连续 5 轮没有提升 eval/ex，则提前停止。

本次实验 tag 为：apr-gpt5-5。

约束：
- 主要只修改 train.py。
- 不要修改 prepare.py、program.md、README.md、data/、data/processed eval 数据或 SQLite 数据库。
- 默认 eval 是 train-holdout，按 db_id 整库隔离；当前 train.py 默认 EVAL_LIMIT = 120。
- 不要使用 bird_mini_dev 训练，也不要硬编码 eval question、db_id 或 gold SQL。
- 每轮选择一个清晰实验想法，先修改 train.py，再 commit 这个实验改动。
- 每轮运行：
   uv run train.py --run-tag apr-gpt5-5 --description "<short description>" > run.log 2>&1
- 训练预算由 prepare.py 里的 TIME_BUDGET=300 控制。
- 每轮结束必须检查 run.log 和 results.csv。
- 如果 status 是 baseline 或 keep，保留 commit。
- 如果 status 是 discard，只能回退你本轮创建的实验 commit；不要丢弃用户未提交改动。
- 如果 status 是 crash，先读 tail -n 80 run.log；只修明显 bug。不可行则放弃该实验想法并继续下一轮。
- 不要提交 data/、outputs/、swanlog/、run.log、results.csv、.venv/、__pycache__/。
- 每轮汇报：实验想法、改动点、commit、ex、soft_f1、r_ves、valid_sql_rate、unsafe_sql_rate、status。
- 先跑一轮 baseline，然后继续做改进，直到达到停止条件或最多 100 轮。
```

## async mode prompt

```
你现在在 autoresearch-trio 项目中。请严格阅读 README.md、program.md、prepare.py、train.py、train_async.py。

目标：参考 karpathy/autoresearch 的循环，使用 TRIO async runner 连续做至多 100 轮 text-to-SQL 实验，目标是最大化 eval/ex。若连续 5 轮没有提升 eval/ex，则提前停止。

本次实验 tag 为：apr-gpt5-5-async。

约束：
- 主要只修改 train.py；train_async.py 只作为异步运行入口使用，除非 async runner 本身有明确 bug。
- 不要修改 prepare.py、program.md、README.md、data/、data/processed eval 数据或 SQLite 数据库。
- 默认 eval 是 train-holdout，按 db_id 整库隔离；当前 train.py 默认 EVAL_LIMIT = 120。
- 不要使用 bird_mini_dev 训练，也不要硬编码 eval question、db_id 或 gold SQL。
- 每轮选择一个清晰实验想法，先修改 train.py，再 commit 这个实验改动。
- 每轮运行：
   uv run train_async.py --run-tag apr-gpt5-5-async --description "<short description>" --train-pipeline-depth 4 --eval-concurrency 16 > run.log 2>&1
- 训练预算由 prepare.py 里的 TIME_BUDGET=300 控制。async runner 可以提交更多 step，但不要为了刷 step 牺牲 eval/ex。
- 每轮结束必须检查 run.log、results.csv 和 SwanLab。
- 如果 status 是 baseline 或 keep，保留 commit。
- 如果 status 是 discard，只能回退你本轮创建的实验 commit；不要丢弃用户未提交改动。
- 如果 status 是 crash，先读 tail -n 80 run.log；只修明显 bug。不可行则放弃该实验想法并继续下一轮。
- 不要提交 data/、outputs/、swanlog/、run.log、results.csv、.venv/、__pycache__/。
- 每轮汇报：实验想法、改动点、commit、ex、soft_f1、r_ves、valid_sql_rate、unsafe_sql_rate、steps、examples_seen、status。
- 先跑一轮 async baseline，然后继续做改进，直到达到停止条件或最多 100 轮。
```
