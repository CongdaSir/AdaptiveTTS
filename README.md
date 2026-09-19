# AdaptiveTTS

AdaptiveTTS 是一个预算受限的 Agentic Test-Time Scaling 原型。每次完整 Agent rollout 都作为不可变的搜索树节点；编排器根据候选轨迹、验证反馈和剩余预算，在 `ANSWER / SAMPLE / REFINE / VERIFY` 中动态选择下一步。

当前版本面向可自动验证的推理任务，通过 vLLM 的 OpenAI 兼容接口调用模型。评测端不会再次在本地加载模型。

## 项目结构

```text
src/
├── core/                 # 搜索树、动作、控制器、规则和预算管理
├── executors/            # vLLM HTTP rollout 执行器
└── evaluation/           # 评测入口、逐题持久化、树解析与可视化
data/                     # AIME24、AIME25、HMMT Feb 2025
scripts/                  # vLLM 服务和通用评测启动脚本
tests/                    # 单元测试
```

`logs/`、`results/`、本地模型和 `.venv/` 不提交到 Git；它们在运行时生成或由用户自行准备。

## 1. 克隆与安装

要求：Linux、Python 3.10+、NVIDIA GPU，以及与 CUDA 12.4 兼容的驱动。依赖固定为 PyTorch 2.6.0（cu124）和 vLLM 0.8.5.post1，不需要修改系统 CUDA。

```bash
git clone git@github.com:CongdaSir/AdaptiveTTS.git
cd AdaptiveTTS

python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

运行测试：

```bash
.venv/bin/python -m unittest discover -s tests -v
```

## 2. 准备模型

先将 Qwen3-4B 下载到本地。启动脚本默认读取：

```text
/home/u2026104455/Models/Qwen3-4B
```

其他机器可以通过 `MODEL_DIR` 指定路径。模型权重不包含在本仓库中。

## 3. 在 GPU 5 启动 vLLM

```bash
GPU_ID=5 \
MODEL_DIR=/home/u2026104455/Models/Qwen3-4B \
PORT=8000 \
./scripts/start_vllm_qwen3_4b.sh
```

脚本使用 `setsid + nohup` 在后台启动服务，因此关闭 VSCode 或终端后进程仍会继续。默认上下文长度为 32768，服务名为 `Qwen3-4B`，日志和 PID 分别写入：

```text
logs/qwen3-4b-vllm.log
logs/qwen3-4b-vllm.pid
```

确认服务就绪：

```bash
curl http://127.0.0.1:8000/v1/models
```

查看服务日志：

```bash
tail -f logs/qwen3-4b-vllm.log
```

## 4. 启动评测

三个内置数据集均使用同一入口，每次选择一个数据集：

```bash
./scripts/run_evaluation.sh aime24
./scripts/run_evaluation.sh aime25
./scripts/run_evaluation.sh hmmt_feb_2025
```

脚本同样使用 `setsid + nohup`。默认参数为：

- 总搜索预算：`80`
- 单次 rollout 最大输出：`16384` tokens
- temperature：`0.6`
- vLLM 端口：`8000`
- 模型服务名：`Qwen3-4B`

可以用环境变量覆盖参数：

```bash
BUDGET=100 \
MAX_NEW_TOKENS=16384 \
TEMPERATURE=0.6 \
PORT=8000 \
./scripts/run_evaluation.sh aime25
```

也可直接调用评测入口：

```bash
.venv/bin/python -m src.evaluation.evaluate \
  --server-url http://127.0.0.1:8000/v1 \
  --model Qwen3-4B \
  --data data/aime25.jsonl \
  --budget 80 \
  --max-new-tokens 16384 \
  --temperature 0.6
```

## 5. 输出与进度

以 AIME25 为例：

```text
results/aime25-qwen3-4b.json          # 全部完成后写出的汇总
results/aime25-Qwen3-4B-tree.jsonl    # 每完成一题立即 flush 的完整搜索树
logs/aime25-qwen3-4b-eval.log         # stderr 和错误日志
logs/aime25-qwen3-4b-eval.pid         # 后台评测 PID
```

搜索树 JSONL 的每一行对应一道题，包含题目、预测、正确性、停止原因、动作序列、节点父子关系、验证结果、资源消耗以及完整 rollout 中间轨迹。评测中断时，已经完成的行仍然可用。

查看进度：

```bash
wc -l results/aime25-Qwen3-4B-tree.jsonl
tail -f logs/aime25-qwen3-4b-eval.log
ps -p "$(cat logs/aime25-qwen3-4b-eval.pid)" -o pid,etime,cmd
```

注意：当前直接启动一次完整评测时，会重写同名搜索树文件；不要在同一个结果名上并行运行两次评测。

## 6. 搜索树解析与可视化

输出全局统计和文本树：

```bash
.venv/bin/python -m src.evaluation.tree_viewer \
  --input results/aime25-Qwen3-4B-tree.jsonl \
  --format text
```

查看指定题目并展开完整轨迹：

```bash
.venv/bin/python -m src.evaluation.tree_viewer \
  --input results/aime25-Qwen3-4B-tree.jsonl \
  --task-id 0 \
  --format text \
  --show-traces
```

生成可交互 HTML：

```bash
.venv/bin/python -m src.evaluation.tree_viewer \
  --input results/aime25-Qwen3-4B-tree.jsonl \
  --format html \
  --output results/aime25-Qwen3-4B-tree.html
```

导出解析后的 JSON：

```bash
.venv/bin/python -m src.evaluation.tree_viewer \
  --input results/aime25-Qwen3-4B-tree.jsonl \
  --format json \
  --output results/aime25-Qwen3-4B-tree-summary.json
```

## 数据与评测说明

`data/*.jsonl` 已转换为统一的 `id / prompt / answer` 格式。当前 verifier 支持规范化字符串、整数、分数和百分数比较；对复杂符号表达式尚未做完整的代数等价判断，因此扩展到其他数学数据集时应同步增强 verifier。

数据集与模型的许可分别遵循其原始发布方条款；本仓库只保存运行所需的轻量评测数据和代码。
