# Exploring LLM-Based Multi-Agent Situation Awareness for Zero-Trust Space-Air-Ground Integrated Network

This paper has been published in **IEEE Journal on Selected Areas in Communications (IEEE JSAC)**.

**Paper:** https://ieeexplore.ieee.org/document/10963886

LLM-SA provides two security-oriented agents, namely the **Strategy Agent** and the **Scoring Agent**, for security strategy generation and risk assessment, respectively. Both agents perform inference using `Meta-Llama-3-8B-Instruct` with their corresponding LoRA adapters.

## Strategy Agent

- **Input:** Security event descriptions and prompts containing candidate security strategies.
- **Functionality:** Generates structured strategy lists based on the provided descriptions.
- **Output:** A list of generated strategies for subsequent evaluation or comparison.
- **Dependencies:** Base model and strategy-generation LoRA adapter (e.g., `llama3_lora_strategy`).

## Scoring Agent

- **Input:** Security event descriptions.
- **Functionality:** Predicts seven security metrics, including Attack Vector, Attack Complexity, Privileges Required, User Interaction, and Confidentiality/Integrity/Availability Impact. Numerical risk scores are then calculated using the corresponding scoring formulas.
- **Output:** A numerical risk score (0–10) for each input, combining Impact and Exploitability scores.
- **Dependencies:** Base model and scoring LoRA adapter (e.g., `llama3_lora_scoring`).

## Data

- A subset of the datasets used for model development.

## Environment and Dependencies

- Python ≥ 3.8
- Project dependencies are listed in `requirements.txt`.

## Model Resources

- **Base Model:** A locally stored or accessible `Meta-Llama-3-8B-Instruct` model.
- **LoRA Adapters:** The corresponding adapter directories, `llama3_lora_strategy` and `llama3_lora_scoring`.

## Installation and Usage

### Environment Setup

- **Operating Systems:** Windows / Linux / macOS, with `git` and `python` installed (Python 3.10+ recommended).
- **Hardware:** A CUDA-compatible GPU is recommended. CPU inference is also supported but may be significantly slower.

**Dependency Installation:**

```bash
cd LLM-SA
python -m venv .venv
```

Activate the virtual environment and install the required dependencies:

```bash
pip install -r requirements.txt
```

### Base Model and LoRA Adapters

**Base Model:** `Meta-Llama-3-8B-Instruct`

**Option 1: Load directly from Hugging Face**

Set `--model_path` to `meta-llama/Meta-Llama-3-8B-Instruct` in the command line. Before loading the model, ensure that you have logged into your Hugging Face account and accepted the model's license agreement.

```bash
pip install huggingface_hub
huggingface-cli login
```

You must also accept the model's access agreement on Hugging Face.

**Option 2: Download the model locally**

Download the base model to a local directory and specify its path using `--model_path`. For example:

```bash
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct --local-dir ./Meta-Llama-3-8B-Instruct
```

**LoRA Adapters:**

The repository includes two LoRA adapter directories:

- `./llama3_lora_scoring`: LoRA adapter for security risk scoring.
- `./llama3_lora_strategy`: LoRA adapter for security strategy generation.

### Running the Scoring Agent

**GPU Inference:**

```bash
python "Scoring agent/scoring_agent.py" --model_path meta-llama/Meta-Llama-3-8B-Instruct --lora_path ./llama3_lora_scoring --device cuda:0 --max_new_tokens 256 --input_file "./data/test dataset.txt" --output_file ./output/scores.txt
```

**CPU Inference:**

```bash
python "Scoring agent/scoring_agent.py" --model_path meta-llama/Meta-Llama-3-8B-Instruct --lora_path ./llama3_lora_scoring --device cpu --max_new_tokens 128 --input_file "./data/test dataset.txt" --output_file ./output/scores.txt
```

**Output:**

The agent generates a risk score ranging from 0 to 10 for each input. The scores are saved to `./output/scores.txt`, with one score per line.

### Running the Strategy Agent

**GPU Inference:**

```bash
python "Strategy agent/strategy_agent.py" --model_path meta-llama/Meta-Llama-3-8B-Instruct --lora_path ./llama3_lora_strategy --device cuda:0 --max_new_tokens 256 --input_file "./data/test dataset.txt" --output_file ./output/strategies.txt
```

**CPU Inference:**

```bash
python "Strategy agent/strategy_agent.py" --model_path meta-llama/Meta-Llama-3-8B-Instruct --lora_path ./llama3_lora_strategy --device cpu --max_new_tokens 128 --input_file "./data/test dataset.txt" --output_file ./output/strategies.txt
```

**Output:**

The agent generates structured strategy lists for the input security descriptions. The results are saved to `./output/strategies.txt`, with one formatted strategy list per line.

### Running the Backend API

Execute the following command from the repository root directory:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

**Available API Endpoints:**

- `GET /health`
- `POST /score`
- `POST /strategy`

### Troubleshooting

- **Hugging Face Access Errors (e.g., HTTP 403):** Log into your Hugging Face account and accept the model's license agreement before attempting to load the model.
- **No GPU or Insufficient GPU Memory:** Set `--device` to `cpu` or reduce `--max_new_tokens`.
- **File Paths Containing Spaces:** Enclose paths in quotation marks, such as `"./data/test dataset.txt"`.
- **Unsupported bfloat16:** If dtype-related errors occur, modify the `torch_dtype` parameter in the model-loading code to use `float16` or `float32`, or switch to CPU inference.

## Citation

If you find our code useful for your research, please consider citing our paper:

```bibtex
@ARTICLE{10963886,
   author={Cao, Xinye and Nan, Guoshun and Guo, Hongcan and Mu, Hanqing and Wang, Long and Lin, Yihan and Zhou, Qinchuan and Li, Jiayi and Qin, Baohua and Cui, Qimei and Tao, Xiaofeng and Fang, He and Du, Haitao and Quek, Tony Q. S.},
   journal={IEEE Journal on Selected Areas in Communications},
   title={Exploring LLM-Based Multi-Agent Situation Awareness for Zero-Trust Space-Air-Ground Integrated Network},
   year={2025},
   volume={43},
   number={6},
   pages={2230-2247},
   keywords={Satellites;Security;Space-air-ground integrated networks;Satellite broadcasting;Feature extraction;Zero Trust;Protocols;Vehicle dynamics;Network security;Low earth orbit satellites;Space-air-ground integrated network;zero trust;LLM-based multi-agent;situation awareness},
   doi={10.1109/JSAC.2025.3560042}
}
```

# Exploring LLM-Based Multi-Agent Situation Awareness for Zero-Trust Space-Air-Ground Integrated Network
论文已发表在IEEE JSAC，链接：https://ieeexplore.ieee.org/document/10963886

LLM‑SA 提供两个面向安全场景的 Agent，分别用于“策略生成”和“风险评分”。它们基于 `Meta‑Llama‑3‑8B‑Instruct` 与对应的 LoRA 适配权重进行推理。

## Strategy Agent

- 输入：安全事件/描述文本以及候选策略集合提示
- 功能：根据描述生成严格的策略列表
- 输出：逐条生成的策略列表，便于后续评估或比对
- 依赖：基础模型与策略生成 LoRA（示例名称：`llama3_lora_strategy`）

## Scoring Agent

- 输入：安全事件/描述文本
- 功能：生成七项指标（Attack Vector、Attack Complexity、Privileges Required、User Interaction、Confidentiality/Integrity/Availability Impact），并依据评分公式计算数值分
- 输出：每条输入对应的数值分（0–10，包含 Impact 与 Exploit 的合成分）
- 依赖：基础模型与评分 LoRA（示例名称：`llama3_lora_scoring`）

## data
- 模型部分数据集

## 环境与依赖

- Python ≥ 3.8
- 项目依赖见 `requirements.txt`

## 模型资源
- 需要本地或可访问的基础模型目录：`Meta‑Llama‑3‑8B‑Instruct`
- 需要对应 LoRA 权重目录：`llama3_lora_strategy` 与 `llama3_lora_scoring`

## 部署与运行

### 环境准备
- 操作系统：Windows / Linux / macOS，已安装 `git` 与 `python`（建议 3.10+）
- 硬件：支持 CUDA 的 GPU（推荐）；如无 GPU 可用 CPU 运行但速度较慢
- 依赖安装：
  - `cd LLM-SA`
  - `python -m venv .venv`
  - 激活虚拟环境后执行
  - `pip install -r requirements.txt`

### 模型与适配器
- 基础模型：`Meta‑Llama‑3‑8B‑Instruct`
  - 方式一（直接用模型 ID 在线加载）：在命令行中将 `--model_path` 设为 `meta-llama/Meta-Llama-3-8B-Instruct`，需要先完成 Hugging Face 账户登录与许可：
    - `pip install huggingface_hub`
    - `huggingface-cli login`
    - 在 Hugging Face 上接受该模型的使用协议
  - 方式二（本地缓存/下载）：将模型下载到本地目录，并把 `--model_path` 指向该目录，例如：
    - `huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct --local-dir ./Meta-Llama-3-8B-Instruct`
- LoRA 适配器：仓库已包含 `./llama3_lora_scoring` 与 `./llama3_lora_strategy`，分别用于评分与策略生成。


### 运行评分 Agent
- 示例命令（GPU）：
  - `python "Scoring agent/scoring_agent.py" --model_path meta-llama/Meta-Llama-3-8B-Instruct --lora_path ./llama3_lora_scoring --device cuda:0 --max_new_tokens 256 --input_file "./data/test dataset.txt" --output_file ./output/scores.txt`
- 示例命令（CPU）：
  - `python "Scoring agent/scoring_agent.py" --model_path meta-llama/Meta-Llama-3-8B-Instruct --lora_path ./llama3_lora_scoring --device cpu --max_new_tokens 128 --input_file "./data/test dataset.txt" --output_file ./output/scores.txt`
- 运行结果：每行一个 0–10 的风险分，写入 `./output/scores.txt`。

### 运行策略 Agent
- 示例命令（GPU）：
  - `python "Strategy agent/strategy_agent.py" --model_path meta-llama/Meta-Llama-3-8B-Instruct --lora_path ./llama3_lora_strategy --device cuda:0 --max_new_tokens 256 --input_file "./data/test dataset.txt" --output_file ./output/strategies.txt`
- 示例命令（CPU）：
  - `python "Strategy agent/strategy_agent.py" --model_path meta-llama/Meta-Llama-3-8B-Instruct --lora_path ./llama3_lora_strategy --device cpu --max_new_tokens 128 --input_file "./data/test dataset.txt" --output_file ./output/strategies.txt`
- 运行结果：按行输出模型生成的严格列表格式字符串，写入 `./output/strategies.txt`。

### 运行后端 API
- 在仓库根目录执行：
  - `uvicorn backend.main:app --host 0.0.0.0 --port 8000`
- 可用接口：
  - `GET /health`
  - `POST /score`
  - `POST /strategy`

### 常见问题
- Hugging Face 权限错误（如 403）：请登录并在页面上接受模型许可，再运行命令。
- 无 GPU 或显存不足：将 `--device` 设为 `cpu` 或降低 `--max_new_tokens`。
- 路径包含空格：命令中请用引号包裹路径，例如 `"./data/test dataset.txt"`。
- bfloat16 不支持：如出现 dtype 相关错误，可在本地调整加载精度为 `float16/float32`（需要修改代码中模型加载的 `torch_dtype`），或改用 CPU 运行。

## 引用

如果参考了我们的代码，请引用以下文章：

```bibtex
@ARTICLE{10963886,
   author={Cao, Xinye and Nan, Guoshun and Guo, Hongcan and Mu, Hanqing and Wang, Long and Lin, Yihan and Zhou, Qinchuan and Li, Jiayi and Qin, Baohua and Cui, Qimei and Tao, Xiaofeng and Fang, He and Du, Haitao and Quek, Tony Q. S.},
   journal={IEEE Journal on Selected Areas in Communications},
   title={Exploring LLM-Based Multi-Agent Situation Awareness for Zero-Trust Space-Air-Ground Integrated Network},
   year={2025},
   volume={43},
   number={6},
   pages={2230-2247},
   keywords={Satellites;Security;Space-air-ground integrated networks;Satellite broadcasting;Feature extraction;Zero Trust;Protocols;Vehicle dynamics;Network security;Low earth orbit satellites;Space-air-ground integrated network;zero trust;LLM-based multi-agent;situation awareness},
   doi={10.1109/JSAC.2025.3560042}
}
```
