# Exploring LLM-Based Multi-Agent Situation Awareness for Zero-Trust Space-Air-Ground Integrated Network
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

## 环境与依赖

- Python ≥ 3.8
- 项目依赖见 `requirements.txt`

## 模型资源

- 需要本地或可访问的基础模型目录：`Meta‑Llama‑3‑8B‑Instruct`
- 需要对应 LoRA 权重目录：`llama3_lora_strategy` 与 `llama3_lora_scoring