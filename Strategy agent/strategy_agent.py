import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

class StrategyPromptAgent:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    def build_messages(self, description, question):
        return [
            {"role": "system", "content": "Please select the most appropriate set of strategies from the list below based on the description, and output them in strict list format directly(['xxx', 'xxx', 'xxxx',...]),please output only a list, not any extraneous words:"},
            {"role": "system", "content": question},
            {"role": "system", "content": "please just output the list alone.The description is following:"},
            {"role": "user", "content": description}
        ]
    def build_prompt(self, description, question):
        messages = self.build_messages(description, question)
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

class StrategyTestAgent:
    def __init__(self, model_path, lora_path, device, max_new_tokens):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.device = torch.device(device)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(self.device)
        self.model = PeftModel.from_pretrained(self.model, model_id=lora_path).to(self.device)
        self.prompt_agent = StrategyPromptAgent(self.tokenizer)
        self.eos_token_id = self.tokenizer.encode("<|eot_id|>")[0]
        self.max_new_tokens = max_new_tokens
    def generate_one(self, description, question):
        text = self.prompt_agent.build_prompt(description, question)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        with torch.no_grad():
            generated_ids = self.model.generate(
                model_inputs.input_ids,
                max_new_tokens=self.max_new_tokens,
                eos_token_id=self.eos_token_id
            )
        trimmed = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        return self.tokenizer.batch_decode(trimmed, skip_special_tokens=True)[0]
    def run_on_files(self, input_file_path, question_file_path, output_file_path):
        with open(input_file_path, "r", encoding="utf-8") as f:
            descriptions = [ln.strip() for ln in f.readlines()]
        with open(question_file_path, "r", encoding="utf-8") as f:
            questions = [ln.strip() for ln in f.readlines()]
        flag = 1
        with open(output_file_path, "w", encoding="utf-8") as out:
            for description, question in zip(descriptions, questions):
                if not description:
                    continue
                print("questionflag" + str(flag))
                response = self.generate_one(description, question)
                out.write("questionflag" + str(flag) + "\n")
                out.write(response + "\n")
                print(response)
                flag += 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/home/wl/security scoring model/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--lora_path", default="./llama3_lora_strategy")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--input_file", default="/home/wl/security scoring model/data/1107_description.txt")
    parser.add_argument("--question_file", default="/home/wl/security scoring model/data/output_1109_multi_choice.txt")
    parser.add_argument("--output_file", default="/home/wl/security scoring model/output/1220_responses_strategy.txt")
    args = parser.parse_args()
    agent = StrategyTestAgent(
        model_path=args.model_path,
        lora_path=args.lora_path,
        device=args.device,
        max_new_tokens=args.max_new_tokens
    )
    agent.run_on_files(
        input_file_path=args.input_file,
        question_file_path=args.question_file,
        output_file_path=args.output_file
    )

if __name__ == "__main__":
    main()
