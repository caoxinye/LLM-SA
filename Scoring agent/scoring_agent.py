import argparse
import ast
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

av = [0.85, 0.62, 0.55, 0.2]
ac = [0.77, 0.44]
pr = [0.85, 0.62, 0.27]
ui = [0.85, 0.62]
conf = [0, 0.22, 0.56]
integ = [0, 0.22, 0.56]
avail = [0, 0.22, 0.56]

map_av = {"Network": 0, "Adjacent Network": 1, "Local": 2, "Physical": 3}
map_ac = {"Low": 0, "High": 1}
map_pr = {"None": 0, "Low": 1, "High": 2}
map_ui = {"None": 0, "Required": 1}
map_impact = {"None": 0, "Low": 1, "High": 2}

def score_from_indices(A):
    CONF = conf[A[4]]
    INTEG = integ[A[5]]
    AVAIL = avail[A[6]]
    AVV = av[A[0]]
    ACC = ac[A[1]]
    PRR = pr[A[2]]
    UII = ui[A[3]]
    Impact = 6.42 * (1 - ((1 - CONF) * (1 - INTEG) * (1 - AVAIL)))
    Exploit = 8.22 * AVV * ACC * PRR * UII
    return min((Impact + Exploit), 10)

class ScoringPromptAgent:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    def build_messages(self, description):
        return [
            {"role": "system", "content": "Please select the following seven options [Attack Vector(\"Network\"/\"Adjacent Network\"/\"Local\"/\"Physical\"), Attack Complexity(\"Low\"/\"High\"), Privileges Required(\"None\"/\"Low\"/\"High\"), User Interaction(\"None\"/\"Required\"), Confidentiality Impact(\"None\"/\"Low\"/\"High\"), Integrity Impact(\"None\"/\"Low\"/\"High\"), Availability Impact(\"None\"/\"Low\"/\"High\")] based on the \"Description\" below, where the \"Description\" is as follows:  "},
            {"role": "user", "content": description}
        ]
    def build_prompt(self, description):
        messages = self.build_messages(description)
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

class ScoringAgent:
    def __init__(self, model_path, lora_path, device, max_new_tokens):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.device = torch.device(device)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(self.device)
        self.model = PeftModel.from_pretrained(self.model, model_id=lora_path).to(self.device)
        self.prompt_agent = ScoringPromptAgent(self.tokenizer)
        self.eos_token_id = self.tokenizer.eos_token_id
        self.max_new_tokens = max_new_tokens
    def generate_list(self, description):
        text = self.prompt_agent.build_prompt(description)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        with torch.no_grad():
            generated_ids = self.model.generate(
                model_inputs.input_ids,
                max_new_tokens=self.max_new_tokens,
                eos_token_id=self.eos_token_id
            )
        trimmed = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        resp = self.tokenizer.batch_decode(trimmed, skip_special_tokens=True)[0]
        start = resp.find("[")
        end = resp.rfind("]")
        if start != -1 and end != -1 and end > start:
            resp = resp[start:end+1]
        try:
            arr = ast.literal_eval(resp)
        except Exception:
            arr = []
        return arr
    def compute_score(self, arr):
        if isinstance(arr, list) and len(arr) == 7:
            idxs = [
                map_av.get(arr[0], 0),
                map_ac.get(arr[1], 0),
                map_pr.get(arr[2], 0),
                map_ui.get(arr[3], 0),
                map_impact.get(arr[4], 0),
                map_impact.get(arr[5], 0),
                map_impact.get(arr[6], 0),
            ]
            return round(score_from_indices(idxs), 2)
        return None
    def run_on_file(self, input_file_path, output_file_path):
        with open(input_file_path, "r", encoding="utf-8") as f:
            descriptions = [ln.strip() for ln in f.readlines()]
        results = []
        for description in descriptions:
            if not description:
                results.append("invalid")
                continue
            arr = self.generate_list(description)
            s = self.compute_score(arr)
            if s is None:
                results.append("invalid")
            else:
                results.append(str(s))
        with open(output_file_path, "w", encoding="utf-8") as out:
            out.write("\n".join(results) + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/home/wl/security scoring model/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--lora_path", default="/home/wl/security scoring model/llama3_lora_scoring")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--input_file", default="/home/wl/security scoring model/data/1107_description.txt")
    parser.add_argument("--output_file", default="/home/wl/security scoring model/output/123_pred_scores.txt")
    args = parser.parse_args()
    agent = ScoringAgent(
        model_path=args.model_path,
        lora_path=args.lora_path,
        device=args.device,
        max_new_tokens=args.max_new_tokens
    )
    agent.run_on_file(args.input_file, args.output_file)

if __name__ == "__main__":
    main()
