import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def load_model(base_model_path, lora_path):
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, lora_path)
    model.eval()
    return model, tokenizer

def generate_expression(model, tokenizer, state_dict):
    state_text = (f"坐标:({state_dict['major']},{state_dict['middle']},{state_dict['fine']}) "
                  f"愉悦度:{state_dict['valence']:.2f} 唤醒度:{state_dict['arousal']:.2f} "
                  f"趋近:{state_dict['approach']:.2f} L:{state_dict['L']} "
                  f"僵化度:{state_dict['stiffness']:.2f} 投射:{state_dict['proj']:.2f} 反哺:{state_dict['nour']:.2f}")
    prompt = f"### 输入状态:\n{state_text}\n\n### 第一人称描述:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        temperature=0.8,
        do_sample=True,
        top_p=0.9
    )
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()

if __name__ == "__main__":
    model, tokenizer = load_model("./models/Qwen2.5-1.5B", "./models/expression_renderer_lora")
    test_state = {
        "major": 2, "middle": 1, "fine": 2,
        "valence": -0.6, "arousal": 0.8, "approach": -0.4,
        "L": 7, "stiffness": 0.5, "proj": 0.7, "nour": 0.2
    }
    print(generate_expression(model, tokenizer, test_state))