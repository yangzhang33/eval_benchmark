import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "/datalake/datastore1/yang/_hf_models/Teuken-7B-base-v0.6"
prompt = "Insert text here..."
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)
inputs = tokenizer(prompt, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to the same device as the model
output = model.generate(input_ids=inputs['input_ids'], max_new_tokens=1000, do_sample=True)
result = tokenizer.decode(output.tolist())
print(result)
