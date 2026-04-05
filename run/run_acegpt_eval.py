from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the model and tokenizer
model_name = "FreedomIntelligence/AceGPT-v2-8B-Chat"
model_name = "FreedomIntelligence/AceGPT-v2-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# Raw prompt using AceGPT's native <User>/<Assistant> format

# Arabic prompt
# prompt = "<User>: فيما يلي أسئلة الاختيار من متعدد حول جبر تجريدي\n\nسؤال: ما هو الدرجة للامتداد الميداني الناتج من Q(sqrt(2), sqrt(3), sqrt(18)) على Q؟\nA. 0\nB. 4\nC. 2\nD. 6\nمن فضلك اختر إجابة واحدة من بين 'A، B، C، D' دون شرح. <Assistant>: "

# English prompt
prompt = "<User>: The following are multiple choice questions about abstract algebra\n\nQuestion: What is the degree of the field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q?\nA. 0\nB. 4\nC. 2\nD. 6\nPlease choose a single answer from 'A, B, C, D' without explanation. <Assistant>: "
prompt = "What is the degree of the field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q?\nA. 0\nB. 4\nC. 2\nD. 6\nPlease choose a single answer from 'A, B, C, D' without explanation."

# Tokenize and generate
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
inputs.pop("token_type_ids", None)
outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)

# Decode and print only the newly generated tokens
generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
print(tokenizer.decode(generated_tokens, skip_special_tokens=True))
