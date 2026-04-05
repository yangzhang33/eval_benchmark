# # -*- coding: utf-8 -*-

# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
# model_path = "inceptionai/jais-13b-chat"

# prompt_eng = "### Instruction: Complete the conversation below between [|Human|] and [|AI|]:\n### Input: [|Human|] {Question}\n### Response: [|AI|]"
# prompt_ar = "### Instruction: أكمل المحادثة أدناه بين [|Human|] و [|AI|]:\n### Input: [|Human|] {Question}\n### Response: [|AI|]"

# device = "cuda" if torch.cuda.is_available() else "cpu"

# tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True)


# def get_response(text,tokenizer=tokenizer,model=model):
#     input_ids = tokenizer(text, return_tensors="pt").input_ids
#     inputs = input_ids.to(device)
#     input_len = inputs.shape[-1]
#     generate_ids = model.generate(
#         inputs,
#         top_p=0.9,
#         temperature=0.3,
#         max_length=2048-input_len,
#         min_length=input_len + 4,
#         repetition_penalty=1.2,
#         do_sample=True,
#     )
#     response = tokenizer.batch_decode(
#         generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
#     )[0]
#     response = response.split("### Response: [|AI|]")
#     return response


# ques= "ما هي عاصمة الامارات؟"
# text = prompt_ar.format_map({'Question':ques})
# print(get_response(text))

# ques = "What is the capital of UAE?"
# text = prompt_eng.format_map({'Question':ques})
# print(get_response(text))


# from transformers import AutoTokenizer, AutoModelForCausalLM

# # Load the model and tokenizer
# model_name = "inceptionai/Jais-2-8B-Chat"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# # Example Arabic prompt
# # system_prompt = "أجب باللغة العربية بطريقة رسمية وواضحة."
# # user_input = "ما هي عاصمة الإمارات؟"

# # Example English prompt
# system_prompt = "Answer in English in a formal and clear manner."
# user_input = "What is the capital of the UAE?"

# # Apply chat template (always)
# chat_text = tokenizer.apply_chat_template(
#     [
#         # {"role": "system", "content": system_prompt},
#         {"role": "user", "content": user_input}
#     ],
#     tokenize=False,
#     add_generation_prompt=True
# )

# # Tokenize and generate
# inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)
# inputs.pop("token_type_ids", None)
# outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)

# # Decode and print only the newly generated tokens
# generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
# print(tokenizer.decode(generated_tokens, skip_special_tokens=True))
# #عاصمة الإمارات العربية المتحدة هي أبوظبي.
