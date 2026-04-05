# from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend, FineGrainedFP8Config

# model_id = "mistralai/Ministral-3-8B-Base-2512"
# model = Mistral3ForConditionalGeneration.from_pretrained(
#     model_id,
#     device_map="auto",
# )
# tokenizer = MistralCommonBackend.from_pretrained(model_id)

# input_ids = tokenizer.encode("Once about a time, France was a", return_tensors="pt")
# input_ids = input_ids.to("cuda")

# output = model.generate(
#     input_ids,
#     max_new_tokens=30,
# )[0]

# decoded_output = tokenizer.decode(output[len(input_ids[0]):])
# print(decoded_output)


import torch
from transformers import Mistral3ForConditionalGeneration, FineGrainedFP8Config, MistralCommonBackend

model_id = "mistralai/Ministral-3-8B-Instruct-2512"

tokenizer = MistralCommonBackend.from_pretrained(model_id)
model = Mistral3ForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    # quantization_config=FineGrainedFP8Config(dequantize=True)
    torch_dtype=torch.bfloat16,
)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Can you explain how transformer models work and why they are useful in natural language processing?",
            }
        ],
    },
]

tokenized = tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True)

tokenized["input_ids"] = tokenized["input_ids"].to(device="cuda")

output = model.generate(
    input_ids=tokenized["input_ids"],
    max_new_tokens=128,
)[0]

decoded_output = tokenizer.decode(output[len(tokenized["input_ids"][0]):])
print(decoded_output)
