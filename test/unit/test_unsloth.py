from unsloth import FastLanguageModel
import torch
from transformers import TextStreamer

model, tokenizer = FastLanguageModel.from_pretrained(
    # model_name = "/mnt/models/llama-3-8b-Instruct-bnb-4bit",
    model_name = "/mnt/models/llama-3-8b-bnb-4bit",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)


    # model_name = "/mnt/models/llama_3_fine_tuned_experiment_oliver",

prompt = """Based on given instruction and context, generate an appropriate response

### Instruction:
{}

### Context:
{}

### Response:
{}
"""

inputs = tokenizer(
    [
        prompt.format(
            "Provide a detailed explanation of the events and circumstances that led to the outbreak of World War II.",  # instruction
            " The goal is to offer a clear and informative account of the factors, political decisions, and international tensions that played a crucial role in triggering World War II. Ensure that the explanation covers the period leading up to the war, key events, and the involvement of major nations",  # context
            " ",  # response
        )
    ] * 1,
    return_tensors="pt",
).to("cuda")


text_streamer = TextStreamer(tokenizer)


x = model.generate(**inputs, streamer=text_streamer, max_new_tokens=1028)
print(x)
print(tokenizer.decode(x[0], skip_special_tokens=True))

