# from ctransformers import AutoModelForCausalLM

# # Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.

# # Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
# # llm = AutoModelForCausalLM.from_pretrained("/mnt/ai-llm/models/Meta-Llama-3-8B-Instruct-GGUF", model_file="Meta-Llama-3-8B-Instruct.Q4_K_M.gguf", model_type="llama", gpu_layers=50)
# # llm = AutoModelForCausalLM.from_pretrained("/mnt/ai-llm/models/Meta-Llama-3-8B-Instruct-GGUF", model_file="Meta-Llama-3-8B-Instruct.Q4_1.gguf", model_type="llama", gpu_layers=50)
# # llm = AutoModelForCausalLM.from_pretrained("/mnt/ai-llm/models/Meta-Llama-3-8B-Instruct-GGUF", model_file="Meta-Llama-3-8B-Instruct.Q5_K_M.gguf", model_type="llama", gpu_layers=50)
# llm = AutoModelForCausalLM.from_pretrained("/mnt/ai-llm/models/Meta-Llama-3-8B-Instruct-GGUF", model_file="llama-3-8b-instruct.Q2_K.gguf", model_type="llama", gpu_layers=50)

# print(llm("AI is going to"))


# python -m vllm.entrypoints.openai.api_server --model /mnt/ai-llm/models/Llama-3-8B-Instruct-GPTQ-8-Bit --max-model-len 8192 --dtype float16


import transformers
import torch

model_id = "/mnt/ai-llm/models/Meta-Llama-3-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",
)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

prompt = pipeline.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
)

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = pipeline(
    prompt,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
print(outputs[0]["generated_text"][len(prompt):])