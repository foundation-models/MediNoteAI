from ctransformers import AutoModelForCausalLM

# Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.

# Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
# llm = AutoModelForCausalLM.from_pretrained("/mnt/ai-llm/models/Meta-Llama-3-8B-Instruct-GGUF", model_file="Meta-Llama-3-8B-Instruct.Q4_K_M.gguf", model_type="llama", gpu_layers=50)
# llm = AutoModelForCausalLM.from_pretrained("/mnt/ai-llm/models/Meta-Llama-3-8B-Instruct-GGUF", model_file="Meta-Llama-3-8B-Instruct.Q4_1.gguf", model_type="llama", gpu_layers=50)
# llm = AutoModelForCausalLM.from_pretrained("/mnt/ai-llm/models/Meta-Llama-3-8B-Instruct-GGUF", model_file="Meta-Llama-3-8B-Instruct.Q5_K_M.gguf", model_type="llama", gpu_layers=50)
llm = AutoModelForCausalLM.from_pretrained("/mnt/ai-llm/models/Meta-Llama-3-8B-Instruct-GGUF", model_file="llama-3-8b-instruct.Q2_K.gguf", model_type="llama", gpu_layers=50)

print(llm("AI is going to"))
