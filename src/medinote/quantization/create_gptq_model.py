
# pip install -U transformers accelerate
# pip install "transformers[sentencepiece]==4.32.1" "optimum==1.12.0" "auto-gptq==0.4.2" "accelerate==0.22.0" "safetensors>=0.3.1" - upgrade
# python -m vllm.entrypoints.api_server --model /mnt/models/llama-3-sqlcoder-8b-gptq


from accelerate import utils
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.gptq import GPTQQuantizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig


# quantizer = GPTQQuantizer(bits=4, dataset=dataset_id, model_seqlen=2048)
# quantizer.quant_method = "gptq"

model_id = "/mnt/models/llama-3-sqlcoder-8b"
# model = AutoModelForCausalLM.from_pretrained(model_id, config=quantizer, torch_dtype=torch.float16, max_memory = {0: "15GIB", 1: "15GIB"})
# tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
# quantized_model = quantizer.quantize_model(model, tokenizer)

tokenizer = AutoTokenizer.from_pretrained(model_id)
quantize_config = BaseQuantizeConfig(bits=4, group_size=128)
quantized_model = AutoGPTQForCausalLM.from_pretrained(model_id, quantize_config, torch_dtype=torch.float16, low_cpu_mem_usage=True)
# model = utils.convert_gpt_to_gptq(model)
# model.half()
token_texts = [
    tokenizer(
        "RAG is an AI framework for retrieving facts from an external knowledge base to ground large language models (LLMs) on the most accurate, up-to-date information and to give users insight into LLMs' generative process"
    )
]

quantized_model.quantize(token_texts)
quantized_model.save_pretrained("/mnt/models/llama-3-sqlcoder-8b-gptq")
