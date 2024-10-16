from time import time
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from optimum.gptq import GPTQQuantizer, load_quantized_model
import torch
from accelerate import init_empty_weights

sentence = "What is Genomics?"
model_name = "/mnt/models/Mixtral-8x7B-v0.1-GPTQ"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

# # quantizer = GPTQQuantizer(bits=4, dataset="c4", block_name_to_quantize = "model.decoder.layers", model_seqlen = 2048)
# quantizer = GPTQQuantizer(bits=4, dataset="wikitext2", use_cuda_fp16=True)

# quantized_model = quantizer.quantize_model(model, tokenizer)

config = AutoConfig.from_pretrained(model_name)
#config.quantization_config["use_exllama"] = True
config.quantization_config["use_exllama"] = True
config.quantization_config["exllama_config"] = {"version":2}

save_folder = "/mnt/models/Mixtral-8x7B-v0.1-GPTQ"

with init_empty_weights():
    empty_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="cuda:0", config=config)
empty_model.tie_weights()
# quantized_model = load_quantized_model(empty_model, save_folder=save_folder, device_map="auto")
quantized_model = load_quantized_model(empty_model, save_folder=save_folder, device_map="auto")
quantized_model.eval()
print("Start inference ....")
start_time = time()
test_ids = tokenizer(sentence, return_tensors="pt").to("cuda:0").input_ids
# beam_output = quantized_model.generate(test_ids, do_sample=True, num_beams=1, max_length=256, no_repeat_ngram_size=2, early_stopping=True)
beam_output = quantized_model.generate(test_ids, do_sample=True, num_beams=5, max_length=256, no_repeat_ngram_size=2, early_stopping=True)
print(tokenizer.batch_decode(beam_output, skip_special_tokens=True))
end_time = time()
print("Inference time: ", (end_time - start_time))

