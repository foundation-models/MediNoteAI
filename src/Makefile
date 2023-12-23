
install_requirements_cpu:
	pip install -r text-generation-webui/requirements_cpu_only.txt

install_requirements_gpu:
	pip install --upgrade pip
	pip install rootpath
	pip install -r text-generation-webui/requirements.txt

download_test_model:
	cd text-generation-webui && python3 download-model.py \
		TheBloke/TinyLlama-1.1B-Chat-v0.3-GGUF
		# TheBloke/stablelm-zephyr-3b-GGUF

run_webui:
	cd text-generation-webui && python3 server.py --share --trust-remote-code

run_api:
	cd FastChat && python3 -m fastchat.serve.model_worker --no-register --trust-remote-code --api --listen --load-in-4bit --model-dir ../text-generation-webui/models --model zephyr-7b-beta --conv-template mistral --port 8888 --controller-address https://controller-llama.ai.dev1.intapp.com --worker-address https://llama.ai.dev1.intapp.com --device cuda --num-gpus 1 --model-names llama,gpt-3.5-turbo,text-davinci-003,text-embedding-ada-002 --use_fast True

download_model:
	cd text-generation-webui && python3 download-model.py TheBloke/Mistral-7B-Instruct-v0.2-GGUF


run_model:
	cd FastChat && python3 -m fastchat.serve.cli --model-path /mnt/models/phi-2-doctor-patient-repeat --device cuda --num-gpus 1
	# cd FastChat && python3 -m fastchat.serve.cli --model-path /mnt/models/zephyr-7b-beta-ptient-doctor --device cuda --num-gpus 1 --dtype float16 --load-8bit --conv-template mistral  
	# cd FastChat && python3 -m fastchat.serve.cli --model-path /mnt/models/zephyr-7b-beta --device cuda --num-gpus 1 --dtype float16 --load-8bit --conv-template mistral  

run_vllm:
	cd FastChat && python3 -m fastchat.serve.vllm_worker --model-path /mnt/models/phi-2 --no-register --port 8888 
	# cd FastChat && python3 -m fastchat.serve.vllm_worker --model-path /mnt/models/phi-2-doctor-patient-repeat
	# cd FastChat && python3 -m fastchat.serve.vllm_worker --model-path /mnt/models/stablelm-zephyr-3b --dtype float16 --tokenizer hf-internal-testing/llama-tokenizer
	# cd FastChat && python3 -m fastchat.serve.vllm_worker --model-path /mnt/models/openchat-3.5-1210-AWQ --quantization awq

run_model2:
	cd FastChat && python3 -m fastchat.serve.cli --model-path /mnt/models/zephyr-7b-beta-patient-doctor --device cuda --num-gpus 1 --conv-template mistral  
	# cd FastChat && python3 -m fastchat.serve.cli --model-path /mnt/models/zephyr-7b-beta --device cuda --num-gpus 1 --dtype float16 --load-8bit --conv-template mistral  

convert_alpaca:
	cd FastChat && python3 -m fastchat.data.convert_alpaca --in-file ../text-generation-webui/training/datasets/know_med_v4.sample-crop-repeat.json --out-file ../text-generation-webui/training/datasets/know_med_v4.sample-crop-repeat.alpaca.json


finetune_qlora:
	cd FastChat && deepspeed fastchat/train/train_lora.py \
		--model_name_or_path ../text-generation-webui/models/zephyr-7b-beta \
		--lora_r 32 \
		--lora_alpha 64 \
		--lora_dropout 0.05 \
		--data_path ../text-generation-webui/training/datasets/know_med_v4.sample-crop-repeat.alpaca.json \
		--bf16 False \
		--output_dir ../text-generation-webui/loras/patient-doctor-crop-repeat \
		--num_train_epochs 1 \
		--per_device_train_batch_size 1 \
		--per_device_eval_batch_size 1 \
		--gradient_accumulation_steps 1 \
		--evaluation_strategy "no" \
		--save_strategy "steps" \
		--save_steps 200 \
		--save_total_limit 2 \
		--learning_rate 2e-5 \
		--weight_decay 0. \
		--warmup_ratio 0.03 \
		--lr_scheduler_type "cosine" \
		--logging_steps 1 \
		--tf32 False \
		--model_max_length 2048 \
		--q_lora True \
		--deepspeed playground/deepspeed_config_s2.json \
		--gradient_checkpointing True \
        --flash_attn False \
        --ddp_find_unused_parameters False


finetune_qlora2:
	cd FastChat && deepspeed fastchat/train/train_lora.py \
		--model_name_or_path ../text-generation-webui/models/phi-2 \
		--lora_r 8 \
		--lora_alpha 16 \
		--lora_dropout 0.05 \
		--data_path ../text-generation-webui/training/datasets/know_med_v4.sample-crop-repeat.alpaca.json \
		--bf16 False \
		--output_dir ../text-generation-webui/loras/phi-2-patient-doctor-crop-repeat \
		--num_train_epochs 3 \
		--per_device_train_batch_size 1 \
		--per_device_eval_batch_size 1 \
		--gradient_accumulation_steps 1 \
		--evaluation_strategy "no" \
		--save_strategy "steps" \
		--save_steps 1200 \
		--save_total_limit 100 \
		--learning_rate 2e-5 \
		--weight_decay 0. \
		--warmup_ratio 0.03 \
		--lr_scheduler_type "cosine" \
		--logging_steps 1 \
		--tf32 False \
		--model_max_length 2048 \
		--q_lora True \
		--deepspeed playground/deepspeed_config_s2.json \
		--gradient_checkpointing False \
        --flash_attn False \
        --ddp_find_unused_parameters False 



merge_lora:
	# cd FastChat && python3 -m fastchat.model.apply_lora --base-model-path ../text-generation-webui/models/zephyr-7b-beta --lora-path /home/agent/workspace/MediNoteAI/src/text-generation-webui/loras/patient-doctor-crop-repeat/checkpoint-200 --target-model-path /mnt/models/zephyr-7b-beta-patient-doctor-repeat
	cd FastChat && python3 -m fastchat.model.apply_lora --base-model-path ../text-generation-webui/models/phi-2 --lora-path /home/agent/workspace/MediNoteAI/src/text-generation-webui/loras/doctor-patient --target-model-path /mnt/models/phi-2-doctor-patient-repeat

run_llama_cpp:
	cd llama.cpp && make -j && ./main -m models/TheBloke_Mistral-7B-Instruct-v0.2-GGUF -p "Building a website can be done in 10 simple steps:\nStep 1:" -n 400 -e

convert:
	cd llama.cpp && python3 convert.py --outfile models/ericzzz_falcon-rw-1b-instruct-openorca-GGUF models/ericzzz_falcon-rw-1b-instruct-openorca 

qualtize:
	cd llama.cpp && ./quantize ./models/TheBloke_Mistral-7B-Instruct-v0.2-GGUF ./models/Mistral-7B-Instruct-v0.2-q4_0-GGUF q4_0