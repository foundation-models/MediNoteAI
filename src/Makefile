
VENV_DIR := ~/.venv

# MODEL_NAME and DATASET are passed as arguments and can be set in env variables
check-model-name:
ifndef MODEL_NAME
	$(error MODEL_NAME is undefined, set it via export MODEL_NAME=...)
endif

check-dataset:
ifndef DATASET
	$(error DATASET is undefined, set it via export DATASET=...)
endif

check-callback_url:
ifndef CALLBACK_URL
	$(error CALLBACK_URL is undefined, set it via export CALLBACK_URL=...)
endif


check-wandb_api_key:
ifndef WANDB_API_KEY
	$(error WANDB_API_KEY is undefined, set it via export WANDB_API_KEY=... or export WANDB_DISABLED=True)
endif

run-vllm-server: check-model-name
	make -j 2 run-embedding_api run-vllm
.PHONY: run-server

run-server: check-model-name
	make -j 2 run-embedding_api run-gptq-model
.PHONY: run-server

run-finetune-server: 
	make -j 2 run-embedding_api run-finetune_api
.PHONY: run-server

# Task for creating a virtual environment and installing requirements
create-venv:
	sudo apt install -y python3.10-venv
	python -m venv $(VENV_DIR)
	$(VENV_DIR)/bin/pip install -r text-generation-webui/requirements_cpu_only.txt
	# @echo "To activate the virtual environment, run source $(VENV_DIR)/bin/activate"
.PHONY: create-venv


install_requirements_cpu:
	pip install -r text-generation-webui/requirements_cpu_only.txt
.PHONY: install_requirements_cpu

install_requirements_gpu:
	pip install --upgrade pip
	pip install rootpath
	pip install -r text-generation-webui/requirements.txt
.PHONY: install_requirements_gpu

hello-world:
	cd text-generation-webui && \
	python3 download-model.py TheBloke/TinyLlama-1.1B-Chat-v0.3-GGUF && \
	python3 server.py --trust-remote-code --model TheBloke_TinyLlama-1.1B-Chat-v0.3-GGUF
.PHONY: download_test_model

run-webui:
	cd text-generation-webui && python3 server.py --share --trust-remote-code
.PHONY: run-webui

run-api:
	cd FastChat && python3 -m fastchat.serve.model_worker --no-register --trust-remote-code --api --listen --load-in-4bit --model-dir ../text-generation-webui/models --model zephyr-7b-beta --conv-template mistral --port 8888 --controller-address https://controller-llama.ai.dev1.intapp.com --worker-address https://llama.ai.dev1.intapp.com --device cuda --num-gpus 1 --model-names llama,gpt-3.5-turbo,text-davinci-003,text-embedding-ada-002 --use_fast True
.PHONY: run-api

run-finetune_api: check-callback_url check-wandb_api_key
	export PYTHONPATH=.:FastChat:langchain/libs/langchain:langchain/libs/core:langchain/libs/community && \
	python3 -m medinote.finetune.finetune_api --port 8888 --host 0.0.0.0
.PHONY: run-finetune_api


run-embedding_api:
	export PYTHONPATH=.:FastChat:langchain/libs/langchain:langchain/libs/core:langchain/libs/community:opencopilot && \
	python3 -m medinote.embedding.encode_api --port 7777 --host 0.0.0.0
.PHONY: run-embedding_api

download_model:
	cd text-generation-webui && python3 download-model.py nvidia/parakeet-rnnt-1.1b
.PHONY: download_model


run-command_line_model: check-model-name
	pip install prompt_toolkit rich && cd FastChat && python3 -m fastchat.serve.cli \
		--model-path /mnt/models/${MODEL_NAME} \
		--device cuda --num-gpus 1
.PHONY: run-command_line_model

run-vllm: check-model-name
	export PYTHONPATH=.:FastChat:langchain/libs/langchain:langchain/libs/core:langchain/libs/community:opencopilot && \
	python3 -m medinote.inference.vllm_worker_copilot \
		--model-path /mnt/models/${MODEL_NAME} \
		--no-register --host 0.0.0.0 --port 8888 
	# --model-path /mnt/models/stablelm-zephyr-3b --dtype float16 --tokenizer hf-internal-testing/llama-tokenizer
	# --model-path /mnt/models/openchat-3.5-1210-AWQ --quantization awq
.PHONY: run-vllm


run-controller:
	cd FastChat && python3 -m fastchat.serve.controller \
	--host 0.0.0.0 --port 21001 \
.PHONY: run-controller

run-openai_api:
	cd FastChat && python3 -m fastchat.serve.openai_api_server \
		--host 0.0.0.0 --port 8000 \
		--controller-address http://localhost:21001 
.PHONY: run-openai_api

# export MODEL_NAME=zephyr-7b-beta
# --load-8bit # may give bad results
run-7b-model: check-model-name
	cd FastChat && python3 -m fastchat.serve.model_worker \
		--dtype float16 \
		--no-register \
		--model-path /mnt/models/${MODEL_NAME} \
		--conv-template mistral \
		--host 0.0.0.0 --port 8888 \
		--controller-address http://localhost:21001 \
		--worker-address http://localhost:8000 
.PHONY: run-7b-model

      	# --gptq-ckpt models/${MODEL_NAME}/model.safetensors \
		# --gptq-wbits 4 \
		# --gptq-groupsize 128 \
		# --gptq-act-order # requires gptq-for-llama that is replaced by autogptq which is not supported by FastChat

# export MODEL_NAME=SOLAR-10.7B-Instruct-v1.0-GPTQ
# export MODEL_NAME=Sakura-SOLAR-Instruct-GPTQ # #1 in the leaderboard
# updated config.json and added "disable_exllama": true
run-gptq-model: check-model-name
	cd FastChat && python3 -m fastchat.serve.model_worker \
		--no-register \
		--device cuda \
        --num-gpus 1 \
		--model-path /mnt/models/${MODEL_NAME} \
		--conv-template mistral \
		--host 0.0.0.0 --port 8888 \
		--controller-address http://localhost:21001 \
		--worker-address http://localhost:8000 
.PHONY: run-gptq-model


run-3b-model: check-model-name
	cd FastChat && python3 -m fastchat.serve.model_worker \
		--model-path /mnt/models/${MODEL_NAME} \
		--conv-template mistral \
		--host 0.0.0.0 --port 8888 \
		--controller-address http://localhost:21001 \
		--worker-address http://localhost:8000 
.PHONY: run-model

convert_alpaca:
	# cd FastChat && python3 -m fastchat.data.convert_alpaca --in-file ../text-generation-webui/training/datasets/know_med_v4.sample-crop-repeat.json --out-file ../text-generation-webui/training/datasets/know_med_v4.sample-crop-repeat.alpaca.json
	cd FastChat && python3 -m fastchat.data.convert_alpaca --in-file ../text-generation-webui/training/datasets/huggingface/BI55-MedText/completion_training_text_conversations.json  --out-file ../text-generation-webui/training/datasets/huggingface/BI55-MedText/completion_training_text_conversations_alpaca.json
.PHONY: convert_alpaca


finetune_qlora: check-dataset check-model-name check-wandb_api_key
	export PYTHONPATH=.:FastChat:langchain/libs/langchain:langchain/libs/core:langchain/libs/community:$PYTHONPATH && \
	python3 -m medinote.finetune.finetune_lora \
		--model_name_or_path /mnt/models/${MODEL_NAME}  \
		--data_path /mnt/datasets/${DATASET}.jsonl \
		--output_dir /mnt/models/qlora_${DATASET} \
		--lora_target_modules_str Wqkv,out_proj \
		--fp16 True \
		--flash_attn True \
    	--flash_rotary True \
		--lora_r 32 \
		--lora_alpha 64 \
		--lora_dropout 0.05 \
		--bf16 False \
		--num_train_epochs 1 \
		--per_device_train_batch_size 1 \
		--per_device_eval_batch_size 1 \
		--gradient_accumulation_steps 1 \
		--evaluation_strategy "no" \
		--save_strategy "steps" \
		--save_steps 500 \
		--save_total_limit 2 \
		--learning_rate 2e-5 \
		--weight_decay 0. \
		--warmup_ratio 0.03 \
		--lr_scheduler_type "cosine" \
		--logging_steps 1 \
		--tf32 False \
		--model_max_length 2048 \
		--q_lora True \
        --ddp_find_unused_parameters False \
		--samples_start_index 0 \
		--samples_end_index 10000
.PHONY: finetune_qlora


deepspeed_qlora: check-dataset check-model-name check-wandb_api_key
	export PYTHONPATH=.:FastChat:langchain/libs/langchain:langchain/libs/core:langchain/libs/community:$PYTHONPATH && \
	deepspeed medinote/finetune/finetune_lora.py \
		--deepspeed FastChat/playground/deepspeed_config_s2.json \
		--model_name_or_path /mnt/models/${MODEL_NAME}  \
		--data_path /mnt/datasets/${DATASET}.training.jsonl \
		--output_dir /mnt/models/qlora_${DATASET}-deepspeed \
		--lora_target_modules_str Wqkv,out_proj \
		--fp16 True \
		--flash_attn True \
    	--flash_rotary True \
		--lora_r 32 \
		--lora_alpha 64 \
		--lora_dropout 0.05 \
		--bf16 False \
		--num_train_epochs 1 \
		--per_device_train_batch_size 1 \
		--per_device_eval_batch_size 1 \
		--gradient_accumulation_steps 1 \
		--evaluation_strategy "no" \
		--save_strategy "steps" \
		--save_steps 500 \
		--save_total_limit 2 \
		--learning_rate 2e-5 \
		--weight_decay 0. \
		--warmup_ratio 0.03 \
		--lr_scheduler_type "cosine" \
		--logging_steps 1 \
		--tf32 False \
		--model_max_length 2048 \
		--q_lora True \
        --ddp_find_unused_parameters False \
		--samples_start_index 70000 \
		--samples_end_index 120000 
.PHONY: deepspeed_qlora

apply_lora: check-dataset check-model-name
	export PYTHONPATH=. && \
	python3 -m medinote.finetune.apply_lora \
	--base-model-path /mnt/models/${MODEL_NAME} \
	--lora-path /mnt/models/qlora_${DATASET}-deepspeed_40000_70000  \
	--target-model-path /mnt/models/${MODEL_NAME}-${DATASET} \
	--trust-remote-code True 
.PHONY: merge_lora

run-llama_cpp:
	cd llama.cpp && make -j && ./main -m /mnt/models/${MODEL_NAME} -p "Building a website can be done in 10 simple steps:\nStep 1:" -n 400 -e
.PHONY: run-llama_cpp

convert:
	cd llama.cpp && python3 convert.py --outfile models/ericzzz_falcon-rw-1b-instruct-openorca-GGUF models/ericzzz_falcon-rw-1b-instruct-openorca 
.PHONY: convert

qualtize:
	cd llama.cpp && ./quantize ./models/TheBloke_Mistral-7B-Instruct-v0.2-GGUF ./models/Mistral-7B-Instruct-v0.2-q4_0-GGUF q4_0
.PHONY: quantize

take_first_1000:
	head -n 1000 /mnt/datasets/${DATASET} > /mnt/datasets/${DATASET}_1000
.PHONY: take_first_1000

check-install-postgres:
	@which psql > /dev/null 2>&1; \
	if [ $$? -eq 0 ]; then \
	    echo "PostgreSQL is already installed."; \
	else \
	    echo "PostgreSQL is not installed. Installing..."; \
	    sudo apt-get update; \
	    sudo apt-get install postgresql postgresql-contrib; \
	fi
.PHONY: check-install-postgres