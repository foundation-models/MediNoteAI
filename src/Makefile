
install_requirements:
	pip install -r text-generation-webui/requirements_cpu_only.txt

download_test_model:
	cd text-generation-webui && python download-model.py \
		TheBloke/stablelm-zephyr-3b-GGUF

run_webui:
	cd text-generation-webui && python server.py --share

download_model:
	cd text-generation-webui && python download-model.py TheBloke/Mistral-7B-Instruct-v0.2-GGUF


run_model:
	cd FastChat && python3 -m fastchat.serve.cli --model-path ../text-generation-webui/models/TheBloke_stablelm-zephyr-3b-GGUF 

run_llama_cpp:
	cd llama.cpp && make -j && ./main -m models/TheBloke_Mistral-7B-Instruct-v0.2-GGUF -p "Building a website can be done in 10 simple steps:\nStep 1:" -n 400 -e

convert:
	cd llama.cpp && python3 convert.py --outfile models/ericzzz_falcon-rw-1b-instruct-openorca-GGUF models/ericzzz_falcon-rw-1b-instruct-openorca 

qualtize:
	cd llama.cpp && ./quantize ./models/TheBloke_Mistral-7B-Instruct-v0.2-GGUF ./models/Mistral-7B-Instruct-v0.2-q4_0-GGUF q4_0