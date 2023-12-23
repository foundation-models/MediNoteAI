fastchat:
	# cd ../src/FastChat/docker && docker build . -t foundationmodels/fastchat:base
	DOCKER_BUILDKIT=1 docker build -f Docerfile.fastchat -t foundationmodels/fastchat:latest .

text-generation-webui:
	# cd ../src/text-generation-webui && DOCKER_BUILDKIT=1 docker build -f docker/nvidia/Dockerfile -t foundationmodels/text-generation-webui:fastchat .
	cp ../src/text-generation-webui/requirements.txt webui/requirements.txt 
	DOCKER_BUILDKIT=1 docker build -f Dockerfile.webui -t foundationmodels/text-generation-webui:fastchat .

vllm:
	DOCKER_BUILDKIT=1 docker build -f Dockerfile.vllm -t foundationmodels/text-generation-webui:vllm .