fastchat:
	# cd ../src/FastChat/docker && docker build . -t foundationmodels/fastchat:base
	DOCKER_BUILDKIT=1 docker build -f Dockerfile.fastchat -t foundationmodels/fastchat:latest .

text-generation-webui:
	# cd ../src/text-generation-webui && DOCKER_BUILDKIT=1 docker build -f docker/nvidia/Dockerfile -t foundationmodels/text-generation-webui:fastchat .
	cp ../src/text-generation-webui/requirements.txt webui/requirements.txt 
	DOCKER_BUILDKIT=1 docker build -f Dockerfile.webui -t foundationmodels/text-generation-webui:agent .

vllm:
	DOCKER_BUILDKIT=1 docker build -f Dockerfile.vllm -t foundationmodels/vllm:cuda12 .

deepspeed:
	DOCKER_BUILDKIT=1 docker build -f Dockerfile.deepspeed -t foundationmodels/deepspeed:cuda12 .

django:
	DOCKER_BUILDKIT=1 docker build -f Dockerfile.django -t foundationmodels/django:base .

