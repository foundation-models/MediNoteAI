fastchat:
	# cd ../src/FastChat/docker && docker build . -t foundationmodels/fastchat:base
	DOCKER_BUILDKIT=1 docker build -f Dockerfile.fastchat -t foundationmodels/fastchat:latest .

text-generation-webui:
	# cd ../src/text-generation-webui && DOCKER_BUILDKIT=1 docker build -f docker/nvidia/Dockerfile -t foundationmodels/text-generation-webui:fastchat .
	cp ../src/text-generation-webui/requirements.txt webui/requirements.txt 
	DOCKER_BUILDKIT=1 docker build -f Dockerfile.webui -t foundationmodels/text-generation-webui:agent .
.PHONY: text-generation-webui

vllm:
	DOCKER_BUILDKIT=1 docker build -f Dockerfile.vllm -t foundationmodels/vllm:cuda12 .
.PHONY: vllm


pipeline:
	DOCKER_BUILDKIT=1 docker build -f Dockerfile.pipeline -t foundationmodels/pipeline:base .
.PHONY: pipeline

nemo:
	DOCKER_BUILDKIT=1 docker build -f Dockerfile.nemo -t foundationmodels/nemo:base .
.PHONY: nemo

deepspeed:
	DOCKER_BUILDKIT=1 docker build -f Dockerfile.deepspeed -t foundationmodels/deepspeed:cuda12 .
.PHONY: deepspeed

django:
	DOCKER_BUILDKIT=1 docker build -f Dockerfile.django -t foundationmodels/django:ollama .
.PHONY: django

check-docker-access-token:
ifndef DOCKER_ACCESS_TOKEN
	$(error DOCKER_ACCESS_TOKEN is undefined, set it via export DOCKER_ACCESS_TOKEN=...)
endif
.PHONY: check-docker-access-token

login: check-docker-access-token
	rm -f ~/.docker/config.json
	docker login --username foundationmodels --password=${DOCKER_ACCESS_TOKEN}
.PHONY: login

prune:
	docker image prune -f
.PHONY: prune