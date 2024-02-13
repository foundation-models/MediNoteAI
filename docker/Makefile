fastchat:
	# cd ../src/FastChat/docker && docker build . -t foundationmodels/fastchat:base
	DOCKER_BUILDKIT=1 docker build -f Dockerfile.fastchat -t foundationmodels/fastchat:latest .
.PHONY: fastchat

text-generation-webui:
	# cd ../src/text-generation-webui && DOCKER_BUILDKIT=1 docker build -f docker/nvidia/Dockerfile -t foundationmodels/text-generation-webui:fastchat .
	cp ../src/text-generation-webui/requirements.txt webui/requirements.txt 
	DOCKER_BUILDKIT=1 docker build -f Dockerfile.webui -t foundationmodels/text-generation-webui:agent .
.PHONY: text-generation-webui

vllm:
	DOCKER_BUILDKIT=1 docker build -f Dockerfile.vllm -t foundationmodels/vllm:latest .
.PHONY: vllm

deepspeed:
	DOCKER_BUILDKIT=1 docker build -f Dockerfile.deepspeed -t foundationmodels/deepspeed:cuda12 .
.PHONY: deepspeed

django:
	DOCKER_BUILDKIT=1 docker build -f Dockerfile.django -t foundationmodels/django:base .
.PHONY: django

webui:
	cd ../webui/autogen/samples/apps/autogen-studio && docker build -f ../../../../../docker/Dockerfile.webui -t foundationmodels/webui:autoagent .
.PHONY: webui

restart:
	service docker stop
	rm ~/.docker/config.json
	service docker start
	echo "docker login -u foundationmodels"
.PHONY: restart

test: 
	docker run -it --rm foundationmodels/vllm:latest bash
.PHONY: test