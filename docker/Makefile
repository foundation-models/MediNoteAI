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


docker-engine:
	DOCKER_BUILDKIT=1 docker build -f Dockerfile.docker-engine -t foundationmodels/docker-engine:base .
.PHONY: docker-engine



sqlcoder:
	DOCKER_BUILDKIT=1 docker build -f Dockerfile.sqlcoder -t foundationmodels/sqlcoder:latest .
.PHONY: sqlcoder

yolo:
	DOCKER_BUILDKIT=1 docker build -f Dockerfile.yolo -t foundationmodels/yolo:latest .
.PHONY: yolo


deepspeed:
	DOCKER_BUILDKIT=1 docker build -f Dockerfile.deepspeed -t foundationmodels/deepspeed:latest .
.PHONY: deepspeed


deepspeed-llava:
	DOCKER_BUILDKIT=1 docker build -f Dockerfile.deepspeed-llava -t foundationmodels/deepspeed-llava:latest .
.PHONY: deepspeed-llava



base:
	DOCKER_BUILDKIT=1 docker build -f Dockerfile.base -t foundationmodels/base:latest .
.PHONY: base


workspace:
	DOCKER_BUILDKIT=1 docker build -f Dockerfile.workspace -t foundationmodels/workspace:latest .
.PHONY: workspace



scheduler:
	DOCKER_BUILDKIT=1 docker build -f Dockerfile.scheduler -t foundationmodels/scheduler:latest .
.PHONY: scheduler


django:
	DOCKER_BUILDKIT=1 docker build -f Dockerfile.django -t foundationmodels/django:base .
.PHONY: django

webui:
	docker build -f Dockerfile.webui -t foundationmodels/webui:gatsby .
.PHONY: webui
# cd ../src/autogen/samples/apps/autogen-studio && docker build -f ../../../../../docker/Dockerfile.webui -t foundationmodels/webui:autoagent .

restart:
	service docker stop
	rm ~/.docker/config.json
	service docker start
	echo "docker login -u foundationmodels"
.PHONY: restart

test: 
	docker run -it --rm foundationmodels/vllm:latest bash
.PHONY: test