fastchat:
	# cd ../src/FastChat/docker && docker build . -t foundationmodels/fastchat:base
	DOCKER_BUILDKIT=1 docker build -f Dockerfile.fastchat -t foundationmodels/fastchat:latest .
.PHONY: fastchat

text-generation-webui:
	# cd ../src/text-generation-webui && DOCKER_BUILDKIT=1 docker build -f docker/nvidia/Dockerfile -t foundationmodels/text-generation-webui:fastchat .
	cp ../src/text-generation-webui/requirements.txt webui/requirements.txt 
	DOCKER_BUILDKIT=1 docker build -f Dockerfile.webui -t foundationmodels/text-generation-webui:agent .
.PHONY: text-generation-webui

ollama-infinity-yolo:
	DOCKER_BUILDKIT=1 docker build -f Dockerfile.ollama-infinity-yolo -t foundationmodels/ollama-infinity-yolo:latest .
.PHONY: ollama-infinity-yolo

encoder:
	DOCKER_BUILDKIT=1 docker build -f Dockerfile.encoder -t foundationmodels/encoder:gte-large-en-v1.5 .
.PHONY: encoder


openhand:
	DOCKER_BUILDKIT=1 docker build -f Dockerfile.openhand -t foundationmodels/openhand:0.9 .
.PHONY: openhand


erpnext:
	DOCKER_BUILDKIT=1 docker build -f Dockerfile.erpnext -t foundationmodels/erpnext:0.1 .
.PHONY: erpnext


nv-embed:
	DOCKER_BUILDKIT=1 docker build -f Dockerfile.nv-embed -t foundationmodels/nv-embed:latest .
.PHONY: nv-embed

openai-vision-api:
	DOCKER_BUILDKIT=1 docker build -f Dockerfile.openai-vision-api -t foundationmodels/openai-vision-api:latest .
.PHONY: openai-vision-api


yolo:
	DOCKER_BUILDKIT=1 docker build -f Dockerfile.yolo -t foundationmodels/yolo:latest .
.PHONY: yolo



web-ui:
	DOCKER_BUILDKIT=1 docker build -f Dockerfile.web-ui -t foundationmodels/web-ui:latest .
.PHONY: web-ui

voice:
	DOCKER_BUILDKIT=1 docker build -f Dockerfile.voice -t foundationmodels/voice:latest .
.PHONY: voice

onnx:
	DOCKER_BUILDKIT=1 docker build -f Dockerfile.onnx -t foundationmodels/onnx:encoder .
.PHONY: onnx

vllm:
	DOCKER_BUILDKIT=1 docker build -f Dockerfile.vllm -t foundationmodels/vllm:latest .
.PHONY: vllm

pipeline:
	DOCKER_BUILDKIT=1 docker build -f Dockerfile.pipeline -t foundationmodels/pipeline:dvc .
.PHONY: pipeline

nemo:
	DOCKER_BUILDKIT=1 docker build -f Dockerfile.nemo -t foundationmodels/nemo:base .
.PHONY: nemo


unsloth:
	DOCKER_BUILDKIT=1 docker build -f Dockerfile.unsloth -t foundationmodels/unsloth:latest .
.PHONY: unsloth

npm-gpu:
	DOCKER_BUILDKIT=1 docker build -f Dockerfile.npm-gpu -t foundationmodels/npm-gpu:latest .
.PHONY: npm-gpu

docker-engine:
	DOCKER_BUILDKIT=1 docker build -f Dockerfile.docker-engine -t foundationmodels/docker-engine:base .
.PHONY: docker-engine



sqlcoder:
	DOCKER_BUILDKIT=1 docker build -f Dockerfile.sqlcoder -t foundationmodels/sqlcoder:latest .
.PHONY: sqlcoder

yolo:
	DOCKER_BUILDKIT=1 docker build -f Dockerfile.yolo -t foundationmodels/yolo:latest .
.PHONY: yolo


voice:
	DOCKER_BUILDKIT=1 docker build -f Dockerfile.voice -t foundationmodels/voice:latest .
.PHONY: voice

unsloth:
	DOCKER_BUILDKIT=1 docker build -f Dockerfile.unsloth -t foundationmodels/unsloth:latest .
.PHONY: unsloth

npm-gpu:
	DOCKER_BUILDKIT=1 docker build -f Dockerfile.npm-gpu -t foundationmodels/npm-gpu:latest .
.PHONY: npm-gpu

docker-engine:
	DOCKER_BUILDKIT=1 docker build -f Dockerfile.docker-engine -t foundationmodels/docker-engine:base .
.PHONY: docker-engine


sqlcoder:
	DOCKER_BUILDKIT=1 docker build -f Dockerfile.sqlcoder -t foundationmodels/sqlcoder:latest .
.PHONY: sqlcoder



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


magentic-one:
	DOCKER_BUILDKIT=1 docker build -f Dockerfile.magentic-one -t foundationmodels/magentic-one:latest .
.PHONY: magentic-one

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