fastchat:
	cd FastChat && docker build . -t foundationmodels/fastchat:base

webui:
	cd ../src/text-generation-webui && DOCKER_BUILDKIT=1 docker build -f docker/nvidia/Dockerfile -t foundationmodels/text-generation-webui:nvidia .
