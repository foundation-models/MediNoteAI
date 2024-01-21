init:
	git submodule update --init --recursive
.PHONY: init

update:
	git fetch --all
	git pull
	git pull --recurse-submodules
.PHONY: update

remove-submodule:
	cd src && \
	git submodule deinit -f $1 && \
	git rm -f $1 && \
	rm -rf ../.git/modules/src/$1
.PHONY: remove-submodule


install:
	cd src && \
	make create-venv && \
	make check-install-postgres
.PHONY: install

docker:
	cd docker && docker compose up -d
.PHONY: docker

deploy:
	cd k8s && kubectl apply -f inferece
.PHONY: deploy

webui:
	cd webui/ollama-webui && \
	rm -f build && \
	ln -s /app/build build && \
	cd backend && \
	export OLLAMA_API_BASE_URL=llama-generative-ai:5000/api && \
	uvicorn main:app --host 0.0.0.0 --port 8888 --forwarded-allow-ips '*'
.PHONY: webui