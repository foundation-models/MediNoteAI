init:
	git submodule update --init --recursive
.PHONY: init

update:
	git fetch --all
	git pull
	git pull --recurse-submodules
.PHONY: update

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
	uvicorn main:app --host 0.0.0.0 --port 8888 --forwarded-allow-ips '*'
