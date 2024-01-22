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

