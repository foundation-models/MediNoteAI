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

now=$(shell date +'%y.%m.%d.%H.%M')

backup:
	# cp -r src /mnt/backup/MediNoteAI_src_${now}
	cd src && \
	find . -name "*.log" -exec rm {} \; && \
	find . -name "*.pyc" -exec rm {} \; && \
	tar -czvf /mnt/backup/MediNoteAI_src_medinote_${now}.tar.gz Makefile medinote
.PHONY: backup

set-demo-env:
	sudo mkdir -p /app/MediNoteAI/src && \
	sudo chown -R 1000:100 /app/MediNoteAI/src && \
	cd /app/MediNoteAI/src && \
	[ -e FastChat ] || ln -s /mnt/backup/MediNoteAI_src_24.02.28.17.47/FastChat . && \
	[ -e opencopilot ] || ln -s /mnt/backup/MediNoteAI_src_24.02.28.17.47/opencopilot . 
	# tar -xvf /mnt/backup/MediNoteAI_src_medinote_24.02.28.17.57.tar.gz
.PHONY: set-demo-env