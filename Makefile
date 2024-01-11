init:
	git submodule update --init --recursive

update:
	git fetch --all
	git pull
	git pull --recurse-submodules

