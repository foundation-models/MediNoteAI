init:
	git submodule update --init --recursive

update:
	git pull
	git submodule update --recursive --remote
