init:
	git submodule update --init --recursive

update:
	git update
	git submodule update --recursive --remote
