# NOTE: This makefile exists to test the Dockerfile build.
DEF_TAG='orquestra-core:latest'

.PHONY: build
build: Dockerfile
	DOCKER_BUILDKIT=0 docker build \
	--tag $(DEF_TAG) .

.PHONY: debug
debug: Dockerfile
	DOCKER_BUILDKIT=1 docker build \
	--progress plain \
	--ssh default --tag $(DEF_TAG) .
