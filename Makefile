SHELL := /bin/bash

MKFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
MKFILE_DIR := $(dir $(MKFILE_PATH))
ROOT_DIR := $(MKFILE_DIR)

DOCKER_COMPOSE_FILES := \
	-f docker-compose.yaml

DATA_DIR ?= $(ROOT_DIR)/../data
ASSETS_DIR ?= $(ROOT_DIR)/../assets

PARAMETERS := ROOT_DIR=$(ROOT_DIR) \
	DATA_DIR=$(DATA_DIR) \
	ASSETS_DIR=$(ASSETS_DIR)

BUILD_COMMAND := ROOT_DIR=$(ROOT_DIR) docker compose $(DOCKER_COMPOSE_FILES) build

RUN_COMMAND := ROOT_DIR=$(ROOT_DIR) docker compose $(DOCKER_COMPOSE_FILES) up


prepare-terminal-for-visualization:
	xhost local:docker

build-images:
	@echo "Building images: $(IMAGES)"
	$(BUILD_COMMAND) $(IMAGES)

run-images: prepare-terminal-for-visualization
	@echo "Running images: $(IMAGES)"
	cd $(ROOT_DIR) && $(RUN_COMMAND) $(IMAGES)

build-concept-graphs:
	cd $(ROOT_DIR) && \
	$(PARAMETERS) \
	docker compose build concept_graphs 

run-concept-graphs:
	cd $(ROOT_DIR) && \
	$(PARAMETERS) \
	docker compose run concept_graphs

# build-concept-fusion:
# 	cd $(ROOT_DIR) && \
# 	docker compose build concept_fusion

# run-concept-fusion:
# 	cd $(ROOT_DIR) && \
# 	$(PARAMETERS) \
# 	docker compose run concept_fusion
