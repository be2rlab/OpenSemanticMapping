SHELL := /bin/bash

MKFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
MKFILE_DIR := $(dir $(MKFILE_PATH))
ROOT_DIR := $(MKFILE_DIR)

DATA_DIR ?= /home/sber20/dev/SBER/data

PARAMETERS := ROOT_DIR=$(ROOT_DIR) \
	DATA_DIR=$(DATA_DIR)

prepare-terminal-for-visualization:
	xhost local:docker

build-concept-graphs:
	cd $(ROOT_DIR) && \
	docker compose build concept_graphs

run-concept-graphs:
	cd $(ROOT_DIR) && \
	$(PARAMETERS) \
	docker compose run concept_graphs
