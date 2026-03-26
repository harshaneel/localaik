SHELL := /bin/bash

GO ?= go
IMAGE ?= gokhalh/localaik:test
PORT ?= 18090
CONTAINER_NAME ?= localaik-dev
BUILD_IMAGE ?= 1
GOCACHE ?= $(CURDIR)/.cache/go-build
GOFILES := $(shell find cmd internal integration -name '*.go' -type f | sort)

export GOCACHE

.PHONY: help fmt fmt-check lint test-unit test-integration test build docker-build docker-up docker-down

help:
	@printf '%s\n' \
		'make fmt                    Format Go sources' \
		'make fmt-check              Fail if Go sources are not formatted' \
		'make lint                   Run formatting check and go vet' \
		'make test-unit              Run unit tests under cmd/ and internal/' \
		'make test-integration       Run all integration tests; assumes docker-up is already running' \
		'make test                   Run lint, unit tests, and integration tests' \
		'make build                  Build the localaik binary' \
		'make docker-build           Build the Docker image' \
		'make docker-up              Start the Docker image on PORT' \
		'make docker-down            Stop and remove the Docker container'

fmt:
	@gofmt -w $(GOFILES)

fmt-check:
	@test -z "$$(gofmt -l $(GOFILES))"

lint: fmt-check
	@$(GO) vet ./...

test-unit:
	@$(GO) test ./cmd/... ./internal/...

test-integration:
	@$(GO) test -count=1 -tags=docker_integration ./integration

test: lint test-unit test-integration

build:
	@$(GO) build ./cmd/localaik

docker-build:
	@docker build -t "$(IMAGE)" .

docker-up:
	@if [[ "$(BUILD_IMAGE)" == "1" ]]; then $(MAKE) docker-build IMAGE="$(IMAGE)"; fi
	@docker rm -f "$(CONTAINER_NAME)" >/dev/null 2>&1 || true
	@docker run -d --name "$(CONTAINER_NAME)" -p "$(PORT):8090" "$(IMAGE)" >/dev/null
	@echo "localaik is starting on http://127.0.0.1:$(PORT)"
	@for _ in $$(seq 1 90); do \
		if curl -fsS "http://127.0.0.1:$(PORT)/health" >/dev/null 2>&1; then \
			echo "localaik is ready on http://127.0.0.1:$(PORT)"; \
			exit 0; \
		fi; \
		sleep 2; \
	done; \
	echo "localaik did not become healthy on http://127.0.0.1:$(PORT)" >&2; \
	docker logs "$(CONTAINER_NAME)" >&2 || true; \
	exit 1

docker-down:
	@docker rm -f "$(CONTAINER_NAME)" >/dev/null 2>&1 || true
