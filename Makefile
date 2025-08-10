# Astiron Spectra Makefile

.PHONY: help build run clean ui test lint format install-dev

# Default target
help:
	@echo "Astiron Spectra - Spectral-Spatial Change Intelligence"
	@echo ""
	@echo "Available targets:"
	@echo "  build      - Build all Docker services"
	@echo "  run        - Run inference pipeline (requires RUN_ID)"
	@echo "  clean      - Clean up containers and volumes"
	@echo "  ui         - Start UI development server"
	@echo "  test       - Run all tests"
	@echo "  lint       - Run linting"
	@echo "  format     - Format code"
	@echo "  install-dev - Install development dependencies"
	@echo ""
	@echo "Examples:"
	@echo "  make build"
	@echo "  make run RUN_ID=20241201_143022"
	@echo "  make ui"

# Build all services
build:
	@echo "Building Astiron Spectra services..."
	docker compose build --parallel

# Run inference pipeline
run:
	@if [ -z "$(RUN_ID)" ]; then \
		echo "Error: RUN_ID is required. Example: make run RUN_ID=20241201_143022"; \
		exit 1; \
	fi
	@echo "Running inference pipeline with RUN_ID=$(RUN_ID)"
	@export RUN_ID=$(RUN_ID) && \
	docker compose run --rm preprocess && \
	docker compose run --rm detect_hsi && \
	docker compose run --rm detect_tir && \
	docker compose run --rm characterize && \
	docker compose run --rm fuse && \
	docker compose run --rm postprocess
	@echo "Pipeline complete. Results in output/runs/$(RUN_ID)/"

# Clean up
clean:
	@echo "Cleaning up containers and volumes..."
	docker compose down -v --remove-orphans
	docker system prune -f

# Start UI development server
ui:
	@echo "Starting Astiron Spectra Studio..."
	cd ui && npm install && npm run build:index && npm run dev

# Run tests
test:
	@echo "Running tests..."
	pytest tests/ -v --cov=common --cov=train --cov=infer
	cd ui && npm test

# Lint code
lint:
	@echo "Running linting..."
	flake8 common/ train/ infer/ --max-line-length=88 --extend-ignore=E203,W503
	black --check common/ train/ infer/
	cd ui && npm run lint

# Format code
format:
	@echo "Formatting code..."
	black common/ train/ infer/
	isort common/ train/ infer/
	cd ui && npm run format

# Install development dependencies
install-dev:
	@echo "Installing development dependencies..."
	pip install -r requirements-dev.txt
	cd ui && npm install

# Demo workflow
demo:
	@echo "Running demo workflow..."
	bash scripts/demo_seed.sh
	$(MAKE) build
	$(MAKE) run RUN_ID=demo_$(shell date +%Y%m%d_%H%M%S)

# Generate submission package
submission:
	@echo "Generating submission package..."
	bash scripts/compute_md5.sh
	bash scripts/pack_submission.sh --team "ASTIRON" --date $(shell date +%d-%b-%Y)

# Development helpers
dev-setup: install-dev
	@echo "Setting up development environment..."
	pre-commit install
	@echo "Development environment ready!"

# Quick smoke test
smoke-test:
	@echo "Running smoke test..."
	docker compose build preprocess
	docker compose run --rm preprocess --help
	@echo "Smoke test passed!"