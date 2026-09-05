.PHONY: setup install install-dev test test-coverage benchmark benchmark-latency benchmark-throughput corpus-verify profile dashboard demo replay-demo format format-check lint typecheck security compile build verify-distribution-split quality clean clean-all monitor help

# Python and virtual environment
PYTHON := python3
VENV := venv
PIP := $(VENV)/bin/pip
PYTHON_VENV := $(VENV)/bin/python

# Setup a development environment with the implementation owner and compatibility facade
setup:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip "setuptools==84.0.0" "wheel==0.48.0"
	$(PIP) install -e ".[dev]"
	$(PIP) install -e "./packaging/tracebook-sim[dashboard,analysis,capture]"
	@echo "Setup complete! Activate with: source $(VENV)/bin/activate"

# Install both public distributions for local use
install:
	$(PIP) install -e .
	$(PIP) install -e ./packaging/tracebook-sim

# Install development dependencies
install-dev:
	$(PIP) install --upgrade "setuptools==84.0.0" "wheel==0.48.0"
	$(PIP) install -e ".[dev]"
	$(PIP) install -e "./packaging/tracebook-sim[dashboard,analysis,capture]"

# Run all tests
test:
	$(PYTHON_VENV) -m pytest tests/ -v --cov=tracebook --cov-report=term-missing --cov-fail-under=75

# Run tests with HTML coverage report
test-coverage:
	$(PYTHON_VENV) -m pytest tests/ -v --cov=tracebook --cov-report=html
	@echo "Coverage report generated in htmlcov/index.html"

# Run benchmarks
benchmark:
	$(PYTHON_VENV) -m tracebook.benchmarks.runner --scenario smoke --output benchmark_results/smoke.json

# Run latency benchmarks specifically
benchmark-latency:
	$(PYTHON_VENV) -m tracebook.benchmarks.runner --scenario fifo_baseline --throughput 500 --output benchmark_results/fifo-latency.json

# Run throughput benchmarks specifically
benchmark-throughput:
	$(PYTHON_VENV) -m tracebook.benchmarks.runner --scenario fifo_baseline --throughput 2000 --output benchmark_results/fifo-throughput.json

# Reproduce the bundled corpus and golden final state
corpus-verify:
	$(PYTHON_VENV) -m tracebook.corpus.cli verify src/tracebook/corpus/fixtures/coinbase-btcusd-synthetic-v1

# Profile with magic-trace (requires magic-trace installation)
profile:
	$(PYTHON_VENV) -m tracebook.simulation.simulation_engine --duration 5 --throughput 500 --algorithm FIFO --magic-trace

# Run the dashboard demo
dashboard:
	$(PYTHON_VENV) -m tracebook.visualization.dashboard --port 8050 --demo-simulation

# Run basic simulation example
demo:
	$(PYTHON_VENV) examples/full_simulation_demo.py

# Replay the bundled normalized order-event sample
replay-demo:
	$(PYTHON_VENV) -m tracebook.events.cli examples/data/sample_events.jsonl --output /tmp/tracebook-replay.json

# Code formatting with black
format:
	$(PYTHON_VENV) -m black src/ tests/ examples/ integrations/ experiments/ tools/

# Check formatting without modifying files
format-check:
	$(PYTHON_VENV) -m black --check src/ tests/ examples/ integrations/ experiments/ tools/

# Lint code with flake8
lint:
	$(PYTHON_VENV) -m flake8 src/ tests/ examples/ integrations/ experiments/ tools/

typecheck:
	$(PYTHON_VENV) -m mypy --python-version "$$($(PYTHON_VENV) -c 'import sys; print("%d.%d" % sys.version_info[:2])')" src/tracebook experiments tools

security:
	$(PYTHON_VENV) -m bandit -q -r src integrations tools

compile:
	$(PYTHON_VENV) -m compileall -q -x 'experiments/private/' src tests examples integrations experiments tools

build:
	$(PYTHON_VENV) -m build --sdist --wheel --outdir dist/conformance .
	$(PYTHON_VENV) -m build --sdist --wheel --outdir dist/simulator packaging/tracebook-sim
	$(PYTHON_VENV) tools/verify_distribution_privacy.py dist/conformance/* dist/simulator/*
	$(PYTHON_VENV) -m twine check dist/conformance/* dist/simulator/*
	$(PYTHON_VENV) tools/verify_sdist_wheel_agreement.py \
		--conformance-wheel dist/conformance/*.whl \
		--conformance-sdist dist/conformance/*.tar.gz \
		--sim-wheel dist/simulator/*.whl \
		--sim-sdist dist/simulator/*.tar.gz \
		--expected-version 0.6.0

# Prove wheel ownership, clean installs, migration, and uninstall safety
verify-distribution-split: build
	$(PYTHON_VENV) tools/verify_distribution_split.py \
		--expected-version 0.6.0 \
		--conformance-wheel dist/conformance/*.whl \
		--sim-wheel dist/simulator/*.whl

# Run the core quality gates used by CI
quality: format-check lint typecheck security compile test

# Clean up generated files
clean:
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf build/
	rm -rf packaging/tracebook-sim/build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf src/*.egg-info/
	rm -rf packaging/tracebook-sim/*.egg-info/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

# Clean everything including virtual environment
clean-all: clean
	rm -rf $(VENV)/

# Performance monitoring (long-running)
monitor:
	$(PYTHON_VENV) -m tracebook.simulation.simulation_engine --duration 60 --throughput 500 --algorithm FIFO

# Help
help:
	@echo "Available commands:"
	@echo "  setup           - Create virtual environment and install dependencies"
	@echo "  install         - Install dependencies"
	@echo "  install-dev     - Install in development mode"
	@echo "  test            - Run all tests with coverage"
	@echo "  test-coverage   - Run tests with HTML coverage report"
	@echo "  benchmark       - Run reproducible benchmark smoke scenario"
	@echo "  benchmark-latency - Run reproducible FIFO latency baseline"
	@echo "  benchmark-throughput - Run reproducible FIFO throughput baseline"
	@echo "  corpus-verify   - Verify the bundled synthetic Coinbase corpus"
	@echo "  profile         - Run simulation with magic-trace/fallback profiling"
	@echo "  dashboard       - Launch performance dashboard"
	@echo "  demo            - Run basic simulation demo"
	@echo "  replay-demo     - Replay the bundled normalized event sample"
	@echo "  format          - Format code with black"
	@echo "  format-check    - Check code formatting"
	@echo "  lint            - Lint code with flake8"
	@echo "  typecheck       - Type-check the package"
	@echo "  security        - Run Bandit security checks"
	@echo "  compile         - Compile source and tests"
	@echo "  build            - Build and validate both public distributions"
	@echo "  verify-distribution-split - Check ownership and clean-install contracts"
	@echo "  quality         - Run all code quality checks"
	@echo "  clean           - Clean generated files"
	@echo "  clean-all       - Clean everything including venv"
	@echo "  monitor         - Run continuous performance monitoring"
	@echo "  help            - Show this help message"
