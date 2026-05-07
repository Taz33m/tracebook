.PHONY: setup install install-dev test test-coverage benchmark benchmark-latency benchmark-throughput profile dashboard demo format lint quality docs clean clean-all monitor help

# Python and virtual environment
PYTHON := python3
VENV := venv
PIP := $(VENV)/bin/pip
PYTHON_VENV := $(VENV)/bin/python

# Setup virtual environment and install dependencies
setup:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "Setup complete! Activate with: source $(VENV)/bin/activate"

# Install dependencies
install:
	$(PIP) install -r requirements.txt

# Install development dependencies
install-dev: install
	$(PIP) install -e ".[dev,dashboard]"

# Run all tests
test:
	$(PYTHON_VENV) -m pytest tests/ -v --cov=tracebook --cov-report=term-missing

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

# Profile with magic-trace (requires magic-trace installation)
profile:
	$(PYTHON_VENV) -m tracebook.simulation.simulation_engine --duration 5 --throughput 500 --algorithm FIFO --magic-trace

# Run the dashboard demo
dashboard:
	$(PYTHON_VENV) -m tracebook.visualization.dashboard --port 8050 --demo-simulation

# Run basic simulation example
demo:
	$(PYTHON_VENV) examples/full_simulation_demo.py

# Code formatting with black
format:
	$(PYTHON_VENV) -m black src/ tests/ examples/

# Lint code with flake8
lint:
	$(PYTHON_VENV) -m flake8 src/ tests/ examples/

# Run code quality checks used for the alpha CI gate
quality: format lint

# Generate documentation
docs:
	@echo "Generating documentation..."
	@mkdir -p docs/generated
	$(PYTHON_VENV) -c "import tracebook.core.orderbook; help(tracebook.core.orderbook)" > docs/generated/orderbook_help.txt

# Clean up generated files
clean:
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
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
	@echo "  profile         - Run simulation with magic-trace/fallback profiling"
	@echo "  dashboard       - Launch performance dashboard"
	@echo "  demo            - Run basic simulation demo"
	@echo "  format          - Format code with black"
	@echo "  lint            - Lint code with flake8"
	@echo "  quality         - Run all code quality checks"
	@echo "  docs            - Generate documentation"
	@echo "  clean           - Clean generated files"
	@echo "  clean-all       - Clean everything including venv"
	@echo "  monitor         - Run continuous performance monitoring"
	@echo "  help            - Show this help message"
