.PHONY: install test benchmark profile clean docs setup

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
	$(PIP) install -e .

# Run all tests
test:
	$(PYTHON_VENV) -m pytest tests/ -v --cov=src --cov-report=term-missing

# Run tests with HTML coverage report
test-coverage:
	$(PYTHON_VENV) -m pytest tests/ -v --cov=src --cov-report=html
	@echo "Coverage report generated in htmlcov/index.html"

# Run benchmarks
benchmark:
	$(PYTHON_VENV) -m pytest tests/benchmarks/ -v --benchmark-only

# Run latency benchmarks specifically
benchmark-latency:
	$(PYTHON_VENV) -m pytest tests/benchmarks/latency_benchmark.py -v --benchmark-only

# Run throughput benchmarks specifically
benchmark-throughput:
	$(PYTHON_VENV) -m pytest tests/benchmarks/throughput_benchmark.py -v --benchmark-only

# Profile with magic-trace (requires magic-trace installation)
profile:
	$(PYTHON_VENV) examples/advanced_profiling.py

# Run the dashboard demo
dashboard:
	$(PYTHON_VENV) examples/dashboard_demo.py

# Run basic simulation example
demo:
	$(PYTHON_VENV) examples/basic_simulation.py

# Code formatting with black
format:
	$(PYTHON_VENV) -m black src/ tests/ examples/

# Lint code with flake8
lint:
	$(PYTHON_VENV) -m flake8 src/ tests/ examples/

# Type checking with mypy
typecheck:
	$(PYTHON_VENV) -m mypy src/

# Run all code quality checks
quality: format lint typecheck

# Generate documentation
docs:
	@echo "Generating documentation..."
	@mkdir -p docs/generated
	$(PYTHON_VENV) -c "import src.core.orderbook; help(src.core.orderbook)" > docs/generated/orderbook_help.txt

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
	$(PYTHON_VENV) -c "from src.simulation.market_simulator import MarketSimulator; MarketSimulator().run_continuous_monitoring()"

# Create project structure
create-structure:
	mkdir -p src/core src/algorithms src/profiling src/simulation src/visualization src/utils
	mkdir -p tests/benchmarks
	mkdir -p examples docs
	touch src/__init__.py src/core/__init__.py src/algorithms/__init__.py
	touch src/profiling/__init__.py src/simulation/__init__.py src/visualization/__init__.py
	touch src/utils/__init__.py tests/__init__.py tests/benchmarks/__init__.py

# Help
help:
	@echo "Available commands:"
	@echo "  setup           - Create virtual environment and install dependencies"
	@echo "  install         - Install dependencies"
	@echo "  install-dev     - Install in development mode"
	@echo "  test            - Run all tests with coverage"
	@echo "  test-coverage   - Run tests with HTML coverage report"
	@echo "  benchmark       - Run all benchmarks"
	@echo "  benchmark-latency - Run latency benchmarks only"
	@echo "  benchmark-throughput - Run throughput benchmarks only"
	@echo "  profile         - Run profiling example"
	@echo "  dashboard       - Launch performance dashboard"
	@echo "  demo            - Run basic simulation demo"
	@echo "  format          - Format code with black"
	@echo "  lint            - Lint code with flake8"
	@echo "  typecheck       - Type check with mypy"
	@echo "  quality         - Run all code quality checks"
	@echo "  docs            - Generate documentation"
	@echo "  clean           - Clean generated files"
	@echo "  clean-all       - Clean everything including venv"
	@echo "  monitor         - Run continuous performance monitoring"
	@echo "  create-structure - Create project directory structure"
	@echo "  help            - Show this help message"
