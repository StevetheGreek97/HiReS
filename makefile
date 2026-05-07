# ============================================================
# HiReS Makefile
# Works on Linux, macOS, and Windows (with Git Bash or WSL)
# ============================================================

PYTHON = python3
VENV   = .venv
PY     = $(VENV)/bin/python
PIP    = $(PY) -m pip
BUILD  = $(PY) -m build
TWINE  = $(PY) -m twine

# Default goal
.DEFAULT_GOAL := help

.PHONY: help venv install run test clean distclean deps-publish build wheel sdist release-test release

# ------------------------------------------------------------
# Setup & Environment
# ------------------------------------------------------------

help:
	@echo ""
	@echo "HiReS — High-Resolution Image Segmentation Pipeline"
	@echo ""
	@echo "Available commands:"
	@echo "  make venv           Create a new virtual environment"
	@echo "  make install        Install HiReS in editable mode"
	@echo "  make run            Show CLI help (smoke test)"
	@echo "  make test           Import and print version"
	@echo "  make clean          Remove caches and build artifacts"
	@echo "  make distclean      Clean + remove dist/ and egg-info"
	@echo "  make deps-publish   Install build and twine in venv"
	@echo "  make build          Clean old dist, then build sdist+wheel"
	@echo "  make release-test   Upload fresh build to TestPyPI"
	@echo "  make release        Upload fresh build to PyPI"
	@echo ""
	@echo "Tips:"
	@echo "  export TWINE_USERNAME=__token__"
	@echo "  export TWINE_PASSWORD=pypi-xxxxxxxxxxxxxxxxxxxxxxxxxx"
	@echo ""

venv:
	@echo "Creating virtual environment..."
	$(PYTHON) -m venv $(VENV)
	@echo "Virtual environment created in $(VENV)"
	@echo "Activate it using:"
	@echo "  source $(VENV)/bin/activate"
	@$(PIP) install --upgrade pip

install:
	@echo "Installing HiReS in editable mode..."
	@$(PIP) install -e .
	@echo "Installed successfully!"

# ------------------------------------------------------------
# Run / Test
# ------------------------------------------------------------

run:
	@echo "Running HiReS CLI..."
	@$(VENV)/bin/hires --help

test:
	@echo "Testing import and version..."
	@$(PY) -c "import HiReS; print('HiReS version:', getattr(HiReS, '__version__', 'unknown'))"

# ------------------------------------------------------------
# Maintenance & Publishing
# ------------------------------------------------------------

clean:
	@echo "Cleaning caches..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	@echo "Clean complete"

distclean: clean
	@echo "Removing build artifacts..."
	rm -rf build dist *.egg-info
	@echo "Dist clean complete"

deps-publish:
	@echo "Installing build & twine..."
	@$(PIP) install --upgrade build twine
	@echo "Done."

# Build both sdist and wheel from a clean state
build: distclean deps-publish
	@echo "Building new distribution..."
	@$(BUILD)
	@echo "Artifacts in ./dist:"
	@ls -1 dist || true

# Convenience targets if you ever want them separately
wheel:
	@$(PIP) install --upgrade build
	@$(BUILD) --wheel

sdist:
	@$(PIP) install --upgrade build
	@$(BUILD) --sdist

# Upload to TestPyPI (recommended first)
release-test: build
	@echo "Uploading to TestPyPI..."
	@$(TWINE) upload --repository testpypi dist/*
	@echo "TestPyPI upload done."
	@echo "Verify: pip install --index-url https://test.pypi.org/simple --extra-index-url https://pypi.org/simple <your-project>"

# Upload to PyPI (production)
release: build
	@echo "Uploading to PyPI..."
	@$(TWINE) upload dist/*
	@echo "PyPI upload done."
