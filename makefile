# ============================================================
# HiReS Makefile
# Works on Linux, macOS, and Windows (with Git Bash or WSL)
# ============================================================

PYTHON = python3
VENV   = .venv
ACT    = . $(VENV)/bin/activate

# Default goal
.DEFAULT_GOAL := help

# ------------------------------------------------------------
#  Setup & Environment
# ------------------------------------------------------------

help:
	@echo ""
	@echo "HiReS — High-Resolution Image Segmentation Pipeline"
	@echo ""
	@echo "Available commands:"
	@echo "  make venv        Create and install dependencies into a new virtual env"
	@echo "  make install     Install HiReS in editable mode"
	@echo "  make run         Run the HiReS pipeline CLI (hires)"
	@echo "  make test        Quick test to verify install"
	@echo "  make clean       Remove cache/build artifacts"
	@echo ""

venv:
	@echo "Creating virtual environment..."
	$(PYTHON) -m venv $(VENV)
	@echo "Virtual environment created in $(VENV)"
	@echo "Activate it using:"
	@echo " source $(VENV)/bin/activate"
	@echo ""
	@$(ACT); pip install --upgrade pip

install:
	@echo "Installing HiReS in editable mode..."
	@$(ACT); pip install -e .
	@echo "Installed successfully!"

# ------------------------------------------------------------
# 🧪 Run / Test
# ------------------------------------------------------------

run:
	@echo "Running HiReS CLI..."
	@$(ACT); hires --help

test:
	@echo "Testing import and version..."
	@$(ACT); python -c "import HiReS; print('HiReS version:', getattr(HiReS, '__version__', 'unknown'))"

# ------------------------------------------------------------
# 🧹 Maintenance
# ------------------------------------------------------------

clean:
	@echo "Cleaning build artifacts..."
	rm -rf build dist *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} +
	@echo "Clean complete"
