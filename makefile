# HiReS development tasks. Run `make help` for a list.
# Uses the interpreter of the active environment; override with `make PYTHON=...`.

PYTHON ?= python

.DEFAULT_GOAL := help
.PHONY: help install lint fix test check examples docs docs-serve build clean release-test

help: ## Show this list
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  make %-13s %s\n", $$1, $$2}'

install: ## Install HiReS in editable mode with the dev and docs extras
	$(PYTHON) -m pip install -e ".[dev,docs]"

lint: ## Check the code with ruff
	$(PYTHON) -m ruff check .

fix: ## Apply the safe ruff fixes
	$(PYTHON) -m ruff check --fix .

test: ## Run the test suite
	$(PYTHON) -m pytest

check: lint test ## Run everything CI runs

examples: ## Re-run the >>> examples in docs/ and rewrite their output and figures
	$(PYTHON) tools/doc_examples.py

docs: ## Build the documentation in strict mode
	$(PYTHON) -m mkdocs build --strict

docs-serve: ## Serve the documentation with live reload
	$(PYTHON) -m mkdocs serve

build: clean ## Build the sdist and wheel into dist/
	$(PYTHON) -m pip install --quiet --upgrade build twine
	$(PYTHON) -m build
	$(PYTHON) -m twine check dist/*

clean: ## Remove build artifacts and caches
	rm -rf build dist site *.egg-info .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} +

release-test: build ## Upload the build to TestPyPI (releases to PyPI are made by CI)
	$(PYTHON) -m twine upload --repository testpypi dist/*
