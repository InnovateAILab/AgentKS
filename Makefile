## Convenience Makefile for linting and formatting

.PHONY: format lint check

format:
	# Format Python files with Black then apply Ruff's auto-fixes
	black .
	ruff format . || true

autofix:
	# Attempt to automatically fix issues: Ruff fixes, then Black, then re-check
	ruff check --fix . || true
	black .
	ruff check .
	flake8 .

lint:
	# Static analysis: Ruff (fast) and Flake8 (for plugins/rules)
	ruff check .
	flake8 .

check:
	# CI-style checks: Ruff, Black (check mode), Flake8
	ruff check .
	black --check .
	flake8 .
