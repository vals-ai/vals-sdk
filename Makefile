.PHONY: help install test style typecheck

ACTIVATE := @. .venv/bin/activate

help:
	@echo "Makefile for vals-sdk"
	@echo "Usage:"
	@echo "  make install        Install dependencies for development"
	@echo "  make test           Run tests"
	@echo "  make style          Lint & Format"
	@echo "  make typecheck      Typecheck"
	@echo "  make codegen        Generate GraphQL client"

install:
	uv venv
	uv sync --dev --group parsing
	@echo "🎉 Done! Run 'source .venv/bin/activate' to activate the virtualenv locally."

install-prod:
	uv venv
	uv sync

venv_check:
	@if [ ! -f .venv/bin/activate ]; then \
		echo "❌ Virtualenv not found! Run \`make install\` first."; \
		exit 1; \
	fi

test: venv_check
	$(ACTIVATE) && pytest

format: venv_check
	$(ACTIVATE) && ruff format .
lint: venv_check
	$(ACTIVATE) && ruff check --fix .
style: format lint

typecheck: venv_check
	$(ACTIVATE) && basedpyright
	
codegen: venv_check
	$(ACTIVATE) && ariadne-codegen --config codegen-config.toml
