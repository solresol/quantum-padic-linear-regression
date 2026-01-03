# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project uses Qiskit to explore quantum acceleration of p-adic machine learning problems, specifically linear regression in p-adic number systems.

## Development Setup

This project uses `uv` for dependency management. Run scripts with:

```bash
uv run python <script.py>
```

Dependencies are defined in `pyproject.toml` and locked in `uv.lock`.

## Dependencies

- qiskit - IBM's quantum computing framework
- qiskit-aer - Qiskit simulator backend
- matplotlib - For circuit visualization
- pylatexenc - LaTeX encoding for circuit diagrams

## CI/CD

GitHub Actions runs on push/PR to main:
- Python 3.x environment
- Installs dependencies from requirements.txt

## Code Standards (from Sweep AI config)

- All new business logic should have corresponding unit tests
- Refactor large functions to be more modular
- Add docstrings to all functions and file headers
