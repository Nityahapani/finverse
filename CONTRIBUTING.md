# Contributing to finverse

Thank you for your interest in contributing.

## Setup

```bash
git clone https://github.com/yourusername/finverse.git
cd finverse
pip install -e ".[dev]"
```

## Running tests

```bash
pytest tests/ -v
# with coverage:
pytest tests/ --cov=finverse --cov-report=term-missing
```

All tests use synthetic data — no API keys or network required.

## Code style

```bash
black finverse/ tests/
ruff check finverse/ tests/
```

## Adding a new module

1. Create `finverse/<package>/<module>.py`
2. Add to `finverse/<package>/__init__.py`
3. Add tests using fixtures from `tests/conftest.py`
4. Update `CHANGELOG.md` under `[Unreleased]`

## Submitting a PR

- One feature or fix per PR
- All tests must pass
- New features require tests with synthetic data only
- Update CHANGELOG.md

## Reporting bugs

Open an issue with Python version, finverse version, minimal example, and full traceback.
