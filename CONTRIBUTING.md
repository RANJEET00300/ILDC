# Contributing to ILDC

We love your input! We want to make contributing to this project as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Development Environment Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/RANJEET00300/ILDC.git
   cd ILDC
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install pre-commit hooks:**
   We use `pre-commit` to ensure code formatting and quality before each commit.
   ```bash
   pip install pre-commit
   pre-commit install
   ```

## Pull Request Process

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed Anything, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints and passes the `pre-commit` checks.
6. Issue that pull request!

## Code Style

- We follow standard Python conventions (PEP 8).
- We use `black` for code formatting.
- We use `ruff` for fast linting.
- The `pre-commit` hooks will run these automatically on your staged files.

## Any Questions?

Feel free to open an issue or start a discussion!
