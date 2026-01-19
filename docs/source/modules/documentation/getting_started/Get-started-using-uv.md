# Getting Started using uv

## Install uv (for the first time)

If you don't have `uv` installed yet, you can install it using the official installer. For Unix-like systems (Linux, macOS), run:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

For more installation options and details, refer to the [official uv documentation](https://github.com/astral-sh/uv).

## Install environment using uv

MASE uses `uv` for modern dependency management. The environment state is defined by three core files:
- `.python-version`: Specifies the Python version used by the project (e.g., `3.11.9`).
- `pyproject.toml`: Defines project dependencies, build system, and metadata.
- `uv.lock`: Locks specific versions of all dependencies to ensure environment consistency across different machines.

1. Clone the MASE repository:
```shell
git clone git@github.com:DeepWok/mase.git
```

2. Create your own branch to work on:
```shell
cd mase
git checkout -b your_branch_name
```

3. Install MASE and its dependencies:
```shell
# uv will automatically read .python-version, pyproject.toml, and uv.lock
# It creates a synchronized virtual environment (default in .venv directory)
# Additionally, `uv sync` automatically installs the MASE project in editable mode
uv sync
```

## How to run code with uv

In the `uv` workflow, the standard way to execute commands is via `uv run`. This ensures the command runs within the correct virtual environment without needing to manually activate it.

1. **Running Python scripts**:
   Use `uv run python` instead of just `python`:
   ```bash
   uv run python path/to/your_script.py
   ```

2. **Running tests**:
   ```bash
   uv run pytest test/
   ```

## Test your installation

1. (Optional but suggested) Verify the installation was successful by importing the software stack:

```bash
uv run python -c "import chop"
```

2. (Optional but suggested) Run the test stack to ensure the codebase is functioning correctly:

```bash
uv run pytest test/
```
