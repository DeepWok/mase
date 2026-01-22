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
   You can run the test suite while ignoring tests that require heavy hardware dependencies (like Verilator) or platform-specific packages (like `mase-triton`):
   ```bash
   uv run pytest test/ --ignore=test/passes/graph/transforms/verilog --ignore=test/passes/module/transforms/onn/test_optical_transformer.py
   ```

## Test your installation

To verify that the MASE software stack (Chop) is correctly installed without needing heavy hardware dependencies (like Verilator), run a simple smoke test:

1. **Verify software stack (Chop)**:
   This test ensures the core IR and graph creation features are working correctly.
   ```bash
   uv run pytest test/ir/graph/test_create_masegraph.py
   ```

2. **Verify Tutorial dependencies**:
   Try importing the main package to ensure all Python dependencies (like `transformers`, `torch`, etc.) are available:
   ```bash
   uv run python -c "import chop; import transformers; print('Environment ready!')"
   ```
