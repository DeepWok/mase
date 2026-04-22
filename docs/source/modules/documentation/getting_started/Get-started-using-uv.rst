Getting Started using uv
========================

Install uv (for the first time)
--------------------------------

If you don't have ``uv`` installed yet, you can install it using the official installer.
For Unix-like systems (Linux, macOS), run:

.. code-block:: bash

    curl -LsSf https://astral.sh/uv/install.sh | sh

For more installation options and details, refer to the `official uv documentation <https://github.com/astral-sh/uv>`_.

Install environment using uv
------------------------------

MASE uses ``uv`` for modern dependency management. The environment state is defined by three core files:

- ``.python-version``: Specifies the Python version used by the project (e.g., ``3.11.9``).
- ``pyproject.toml``: Defines project dependencies, build system, and metadata.
- ``uv.lock``: Locks specific versions of all dependencies to ensure environment consistency across different machines.

1. Clone the MASE repository:

.. code-block:: shell

    git clone git@github.com:DeepWok/mase.git

2. Create your own branch to work on:

.. code-block:: shell

    cd mase
    git checkout -b your_branch_name

3. Install MASE and its dependencies:

.. code-block:: shell

    # uv will automatically read .python-version, pyproject.toml, and uv.lock
    # It creates a synchronized virtual environment (default in .venv directory)
    # Additionally, `uv sync` automatically installs the MASE project in editable mode
    uv sync

How to run code with uv
------------------------

In the ``uv`` workflow, the standard way to execute commands is via ``uv run``.
This ensures the command runs within the correct virtual environment without needing to manually activate it.

1. **Running Python scripts** — use ``uv run python`` instead of just ``python``:

.. code-block:: bash

    uv run python path/to/your_script.py

2. **Running tests**:

.. code-block:: bash

    uv run pytest test/ --ignore=test/passes/graph/transforms/verilog

Test your installation
-----------------------

1. Verify the core IR and graph creation features are working correctly:

.. code-block:: bash

    uv run pytest test/ir/graph/test_create_masegraph.py

2. Verify all Python dependencies are available:

.. code-block:: bash

    uv run python -c "import chop; import transformers; print('Environment ready!')"
