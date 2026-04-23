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

There are two equivalent ways to run code in the ``uv`` environment:

**Option A — Use** ``uv run``

No need to activate the environment manually. Prefix every command with ``uv run``:

.. code-block:: bash

    uv run python path/to/your_script.py
    uv run pytest test/

**Option B — Activate the virtual environment**

Activate the ``.venv`` created by ``uv sync``, then use ``python`` directly as normal:

.. code-block:: bash

    # Linux / macOS
    source .venv/bin/activate

    # Once activated, run commands without the uv run prefix
    python path/to/your_script.py
    pytest test/

    # To deactivate when done
    deactivate

Test your installation
-----------------------

1. Verify the core IR and graph creation features are working correctly:

.. code-block:: bash

    uv run pytest test/ir/graph/test_create_masegraph.py

2. Verify all Python dependencies are available:

.. code-block:: bash

    uv run python -c "import chop; import transformers; print('Environment ready!')"
