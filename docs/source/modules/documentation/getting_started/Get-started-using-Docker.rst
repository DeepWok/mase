Getting Started using Docker
=============================

Docker provides a pre-built environment with all MASE dependencies installed, which is useful for users who want to avoid manual environment setup, or who need a reproducible environment across different machines.

Prerequisites
-------------

1. Install `Docker Desktop <https://docs.docker.com/get-docker/>`_ for your platform (Linux, macOS, or Windows).

2. **For GPU support (Linux only):** Install the `NVIDIA Container Toolkit <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html>`_ so that Docker containers can access your NVIDIA GPU.

   .. note::

       GPU passthrough is only supported on Linux hosts with an NVIDIA GPU. On macOS (including Apple Silicon) and Windows, containers run in CPU-only mode.

Pull the MASE Docker Image
--------------------------

The pre-built MASE image is hosted on Docker Hub:

.. code-block:: shell

    docker pull bingleilou/mase-docker-cpu-triton:latest

Run the Container
-----------------

**CPU only:**

.. code-block:: shell

    docker run -it --rm \
        -v $(pwd):/workspace \
        bingleilou/mase-docker-cpu-triton:latest bash

**With GPU access (Linux + NVIDIA only):**

.. code-block:: shell

    docker run -it --rm --gpus all \
        -v $(pwd):/workspace \
        bingleilou/mase-docker-cpu-triton:latest bash

The ``-v $(pwd):/workspace`` flag mounts your current directory into the container so you can access your local files.

Clone and Run MASE Inside the Container
----------------------------------------

Once inside the container shell:

.. code-block:: shell

    git clone https://github.com/DeepWok/mase.git
    cd mase
    uv run python docs/source/modules/documentation/tutorials/tutorial_1_introduction_to_mase.py

Imperial College Students
--------------------------

If you are an Imperial College student working on the ADLS module and need access to a GPU server, refer to :doc:`Get-started-students` for instructions on connecting to the department servers.
