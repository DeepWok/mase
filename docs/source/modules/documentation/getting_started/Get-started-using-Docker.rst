Getting Started using Docker
=============================

Docker provides a pre-built environment with all MASE dependencies installed,
which is useful for users who want to avoid manual environment setup or need
a reproducible environment across different machines.

Prerequisites
-------------

Install `Docker Desktop <https://docs.docker.com/get-docker/>`_ for your platform (Linux, macOS, or Windows).

.. note::

    GPU support requires a Linux host with an NVIDIA GPU and the
    `NVIDIA Container Toolkit <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html>`_
    installed. On macOS and Windows, containers run in CPU-only mode.

Pull the MASE Docker Image
--------------------------

Two official images are available on Docker Hub:

**CPU-only** (recommended for development and tutorials):

.. code-block:: shell

    docker pull docker.io/deepwok/mase-docker-cpu:latest

**GPU** (for GPU-accelerated training and inference, Linux + NVIDIA only):

.. code-block:: shell

    docker pull docker.io/deepwok/mase-docker-gpu:latest

Set Up and Run MASE
--------------------

1. Clone MASE to your local machine:

.. code-block:: shell

    git clone https://github.com/DeepWok/mase.git
    cd mase

2. Start the container, mounting the local ``mase/`` directory into ``/workspace``:

.. code-block:: shell

    # CPU only
    docker run -it --rm \
        -v $(pwd):/workspace \
        docker.io/deepwok/mase-docker-cpu:latest bash

    # With GPU (Linux + NVIDIA only)
    docker run -it --rm --gpus all \
        -v $(pwd):/workspace \
        docker.io/deepwok/mase-docker-gpu:latest bash

.. note::

   **On systems using Podman as a Docker emulator (e.g. RHEL/CentOS 9):**

   SELinux will block access to the mounted volume. Add ``:z`` to the volume mount flag:

   .. code-block:: shell

      docker run -it --rm \
          -v $(pwd):/workspace:z \
          docker.io/deepwok/mase-docker-cpu:latest bash

   The ``:z`` flag relabels the directory so SELinux allows container access.

   The standard ``--gpus all`` flag does not work with Podman. Use the following flags instead:

   .. code-block:: shell

      docker run -it --rm \
          --security-opt=label=disable \
          --device nvidia.com/gpu=all \
          -v $(pwd):/workspace:z \
          docker.io/deepwok/mase-docker-gpu:latest bash

   - ``--device nvidia.com/gpu=all`` replaces ``--gpus all`` for Podman's CDI-based GPU passthrough.
   - ``--security-opt=label=disable`` disables SELinux restrictions on the container.
   Validate GPU access inside the container:

   .. code-block:: shell

       nvidia-smi
       python3 -c "import torch; print(torch.cuda.is_available())"


3. Inside the container, run a tutorial directly (MASE is pre-installed in the image):

.. code-block:: shell

    cd /workspace/mase
    python3 docs/source/modules/documentation/tutorials/tutorial_1_introduction_to_mase.py

.. note::

    If you intend to modify MASE source code and have changes take effect immediately,
    run the following once inside the container:

    .. code-block:: shell

        cd /workspace/mase
        pip3 install -e .


Imperial College Students
--------------------------

If you are an Imperial College student and need access to a GPU server,
refer to :doc:`Get-started-students` for instructions on connecting to
the department servers.
