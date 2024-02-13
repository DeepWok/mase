Getting Started
=============================

To use MASE, you can easily create an environment with all required dependendies using Docker or Anaconda. Follow the instructions on the links below according to your preferred method.

.. hint::

    Some parts of the flow may assume you have a version of Vivado/Vitis installed. Other parts, such as the emit verilog flow, require Verilator, which is included in the Docker container, but not on the Conda environment. If you prefer using Conda, you can just install Verilator locally.

Students
----------------

For students at Imperial College London taking the Advanced Deep Learning Systems module, or taking MSc/MEng projects with the DeepWok Lab, your setup will depend on the requirements for your project.

- Software stream students: you can run MASE on your local machine using Docker or Conda.
    - If you'd like to run batch jobs, you can use the Conda flow on the `ee-mill1` or `ee-mill3` servers.
    - If you need GPU access for your ADLS project, for example to fine tune Pytorch models, you can use Colab by following the instructions `here <https://github.com/DeepWok/mase/blob/main/docs/labs/lab1.ipynb>`_.

- Hardware stream students: if you're taking the ADLS module, you'll need Verilator. You can either use the Docker method (see instructions in the link below) or install Verilator locally in your machine and use the Conda method instead.
    - If you need access to Vivado tools for your MSc/MEng project, you'll need to use the ``beholder1`` server. If you need Verilator as well, use Docker, otherwise Conda is also fine.

.. hint::

    If you're using ``beholder1`` or ``ee-mill1``, first ssh into your required server as follows before following the setup instructions. Make sure you're running the Imperial VPN on your machine.

    .. code-block:: bash

        ssh <username>@<server_name>.ee.ic.ac.uk

Setup Instructions
----------------

.. toctree::
    :maxdepth: 1

    getting_started/Get-started-using-Anaconda
    getting_started/Get-started-using-Docker
    getting_started/Get-started-students