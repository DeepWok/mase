Additional Instructions for Imperial College Students
=============================

For students at Imperial College London taking the Advanced Deep Learning Systems module, or taking MSc/MEng projects with the DeepWok Lab, your setup will depend on the requirements for your project.

* Software stream students: you can run MASE on your local machine using Docker or Conda.
    * If you'd like to run batch jobs, you can use the Conda flow on the `ee-mill1` or `ee-mill3` servers.
    * If you need GPU access for your ADLS project, for example to fine tune Pytorch models, you can use Colab by following the instructions `here <https://github.com/DeepWok/mase/blob/main/docs/labs/lab1.ipynb>`_.

* Hardware stream students: if you're taking the ADLS module, you'll need Verilator. You can either use the Docker method (see instructions in the links) or install Verilator locally in your machine and use the Conda method instead.
    * If you need access to Vivado tools for your MSc/MEng project, you'll need to use the ``beholder1`` server. If you need Verilator as well, use Docker, otherwise Conda is also fine.


SSH into an Imperial server
----------------

.. hint::

    If you're using ``beholder1`` or ``ee-mill1``, first ssh into your required server as follows before following the setup instructions. Make sure you're running the Imperial VPN on your machine.

    .. code-block:: bash

        ssh <username>@<server_name>.ee.ic.ac.uk

VS Code (optional)
----------------

My suggestion to work on the server is to install VS Code on your local machine and develop with it. You can also use your own IDE for the project, such as `emacs` or `vim`. 

1. Download and install `VS Code <https://code.visualstudio.com/>`_.
2. Click the *extension* icon on the left bar (Cntrl + Shift + x), and search for *Remote - SSH* and *Remote Development* packages and install them.
3. Click *Help* at the top and choose *Show All Commands* (Cntrl + Shift + P) and open the console to run a few commands.
  - Type :code:`ssh` in the console and choose *Remote SSH - Add New SSH Host*
  - Then you are asked to enter your ssh command, in our case, use: :code:`ssh username@ee-beholder1.ee.ic.ac.uk`
  - Enter your password and you are in! (You may be asked to choose the platform for the first time, choose *Linux*).
4. Click *File* at the top-left and choose *Open Folder* to add the directory where you do your work to the explorer.
5. To use Terminal, you can click *Terminal* on the top and choose *New Terminal*.

Then you have a basic workspace!