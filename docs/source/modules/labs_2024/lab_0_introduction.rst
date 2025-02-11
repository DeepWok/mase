
.. image:: ../../imgs/deepwok.png
   :width: 160px
   :height: 160px
   :scale: 100 %
   :alt: logo
   :align: center

Lab 0: Introduction to Mase
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <div align="center">
   <p align="center">
      ELEC70109/EE9-AML3-10/EE9-AO25
      <br />
   Written by
      <a href="https://aaron-zhao123.github.io/">Aaron Zhao </a> and
      <a href="https://www.pedrogimenes.co.uk/">Pedro Gimenes </a>
   </p>
   </div>

General introduction
====================

In this lab, you will learn how to use the basic functionalities of Mase. You will be required to run through some of the `tutorials <https://deepwok.github.io/mase/modules/documentation/tutorials.html>`__ in the documentation, which introduce you to the fundamental aspects of the framework, including:

1. Importing models into the framework and generating a compute graph
2. Understanding the Mase IR and its benefit over other ways of representing Machine Learning workloads
3. Writing and executing Torch FX passes to optimize a model

You will start by generating a MaseGraph for a Bert model. You will then fine-tune this model using a LoRA adapter to achieve high performance on the IMDB sequence classification dataset. In future labs, you will build off this work to explore more advanced features of the MASE framework.

Learning tasks
==============

1. Make sure you have read and understood the installation of the framework, detailed `here <https://deepwok.github.io/mase/modules/documentation/getting_started.html>`__.

2. Go through `"Tutorial 1: Introduction to the Mase IR, MaseGraph and Torch FX passes" <https://github.com/DeepWok/mase/blob/main/docs/source/modules/documentation/tutorials/tutorial_1_introduction_to_mase.ipynb>`__ to understand the basic concepts of the framework.

3. Go through `"Tutorial 2: Insert a LoRA adapter to Finetune Bert for Sequence Classification" <https://github.com/DeepWok/mase/blob/main/docs/source/modules/documentation/tutorials/tutorial_2_lora_finetune.ipynb>`__ to understand how to fine-tune a model using the LoRA adapter.

TroubleShooting
================

You may find that you have to use `Python3.11` but Google Colab only provides `Python3.10`. In this case, you can use the following command to force the kernel ot use `Python3.11`:

.. code-block:: text

   #The code below installs 3.11 (assuming you now have 3.10 in colab) and restarts environment, so you can run your cells.
   import sys #for version checker
   import os #for restart routine

   if '3.11' in sys.version:
      print('You already have 3.11, nothing to install')
   elif '3.10' in sys.version:
      print("Python version is: ", sys.version)

      print("Printing content of /usr/local/lib/python* to see available versions")
      !ls /usr/local/lib/python*

      #install python 3.11 and dev utils
      #you may not need all the dev libraries, but I haven't tested which aren't necessary.
      !sudo apt-get update -y > /dev/null
      !sudo apt-get install python3.11 python3.11-dev python3.11-distutils libpython3.11-dev > /dev/null
      !sudo apt-get install python3.11-venv binfmt-support  > /dev/null #recommended in install logs of the command above

      #change alternatives
      !sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 > /dev/null
      !sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 2 > /dev/null

      # install pip
      !curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11  > /dev/null

      #install colab's dependencies
      !python3 -m pip install ipython==8.10.0 traitlets jupyter psutil matplotlib setuptools ipython_genutils ipykernel jupyter_console notebook prompt_toolkit httplib2 astor  > /dev/null

      #minor cleanup
      !sudo apt autoremove > /dev/null

      #link to the old google package
      !ln -s /usr/local/lib/python3.10/dist-packages/google /usr/local/lib/python3.11/dist-packages/google > /dev/null

      #this is just to verify if 3.11 folder was indeed created
      print("Printing content of /usr/local/lib/python3.11/")
      !ls /usr/local/lib/python3.11/

      #restart environment so you don't have to do it manually
      os.kill(os.getpid(), 9)
   else:
      print("Your out of the box Python is not 3.10, so probably the script will not work, so pls feel free to edit the script to ignore then check and re-run: ", sys.version)
