The following lines should print out your current python version

```bash
conda env list # this provides the conda env names
conda create -n torch python=3.10 -y
conda activate torch # this activates the torch environment
which python3 # check whether you are using the correct python3 from anaconda, instead of the system python
```

Check your path

```bash
pwd #/Users/aaron/Projects/mase-tools/docs/examples
```

```bash
python3 -m pip install --user --upgrade pip
python3 -m pip install -r ../machop/requirements.txt
```

You can see a scripted version of this in `Get-started-on-ee-tarrasque-using-Anacoda-software.md`