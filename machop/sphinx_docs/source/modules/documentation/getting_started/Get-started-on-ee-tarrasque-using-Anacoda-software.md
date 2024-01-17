# For software project students: Get started on ee tarrasque (using Anacoda)

## Install Anaconda (for the first time)

SSH to the server:
```shell
ssh ${USERNAME}@ee-tarrasque.ee.ic.ac.uk
```

If you have not installed Anaconda before. Download and install Anaconda :
```shell
cd /home/${USERNAME}
wget https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh
bash Anaconda3-2022.10-Linux-x86_64.sh
source /home/${USERNAME}/.bashrc
conda config --set auto_activate_base false
```

## Install environment using Conda

Download Mase:
```shell
cd /home/${USERNAME}
git clone git@github.com:JianyiCheng/mase-tools.git
```

Create your own branch to work on:
```shell
cd /home/${USERNAME}/mase-tools
git checkout -b your_branch_name
```

Install PyTorch and load Conda environments:
```shell
cd /home/${USERNAME}/mase-tools
bash scripts/init-conda.sh
```

