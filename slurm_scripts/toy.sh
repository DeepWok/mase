#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=3850
#SBATCH --gres=gpu:ampere_a100:3
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --account=su114-gpu
module purge
module load CUDA/11.7.0 GCCcore/11.3.0 GCC/11.3.0 OpenMPI/4.1.4 Python/3.10.4

source "${HOME}/venvs/mase/bin/activate"

WORK_DIR="${HOME}/Projects/mase-tools"
cd "${WORK_DIR}/software"

srun ./chop --train --dataset=cifar10 --model=toy --modify-sw configs/tests/integer.toml --save toy_modified --batch-size 512 --cpu 32 --gpu 3
# srun ./chop --train --dataset=cifar10 --model=toy --modify-sw configs/tests/integer.toml --save toy_modified --batch-size 512 --cpu 32 --gpu 3
