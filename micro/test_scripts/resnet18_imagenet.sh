#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=30
#SBATCH --mem-per-cpu=3850
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --account=su114-gpu
#SBATCH --signal=SIGUSR1@90
#SBATCH --output=modify-sw_OPT_2.7b@patched_opt_integer_log.txt

module purge
module load CUDA/11.7.0 GCCcore/11.3.0 GCC/11.3.0 OpenMPI/4.1.4 Python/3.10.4
source /home/c/cz98/venvs/mase-old/bin/activate

MASE_DIR="/home/c/cz98/Projects/mase-tools/"
MACHOP_DIR="$MASE_DIR/software"
PROJECT_DIR="$MASE_DIR/micro"
PROJECT="resnet18_iamgenet"

cd $MACHOP_DIR

echo ------------------------- resnet18 train starts ----------------------------

srun python chop --train --model=resnet18 --pretrained --task=cls --dataset=resnet18 --project-dir=${PROJECT_DIR} --project=${PROJECT} --training-optimizer=adamw --seed=3407 --learning-rate=1e-4 --batch-size 1 --max-epochs 20 --batch-size 128 --cpu 16 --gpu 1 --nodes 1 --strategy "ddp"

echo ------------------------- resnet18 train is done ----------------------------
