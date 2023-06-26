#! /usr/bin/bash

CONDA_ENV=mase-sw
MASE_DIR="/home/cheng/Projects/mase-tools"

cd ${MASE_DIR}/software

echo $(pwd)

# Vision model
# train
project_name="rt_resnet18"

echo ===== --train starts ====
conda run -n ${CONDA_ENV} python chop --debug --train --model resnet18 --task cls --dataset cifar10 --batch-size 2 --pretrained --project ${project_name}
echo ===== --train done =====
# modify-sw
rf -r ../mase_output/${project_name}
echo ===== --modify-sw starts ====
conda run -n ${CONDA_ENV} python chop --debug --modify-sw --modify-sw-config ./configs/modify-sw/integer.toml --model resnet18 --task cls --dataset cifar10 --batch-size 1 --pretrained --project ${project_name}
echo ===== --modify-sw done =====
# load and train quantized model
echo ===== --train modfied starts =====
conda run -n ${CONDA_ENV} python chop --debug --train --model resnet18 --task cls --dataset cifar10 --batch-size 2 --load ../mase_output/${project_name}/software/modify-sw/modified_model.pkl --load-type pkl --project ${project_name}
echo ===== --train modifed done =====
