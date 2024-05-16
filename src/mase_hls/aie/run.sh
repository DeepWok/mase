#!/bin/bash

M='128'
N='128'
K='512'
tiling_l1_m='32'
tiling_l1_n='32'
tiling_l1_k='32'
tiling_l2_m='64'
tiling_l2_n='64'
tiling_l2_k='128'

python3 mm.py ${M} ${N} ${K} ${tiling_l1_m} ${tiling_l1_n} ${tiling_l1_k} ${tiling_l2_m} ${tiling_l2_n} ${tiling_l2_k}
