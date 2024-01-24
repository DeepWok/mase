#!/bin/bash

# Define your batch sizes here
BATCH_SIZES=(0.000001 0.00001 0.0001 0.001 0.01)

# Loop over each batch size
for BATCH_SIZE in "${BATCH_SIZES[@]}"
do
    echo "Running model with LR: $BATCH_SIZE"
    
    # Run your command here, replace 'your_command' and '...other-options...' accordingly
   ./ch train jsc-tiny jsc --max-epochs 10 --batch-size 256 --learning-rate $BATCH_SIZE

done