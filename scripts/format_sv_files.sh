#!/bin/bash

# Change directory to the specified folder
#cd /path/to/your/folder

# Find all .sv files recursively in the folder and its subfolders
find . -type f -name "*.sv" | while read -r file; do
    # Check if the file exists and is a regular file
    if [[ -f $file ]]; then
        # Format the file using verible-verilog-format
        verible-verilog-format "$file" > temp_file && mv temp_file "$file"
        echo "Formatted: $file"
    fi
done