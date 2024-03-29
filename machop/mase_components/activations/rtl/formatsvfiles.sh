#!/bin/bash

# The directory containing your .sv files
DIRECTORY="./"

# Process all .sv files
for file in $(find "$DIRECTORY" -type f -name "*.sv"); do
    TEMP_FILE="$(mktemp)"

    # Run verible-verilog-format and output to temporary file
    /srcPkgs/verible/bin/verible-verilog-format "$file" > "$TEMP_FILE" 

    # Check if verible-verilog-format was successful
    if [ $? -eq 0 ]; then
        # Overwrite the original file with the temporary file
        mv "$TEMP_FILE" "$file"
    else
        echo "Error formatting file: $file"
        rm "$TEMP_FILE"
    fi
done

echo "Formatting complete."
