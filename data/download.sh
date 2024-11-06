#!/bin/bash

# Check if the file with URLs is provided as an argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <file_with_urls>"
    exit 1
fi

# Create the pdfs directory if it doesn't exist
mkdir -p ./data/manuals_pdf

# Read the file line by line
while IFS= read -r url; do
    # Download the file to the pdfs folder
    echo "Downloading $url..."
    wget -P ./data/manuals_pdf "$url"
done < "$1"

echo "Download completed."
