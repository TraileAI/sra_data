#!/bin/bash

echo "=== Starting Render Build Process ==="

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Skip CSV download during build - files will be uploaded to external storage
echo "CSV files will be downloaded from external storage during runtime"

echo "=== Build Process Complete ==="