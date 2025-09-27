#!/bin/bash

echo "=== Starting Render Build Process ==="

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Download Git LFS files (CSV data)
echo "Downloading Git LFS files..."
git lfs pull

echo "=== Build Process Complete ==="