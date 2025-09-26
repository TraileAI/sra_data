#!/bin/bash

echo "=== Starting Render Build Process ==="

# Install Git LFS if not available
if ! command -v git-lfs &> /dev/null; then
    echo "Installing Git LFS..."
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
    apt-get install git-lfs
fi

# Initialize Git LFS
echo "Initializing Git LFS..."
git lfs install

# Pull all LFS files
echo "Pulling Git LFS files..."
git lfs pull

# Verify LFS files were downloaded
echo "Verifying LFS files..."
echo "FMP data directory:"
ls -la fmp_data/ | head -10

echo "Equity quotes directory:"
ls -la fmp_data/equity_quotes/ | head -5

echo "Fundata directory:"
ls -la fundata/data/ | head -5

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "=== Build Process Complete ==="