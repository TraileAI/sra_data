#!/bin/bash

echo "=== Starting Render Build Process ==="

# Install Git LFS (needed for runtime LFS pull)
if ! command -v git-lfs &> /dev/null; then
    echo "Installing Git LFS..."
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
    apt-get install git-lfs
fi

# Initialize Git LFS (but don't pull yet - we'll do that at runtime)
echo "Initializing Git LFS..."
git lfs install

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "=== Build Process Complete ==="
echo "Note: LFS files will be downloaded at application startup if needed"