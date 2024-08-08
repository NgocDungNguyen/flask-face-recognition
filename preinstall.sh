#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e

# Update package lists
apt-get update

# Install dependencies for dlib
apt-get install -y cmake build-essential

echo "CMake and build-essential installed successfully"
