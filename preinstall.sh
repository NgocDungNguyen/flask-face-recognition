#!/bin/bash
set -e

apt-get update
apt-get install -y build-essential libopenblas-dev liblapack-dev wget

# Download and install CMake
wget https://github.com/Kitware/CMake/releases/download/v3.25.0/cmake-3.25.0-linux-x86_64.sh
chmod +x cmake-3.25.0-linux-x86_64.sh
./cmake-3.25.0-linux-x86_64.sh --skip-license --prefix=/usr/local

# Install dlib manually
pip install dlib
