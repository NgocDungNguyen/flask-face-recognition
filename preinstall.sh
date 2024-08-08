#!/bin/bash
set -e

apt-get update
apt-get install -y cmake build-essential libopenblas-dev liblapack-dev
pip install cmake

# Install dlib manually
pip install dlib