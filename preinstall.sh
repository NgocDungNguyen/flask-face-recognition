#!/bin/bash
set -e

apt-get update
apt-get install -y cmake build-essential

# Install dlib manually
pip install dlib
