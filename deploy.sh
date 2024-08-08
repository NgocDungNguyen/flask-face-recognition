#!/bin/bash

set -e

# Update and install dependencies
apt-get update
apt-get install -y cmake build-essential libopenblas-dev liblapack-dev

# Ensure pip is up-to-date
pip install --upgrade pip

# Install required Python packages
pip install -r requirements.txt
