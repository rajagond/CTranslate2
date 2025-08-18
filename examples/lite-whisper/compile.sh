#!/bin/bash

# Get the absolute path of the directory containing this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Assume the project root is two levels up from the script directory
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Create and enter build directory
BUILD_DIR="$PROJECT_ROOT/build"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Run cmake with desired options
cmake "$PROJECT_ROOT" -DWITH_MKL=OFF -DWITH_OPENBLAS=ON -DWITH_CUDA=ON -DWITH_CUDNN=ON -DCUDA_ARCH_LIST="Ampere"

# Compile and install
make -j$(nproc)
make install
ldconfig

# Install Python package
PYTHON_DIR="$PROJECT_ROOT/python"
cd "$PYTHON_DIR"
rm -rf build
pip3 install -e . -v
