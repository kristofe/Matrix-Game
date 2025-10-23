#!/bin/bash

# Installation script for Matrix-Game-2 requirements
# This script handles NVIDIA packages that require special installation

set -e  # Exit on any error

echo "Installing Matrix-Game-2 requirements..."

# First, install nvidia-pyindex to enable NVIDIA package repository
echo "Installing nvidia-pyindex..."
pip install nvidia-pyindex

# Install regular requirements (excluding tensorrt)
echo "Installing regular requirements..."
pip install -r requirements.txt

# Install tensorrt directly (not nvidia-tensorrt)
echo "Installing tensorrt..."
pip install tensorrt

# Check for GLIBCXX compatibility issue
echo "Checking TensorRT installation..."
if python -c "import tensorrt as trt; print('TensorRT version:', trt.__version__)" 2>/dev/null; then
    echo "TensorRT installed successfully!"
else
    echo "Warning: TensorRT installation has compatibility issues."
    echo "This is likely due to GLIBCXX version mismatch."
    echo "TensorRT may still work for most use cases, but some features might be limited."
    echo ""
    echo "To fix this issue, you can:"
    echo "1. Update your system's libstdc++ library"
    echo "2. Use a different TensorRT version compatible with your system"
    echo "3. Use TensorRT in a Docker container with compatible libraries"
fi

echo "Installation completed successfully!"
echo ""
echo "Note: If you encounter any issues with tensorrt, you may need to:"
echo "1. Ensure you have CUDA installed on your system"
echo "2. Check that your CUDA version is compatible with tensorrt"
echo "3. Verify installation with: python -c 'import tensorrt as trt; print(trt.__version__)'"
