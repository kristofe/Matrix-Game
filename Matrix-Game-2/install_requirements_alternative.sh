#!/bin/bash

# Alternative installation script for Matrix-Game-2 requirements
# This script tries different TensorRT versions for better compatibility

set -e  # Exit on any error

echo "Installing Matrix-Game-2 requirements (alternative method)..."

# First, install nvidia-pyindex to enable NVIDIA package repository
echo "Installing nvidia-pyindex..."
pip install nvidia-pyindex

# Install regular requirements (excluding tensorrt)
echo "Installing regular requirements..."
pip install -r requirements.txt

# Try different TensorRT versions for better compatibility
echo "Trying to install compatible TensorRT version..."

# First, try the latest version
echo "Attempting to install latest TensorRT..."
if pip install tensorrt 2>/dev/null; then
    echo "Latest TensorRT installed successfully!"
elif pip install tensorrt==10.6.1 2>/dev/null; then
    echo "TensorRT 10.6.1 installed successfully!"
elif pip install tensorrt==10.5.0 2>/dev/null; then
    echo "TensorRT 10.5.0 installed successfully!"
elif pip install tensorrt==10.4.0 2>/dev/null; then
    echo "TensorRT 10.4.0 installed successfully!"
else
    echo "Warning: Could not install any TensorRT version."
    echo "This might be due to system compatibility issues."
    echo "You may need to:"
    echo "1. Update your system libraries"
    echo "2. Use a different Python environment"
    echo "3. Install TensorRT manually from NVIDIA's website"
fi

# Check installation
echo "Checking TensorRT installation..."
if python -c "import tensorrt as trt; print('TensorRT version:', trt.__version__)" 2>/dev/null; then
    echo "TensorRT is working correctly!"
else
    echo "TensorRT has compatibility issues but may still work for basic operations."
fi

echo "Installation completed!"
