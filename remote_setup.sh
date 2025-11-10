#!/bin/bash
# LLM-Swarm-Brain Remote Server Setup Script
# Usage: bash remote_setup.sh

set -e  # Exit on error

echo "=========================================="
echo "LLM-Swarm-Brain Remote Setup"
echo "=========================================="

# Check if running on Ubuntu
if [ -f /etc/os-release ]; then
    . /etc/os-release
    if [ "$ID" != "ubuntu" ]; then
        echo "Warning: This script is designed for Ubuntu. Detected: $ID"
    fi
fi

# Update system packages
echo ""
echo "[1/7] Updating system packages..."
sudo apt-get update
sudo apt-get install -y python3.9 python3.9-venv python3-pip git curl wget

# Check NVIDIA GPU availability
echo ""
echo "[2/7] Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    echo "✓ NVIDIA GPU detected"
else
    echo "⚠ Warning: nvidia-smi not found. GPU acceleration may not be available."
    echo "  Install NVIDIA drivers and CUDA toolkit if needed."
fi

# Clone repository if not already present
echo ""
echo "[3/7] Setting up repository..."
REPO_DIR="LLM-Swarm-Brain"
if [ ! -d "$REPO_DIR" ]; then
    echo "Cloning repository..."
    git clone https://github.com/zoadrazorro/LLM-Swarm-Brain.git
    cd "$REPO_DIR"
else
    echo "Repository already exists. Updating..."
    cd "$REPO_DIR"
    git pull
fi

# Create virtual environment
echo ""
echo "[4/7] Creating Python virtual environment..."
if [ ! -d "venv" ]; then
    python3.9 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "[5/7] Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "[6/7] Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "[7/7] Installing dependencies..."
pip install -r requirements.txt
pip install -e .

echo ""
echo "=========================================="
echo "✓ Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Activate the environment: source venv/bin/activate"
echo "  2. Run inference test: python inference_test.py"
echo "  3. Or run basic example: python examples/basic_usage.py"
echo ""
echo "To check GPU status: nvidia-smi"
echo ""
