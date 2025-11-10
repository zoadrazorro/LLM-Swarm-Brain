#!/bin/bash
# Quick setup without sudo (assumes system packages already installed)
set -e

echo "=========================================="
echo "LLM-Swarm-Brain Quick Setup"
echo "=========================================="

cd LLM-Swarm-Brain

# Create virtual environment
echo ""
echo "[1/3] Creating Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate and upgrade pip
echo ""
echo "[2/3] Upgrading pip..."
source venv/bin/activate
pip install --upgrade pip

# Install dependencies
echo ""
echo "[3/3] Installing dependencies..."
pip install -r requirements.txt
pip install -e .

echo ""
echo "=========================================="
echo "✓ Quick Setup Complete!"
echo "=========================================="
