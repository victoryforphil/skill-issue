#!/bin/bash
set -e

echo "🚀 Setting up Depth Anything 3 Webcam Project"
echo "=============================================="

# Check if UV is installed
if ! command -v uv &> /dev/null; then
    echo "❌ UV not found. Installing UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo "✅ UV installed successfully"
else
    echo "✅ UV is already installed"
fi

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
uv sync

echo ""
echo "✅ Installation complete!"
echo ""
echo "🎥 Quick Start:"
echo "  uv run depthany-webcam              # Start webcam depth estimation"
echo "  uv run depthany-webcam --fps-display # With FPS counter"
echo "  uv run depthany-image <image.jpg> --show # Process single image"
echo ""
echo "📚 See README.md for more options"

