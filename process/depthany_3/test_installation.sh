#!/bin/bash
# Test installation script for Depth Anything 3

echo "🧪 Testing Depth Anything 3 Installation"
echo "=========================================="

# Run the test
uv run python hello.py

echo ""
echo "=========================================="
echo "✅ If all checks passed, you're ready to go!"
echo ""
echo "Try these commands:"
echo "  uv run depthany-webcam              # Webcam demo"
echo "  uv run depthany-webcam --fps-display # With FPS"
echo ""

