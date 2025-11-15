"""Quick test script to verify Depth Anything 3 installation."""

import sys


def main():
    print("🎉 Depth Anything 3 - Installation Test")
    print("=" * 50)
    
    # Check imports
    print("\n📦 Checking dependencies...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            print(f"   🚀 CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("   💻 Running on CPU")
    except ImportError as e:
        print(f"❌ PyTorch not found: {e}")
        return 1
    
    try:
        import cv2
        print(f"✅ OpenCV {cv2.__version__}")
    except ImportError as e:
        print(f"❌ OpenCV not found: {e}")
        return 1
    
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy not found: {e}")
        return 1
    
    try:
        import matplotlib
        print(f"✅ Matplotlib {matplotlib.__version__}")
    except ImportError as e:
        print(f"❌ Matplotlib not found: {e}")
        return 1
    
    try:
        from depthany_3.model import DepthAnything3
        print(f"✅ depthany_3 package")
    except ImportError as e:
        print(f"❌ depthany_3 package not found: {e}")
        return 1
    
    print("\n🎥 Checking webcam...")
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✅ Webcam detected")
            cap.release()
        else:
            print("⚠️  No webcam detected (you can still use image mode)")
    except Exception as e:
        print(f"⚠️  Could not check webcam: {e}")
    
    print("\n" + "=" * 50)
    print("✅ Installation verified!")
    print("\n🚀 Quick Start:")
    print("   uv run depthany-webcam              # Webcam depth estimation")
    print("   uv run depthany-webcam --fps-display # With FPS counter")
    print("   uv run depthany-image <image> --show # Process image")
    print("\n📚 See README.md for more information")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
