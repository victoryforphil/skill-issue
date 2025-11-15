"""Simple image depth estimation example."""

import cv2
import numpy as np
import sys
from pathlib import Path
from depthany_3.model import DepthAnything3
import matplotlib.pyplot as plt


def main():
    """Process a single image and save depth map."""
    if len(sys.argv) < 2:
        print("Usage: python process_image.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not Path(image_path).exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)
    
    print(f"Loading image: {image_path}")
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not load image: {image_path}")
        sys.exit(1)
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    print("Initializing model...")
    model = DepthAnything3(model_size="small")
    
    print("Predicting depth...")
    depth = model.predict(image_rgb)
    
    # Normalize and colorize
    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    depth_colored = plt.cm.viridis(depth_normalized)[:, :, :3]
    depth_colored = (depth_colored * 255).astype(np.uint8)
    depth_colored_bgr = cv2.cvtColor(depth_colored, cv2.COLOR_RGB2BGR)
    
    # Combine
    combined = np.hstack([image, depth_colored_bgr])
    
    # Save
    output_path = Path(image_path).stem + "_depth.png"
    cv2.imwrite(output_path, combined)
    print(f"Saved result to: {output_path}")
    
    # Display
    cv2.imshow("Depth Estimation", combined)
    print("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

