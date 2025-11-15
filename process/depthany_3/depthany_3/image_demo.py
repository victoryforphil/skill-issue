"""Single image depth estimation demo."""

import cv2
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

from depthany_3.model import DepthAnything3


def main():
    """Main function for single image depth estimation."""
    parser = argparse.ArgumentParser(description="Single image depth estimation")
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to input image"
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="small",
        choices=["small", "base", "large", "giant", "nested"],
        help="Model size to use (small/base/large/giant/nested)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save output image"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show result in window"
    )
    
    args = parser.parse_args()
    
    # Load image
    print(f"Loading image from {args.image_path}...")
    image = cv2.imread(args.image_path)
    if image is None:
        print(f"Error: Could not load image from {args.image_path}")
        return
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Initialize model
    print(f"Initializing Depth Anything 3 ({args.model_size})...")
    model = DepthAnything3(model_size=args.model_size)
    
    # Predict depth
    print("Predicting depth...")
    depth = model.predict(image_rgb)
    
    # Normalize depth for visualization
    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    
    # Create colorized depth map
    depth_colored = plt.cm.viridis(depth_normalized)[:, :, :3]
    depth_colored = (depth_colored * 255).astype(np.uint8)
    depth_colored_bgr = cv2.cvtColor(depth_colored, cv2.COLOR_RGB2BGR)
    
    # Combine original and depth
    combined = np.hstack([image, depth_colored_bgr])
    
    # Save output
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), combined)
        print(f"Saved result to {args.output}")
    
    # Show result
    if args.show:
        cv2.imshow("Depth Estimation Result", combined)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    print("Done!")


if __name__ == "__main__":
    main()

