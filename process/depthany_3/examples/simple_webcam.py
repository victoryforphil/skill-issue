"""Simple webcam depth estimation example."""

import cv2
import numpy as np
from depthany_3.model import DepthAnything3
import matplotlib.pyplot as plt


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    """Colorize depth map using viridis colormap."""
    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    depth_colored = plt.cm.viridis(depth_normalized)[:, :, :3]
    depth_colored = (depth_colored * 255).astype(np.uint8)
    depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_RGB2BGR)
    return depth_colored


def main():
    """Run simple webcam depth estimation."""
    print("Initializing model...")
    model = DepthAnything3(model_size="small")
    
    print("Opening webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Predict depth
        depth = model.predict(frame_rgb)
        
        # Colorize
        depth_colored = colorize_depth(depth)
        
        # Resize if needed
        if depth_colored.shape[:2] != frame.shape[:2]:
            depth_colored = cv2.resize(depth_colored, (frame.shape[1], frame.shape[0]))
        
        # Show side by side
        combined = np.hstack([frame, depth_colored])
        cv2.imshow("Webcam Depth Estimation", combined)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

