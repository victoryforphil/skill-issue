"""Real-time webcam depth estimation using Depth Anything 3."""

import cv2
import numpy as np
import argparse
import time
from typing import Optional
import matplotlib.pyplot as plt

from depthany_3.model import DepthAnything3


def colorize_depth(depth: np.ndarray, vmin: Optional[float] = None, vmax: Optional[float] = None) -> np.ndarray:
    """
    Colorize depth map for visualization.
    
    Args:
        depth: Depth map (H, W)
        vmin: Minimum value for normalization
        vmax: Maximum value for normalization
        
    Returns:
        Colorized depth map (H, W, 3) in BGR format
    """
    if vmin is None:
        vmin = depth.min()
    if vmax is None:
        vmax = depth.max()
    
    # Normalize to 0-1
    depth_normalized = (depth - vmin) / (vmax - vmin + 1e-8)
    depth_normalized = np.clip(depth_normalized, 0, 1)
    
    # Apply colormap
    depth_colored = plt.cm.viridis(depth_normalized)[:, :, :3]
    depth_colored = (depth_colored * 255).astype(np.uint8)
    
    # Convert RGB to BGR for OpenCV
    depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_RGB2BGR)
    
    return depth_colored


def main():
    """Main function for webcam depth estimation."""
    parser = argparse.ArgumentParser(description="Real-time webcam depth estimation")
    parser.add_argument(
        "--model-size",
        type=str,
        default="small",
        choices=["small", "base", "large", "giant", "nested"],
        help="Model size to use (small/base/large/giant/nested)"
    )
    parser.add_argument(
        "--camera-id",
        type=int,
        default=0,
        help="Camera device ID"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Camera width"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Camera height"
    )
    parser.add_argument(
        "--fps-display",
        action="store_true",
        help="Display FPS counter"
    )
    parser.add_argument(
        "--save-video",
        type=str,
        default=None,
        help="Path to save output video"
    )
    
    args = parser.parse_args()
    
    print(f"Initializing Depth Anything 3 ({args.model_size})...")
    model = DepthAnything3(model_size=args.model_size)
    
    print(f"Opening camera {args.camera_id}...")
    cap = cv2.VideoCapture(args.camera_id)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera_id}")
        return
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    # Get actual resolution
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {actual_width}x{actual_height}")
    
    # Video writer setup
    video_writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            args.save_video,
            fourcc,
            20.0,
            (actual_width * 2, actual_height)
        )
        print(f"Recording to {args.save_video}")
    
    # FPS calculation
    fps_counter = []
    
    print("\nControls:")
    print("  - Press 'q' to quit")
    print("  - Press 's' to save current frame")
    print("  - Press 'r' to reset depth range")
    print("\nStarting depth estimation...\n")
    
    frame_count = 0
    depth_vmin, depth_vmax = None, None
    
    try:
        while True:
            start_time = time.time()
            
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            # Convert BGR to RGB for model
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Predict depth
            depth = model.predict(frame_rgb)
            
            # Update depth range for consistent visualization
            if depth_vmin is None:
                depth_vmin, depth_vmax = depth.min(), depth.max()
            else:
                # Smooth update
                depth_vmin = 0.95 * depth_vmin + 0.05 * depth.min()
                depth_vmax = 0.95 * depth_vmax + 0.05 * depth.max()
            
            # Colorize depth
            depth_colored = colorize_depth(depth, depth_vmin, depth_vmax)
            
            # Resize depth to match frame size if needed
            if depth_colored.shape[:2] != frame.shape[:2]:
                depth_colored = cv2.resize(depth_colored, (frame.shape[1], frame.shape[0]))
            
            # Combine original and depth side by side
            combined = np.hstack([frame, depth_colored])
            
            # Add FPS counter
            if args.fps_display:
                fps = 1.0 / (time.time() - start_time + 1e-6)
                fps_counter.append(fps)
                if len(fps_counter) > 30:
                    fps_counter.pop(0)
                avg_fps = np.mean(fps_counter)
                cv2.putText(
                    combined,
                    f"FPS: {avg_fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
            
            # Add labels
            cv2.putText(combined, "RGB", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(combined, "Depth", (frame.shape[1] + 10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display
            cv2.imshow("Depth Anything 3 - Webcam", combined)
            
            # Save to video
            if video_writer is not None:
                video_writer.write(combined)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"depth_frame_{frame_count:04d}.png"
                cv2.imwrite(filename, combined)
                print(f"Saved {filename}")
            elif key == ord('r'):
                depth_vmin, depth_vmax = None, None
                print("Reset depth range")
            
            frame_count += 1
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        print("\nCleaning up...")
        cap.release()
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()
        print("Done!")


if __name__ == "__main__":
    main()

