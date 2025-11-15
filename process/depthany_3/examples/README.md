# Examples

This directory contains simple example scripts demonstrating how to use the Depth Anything 3 package.

## Simple Webcam Example

A minimal webcam depth estimation script:

```bash
uv run python examples/simple_webcam.py
```

Features:
- Minimal code (~50 lines)
- Real-time depth estimation
- Side-by-side RGB and depth display
- Press 'q' to quit

## Simple Image Processing

Process a single image:

```bash
uv run python examples/process_image.py path/to/image.jpg
```

Features:
- Simple image depth estimation
- Saves result as `<filename>_depth.png`
- Displays result in window

## Full-Featured CLI

For more features, use the full CLI tools:

### Webcam with Advanced Features

```bash
# Basic usage
uv run depthany-webcam

# With all features
uv run depthany-webcam --model-size large --fps-display --save-video output.mp4
```

### Image Processing with Advanced Features

```bash
uv run depthany-image image.jpg --show --output result.png --model-size large
```

## Python API

You can also use the package directly:

```python
from depthany_3.model import DepthAnything3
import cv2

# Initialize
model = DepthAnything3(model_size="small")

# Load image
image = cv2.imread("image.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Predict
depth = model.predict(image_rgb)

# depth is numpy array (H, W) with depth values
print(depth.shape, depth.min(), depth.max())
```

## Tips

1. **Model Selection**:
   - `small`: Fast, good for real-time on CPU
   - `base`: Balanced
   - `large`: Best quality, needs GPU for real-time

2. **Performance**:
   - Lower resolution = faster processing
   - GPU acceleration significantly improves speed
   - Small model can run real-time on CPU

3. **Visualization**:
   - Viridis colormap: purple (far) to yellow (near)
   - Can use other matplotlib colormaps (plasma, inferno, etc.)

