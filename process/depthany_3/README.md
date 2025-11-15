# Depth Anything 3 - Webcam Depth Estimation

Real-time depth estimation from webcam using [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3).

## Features

- 🎥 Real-time webcam depth estimation
- 🖼️ Single image depth prediction
- 🎨 Beautiful depth visualization with colormaps
- ⚡ GPU acceleration support
- 📊 FPS counter
- 💾 Video recording capability

## Installation

This project uses [UV](https://github.com/astral-sh/uv) as the package manager.

### Prerequisites

- Python 3.10+
- UV package manager
- Webcam (for real-time demo)

### Install UV

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install Dependencies

```bash
cd /Users/alex/repos/vfp/skill-issue/process/depthany_3
uv sync
```

### Install Depth Anything 3 (Optional - for better results)

**⚠️ macOS ARM Note**: The official package won't compile on M1/M2/M3 due to `xformers`. The placeholder model works for testing. For production, use Linux or Docker.

**On Linux/CUDA**:
```bash
git clone https://github.com/ByteDance-Seed/Depth-Anything-3
cd Depth-Anything-3
pip install -e .
```

## Usage

### Webcam Real-Time Depth Estimation

```bash
# Run with default settings (small model)
uv run depthany-webcam

# Use larger model for better quality
uv run depthany-webcam --model-size large

# Show FPS counter
uv run depthany-webcam --fps-display

# Record video
uv run depthany-webcam --save-video output.mp4

# Use different camera
uv run depthany-webcam --camera-id 1

# Custom resolution
uv run depthany-webcam --width 1280 --height 720
```

### Controls (Webcam Mode)

- `q` - Quit
- `s` - Save current frame
- `r` - Reset depth range

### Single Image Depth Estimation

```bash
# Basic usage
uv run depthany-image path/to/image.jpg --show

# Save output
uv run depthany-image path/to/image.jpg --output result.png

# Use larger model
uv run depthany-image path/to/image.jpg --model-size large --show
```

## Model Sizes

- **small**: Fast, good for real-time on CPU (~0.08B params)
- **base**: Balanced speed and quality (~0.12B params)
- **large**: Best quality, requires GPU for real-time (~0.35B params)
- **giant**: Highest quality, GPU required (~1.15B params)
- **nested**: Best with metric depth estimation (~1.40B params)

## Python API

```python
from depthany_3.model import DepthAnything3
import cv2

# Initialize model
model = DepthAnything3(model_size="small")

# Load and predict
image = cv2.imread("image.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
depth = model.predict(image_rgb)

# depth is a numpy array with shape (H, W)
```

## Requirements

- torch >= 2.0.0
- opencv-python >= 4.8.0
- numpy >= 1.24.0
- matplotlib >= 3.7.0

## Troubleshooting

### No camera detected

- Check camera permissions in system settings
- Try different camera IDs: `--camera-id 1`, `--camera-id 2`, etc.
- List available cameras: `ls /dev/video*` (Linux)

### Slow performance

- Use smaller model: `--model-size small`
- Reduce resolution: `--width 320 --height 240`
- Ensure CUDA is available for GPU acceleration

### Model download issues

The first run will download model weights (~1.4GB for large model, ~4.5GB for giant/nested). If download fails:
- Check internet connection
- Models are cached in `~/.cache/huggingface/hub`

## Credits

Based on [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) by ByteDance-Seed.

**Paper**: "Depth Anything 3: Recovering the visual space from any views" (arXiv:2511.10647)

## License

Apache 2.0

