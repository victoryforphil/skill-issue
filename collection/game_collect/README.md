# Game Collector

A Python tool for recording game window screenshots and joystick input for ML training. Captures synchronized video frames and controller input at configurable framerates (1-50 Hz) and saves them to efficient parquet files.

## Features

- 📸 **Window Capture**: Fast screenshot capture using `mss`
- 🎮 **Joystick Input**: Complete joystick state capture (axes, buttons, hats)
- ⚡ **Configurable Framerate**: 1-50 Hz capture rate
- 💾 **Parquet Storage**: Efficient storage with compression
- 🔄 **Synchronized Data**: Screenshots and inputs timestamped together
- 🎯 **ML Ready**: Perfect for training Image → Joystick models

## Installation

Using `uv` (recommended):

```bash
cd collection/game_collect
uv pip install -e .
```

Or with regular pip:

```bash
cd collection/game_collect
pip install -e .
```

## Quick Start

1. **List available devices** (monitors and joysticks):
   ```bash
   uv run game-collect --list-devices
   ```

2. **Start recording** (default: 30 FPS, monitor 1, joystick 0):
   ```bash
   uv run game-collect -o ./recordings
   ```

3. **Press Ctrl+C** to stop recording

## Usage

### Basic Recording

```bash
# Record at 30 FPS to ./recordings directory
uv run game-collect -o ./recordings

# Record at 10 FPS with custom session name
uv run game-collect -o ./recordings -f 10 -n racing_game_01

# Use specific monitor and joystick
uv run game-collect -o ./recordings -m 2 -j 1
```

### Command Line Options

```
-o, --output-dir DIR     Output directory for recordings (default: ./recordings)
-f, --framerate HZ       Capture framerate 1-50 Hz (default: 30)
-m, --monitor INDEX      Monitor index to capture (default: 1)
-j, --joystick INDEX     Joystick index to use (default: 0)
-n, --session-name NAME  Session name (default: timestamp)
--list-devices           List available monitors and joysticks
```

## Data Format

Recordings are saved as parquet files with the following columns:

- `timestamp`: Time in seconds since recording start
- `frame_number`: Sequential frame counter
- `screenshot`: PNG-encoded image bytes
- `name`: Joystick name
- `axis_N`: Joystick axis values (-1.0 to 1.0)
- `button_N`: Button states (0 or 1)
- `hat_N_x`, `hat_N_y`: D-pad/hat values (-1, 0, or 1)

### Loading Data

```python
import pandas as pd
from PIL import Image
import io

# Load recording
df = pd.read_parquet('recordings/20241115_143022.parquet')

# Access first frame
first_frame = df.iloc[0]
timestamp = first_frame['timestamp']
image = Image.open(io.BytesIO(first_frame['screenshot']))
axis_0 = first_frame['axis_0']  # Left stick X
axis_1 = first_frame['axis_1']  # Left stick Y
button_0 = first_frame['button_0']  # A button (varies by controller)
```

## Viewing Recordings

Use the `game-viewer` tool to inspect recordings:

```bash
# Show recording info
uv run game-viewer recordings/session.parquet --info

# Extract a specific frame
uv run game-viewer recordings/session.parquet --extract-frame 100 -o frame.png

# Export all frames to images
uv run game-viewer recordings/session.parquet --export-frames ./frames

# Export every 10th frame (for quick preview)
uv run game-viewer recordings/session.parquet --export-frames ./frames --interval 10
```

## Using Data for ML Training

See the `examples/` directory for complete examples:

```bash
# Run the example training script
uv run python examples/train_example.py
```

The example shows how to:
- Load multiple recording files
- Extract and normalize images
- Prepare joystick data
- Create train/val splits

## Development

Run directly without installing:

```bash
# Run recorder
uv run python -m game_collect.recorder -o ./recordings

# Run viewer
uv run python -m game_collect.viewer recordings/session.parquet --info
```

## System Requirements

- Python 3.12+
- A connected joystick/gamepad
- macOS, Linux, or Windows

## Tips

1. **Find Your Monitor**: Use `--list-devices` to see monitor indices. Monitor 0 is usually all monitors combined, 1+ are individual displays.

2. **Controller Setup**: Make sure your controller is connected before starting. The program will list available controllers with `--list-devices`.

3. **Framerate Selection**: 
   - 10-15 Hz: Good for slow-paced games, smaller files
   - 30 Hz: Balanced for most games
   - 50 Hz: Fast-paced games, larger files

4. **Window Focus**: Make sure the game window is visible on the selected monitor during recording.

## License

See LICENSE file in the repository root.

