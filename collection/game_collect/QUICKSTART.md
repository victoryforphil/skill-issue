# Quick Start Guide

Get up and running with game recording in 5 minutes!

## 1. Install Dependencies

```bash
cd collection/game_collect
uv pip install -e .
```

This will install all required packages:
- `mss` - Fast screen capture
- `pygame` - Joystick input
- `pillow` - Image processing
- `pandas` - Data handling
- `pyarrow` - Parquet file format
- `numpy` - Numerical operations

## 2. Connect Your Controller

Make sure your game controller/joystick is connected to your computer before starting.

## 3. Check Your Setup

```bash
# List available monitors and controllers
uv run game-collect --list-devices
```

You should see output like:

```
Available monitors:
  0: {'left': 0, 'top': 0, 'width': 2560, 'height': 1440}
  1: {'left': 0, 'top': 0, 'width': 2560, 'height': 1440}

Found 1 joystick(s):
  0: Xbox Controller
     Axes: 6
     Buttons: 11
     Hats: 1
```

## 4. Start Recording

```bash
# Basic recording (30 FPS, monitor 1, joystick 0)
uv run game-collect -o ./recordings
```

The program will start capturing:
- Screenshots from your monitor
- All joystick inputs (axes, buttons, d-pad)

Press **Ctrl+C** to stop recording.

## 5. Review Your Recording

```bash
# Show recording information
uv run game-viewer recordings/*.parquet --info
```

Output will show:
- Number of frames captured
- Duration of recording
- File size
- Joystick details
- Sample frame data

## 6. Advanced Options

### Custom Framerate

```bash
# Slower framerate (10 FPS) for smaller files
uv run game-collect -o ./recordings -f 10

# Faster framerate (50 FPS) for fast-paced games
uv run game-collect -o ./recordings -f 50
```

### Specific Monitor

```bash
# Use monitor 2
uv run game-collect -o ./recordings -m 2
```

### Named Sessions

```bash
# Give your recording a meaningful name
uv run game-collect -o ./recordings -n "mario_kart_race_1"
```

### Multiple Controllers

```bash
# Use the second controller
uv run game-collect -o ./recordings -j 1
```

## 7. Using the Data

See `examples/train_example.py` for a complete example of loading and preparing data for ML training.

```bash
uv run python examples/train_example.py
```

## Troubleshooting

### No joysticks found

- Make sure your controller is connected
- Try disconnecting and reconnecting
- On Linux, you may need to install joystick drivers

### Permission errors on Linux

You may need to add your user to the `input` group:

```bash
sudo usermod -a -G input $USER
# Log out and back in
```

### Screen capture not working

- On macOS, grant screen recording permissions in System Preferences → Privacy & Security
- Make sure the game window is visible on the selected monitor

### Import errors

Make sure you've installed the package:

```bash
cd collection/game_collect
uv pip install -e .
```

## Next Steps

1. Record multiple sessions of gameplay
2. Use `game-viewer` to inspect recordings
3. Load data with pandas for ML training
4. Train your Image → Joystick model!

Happy recording! 🎮

