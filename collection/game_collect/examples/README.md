# Examples

This directory contains example scripts for working with game recordings.

## train_example.py

Demonstrates how to load parquet recordings and prepare them for ML training:

```bash
# Run the example
uv run python examples/train_example.py
```

This script shows:
- Loading parquet files
- Extracting images and joystick data
- Analyzing joystick usage statistics
- Preparing data for training (normalization, resizing, train/val split)

## Custom Usage

You can also use the recordings in your own scripts:

```python
import pandas as pd
from PIL import Image
import io

# Load a recording
df = pd.read_parquet('recordings/my_session.parquet')

# Iterate through frames
for idx, row in df.iterrows():
    timestamp = row['timestamp']
    
    # Get image
    img = Image.open(io.BytesIO(row['screenshot']))
    
    # Get joystick state
    left_stick_x = row['axis_0']
    left_stick_y = row['axis_1']
    button_a = row['button_0']
    
    # Your processing here...
```

