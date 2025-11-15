#!/usr/bin/env python3
"""
Example script showing how to load and use game recordings for ML training.

This demonstrates how to:
1. Load parquet recordings
2. Extract images and joystick data
3. Prepare data for training Image -> Joystick models
"""

import io
from pathlib import Path
from typing import Tuple, List

import pandas as pd
import numpy as np
from PIL import Image


def load_dataset(parquet_files: List[Path]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load multiple recording files and prepare for ML training.
    
    Args:
        parquet_files: List of parquet recording files
    
    Returns:
        Tuple of (images, joystick_states)
        - images: numpy array of shape (N, H, W, 3)
        - joystick_states: numpy array of shape (N, num_features)
    """
    all_images = []
    all_joystick_states = []
    
    for parquet_file in parquet_files:
        print(f"Loading {parquet_file.name}...")
        df = pd.read_parquet(parquet_file)
        
        # Extract joystick columns (axes, buttons, hats)
        joystick_cols = [c for c in df.columns 
                        if c.startswith(('axis_', 'button_', 'hat_'))]
        
        for idx, row in df.iterrows():
            # Load image
            img_bytes = row['screenshot']
            img = Image.open(io.BytesIO(img_bytes))
            img_array = np.array(img)
            all_images.append(img_array)
            
            # Extract joystick state
            joystick_state = row[joystick_cols].values
            all_joystick_states.append(joystick_state)
    
    images = np.array(all_images)
    joystick_states = np.array(all_joystick_states)
    
    print(f"\nDataset loaded:")
    print(f"  Images shape: {images.shape}")
    print(f"  Joystick states shape: {joystick_states.shape}")
    
    return images, joystick_states


def prepare_for_training(
    images: np.ndarray,
    joystick_states: np.ndarray,
    train_split: float = 0.8,
    resize_to: Tuple[int, int] = (224, 224),
) -> dict:
    """
    Prepare data for training with normalization and train/val split.
    
    Args:
        images: Array of images (N, H, W, 3)
        joystick_states: Array of joystick states (N, num_features)
        train_split: Fraction of data for training
        resize_to: Target image size (H, W)
    
    Returns:
        Dictionary with train/val splits
    """
    # Resize images if needed
    if images.shape[1:3] != resize_to:
        print(f"Resizing images to {resize_to}...")
        resized_images = []
        for img in images:
            img_pil = Image.fromarray(img)
            img_resized = img_pil.resize((resize_to[1], resize_to[0]))
            resized_images.append(np.array(img_resized))
        images = np.array(resized_images)
    
    # Normalize images to [0, 1]
    images = images.astype(np.float32) / 255.0
    
    # Split data
    n_train = int(len(images) * train_split)
    
    dataset = {
        'X_train': images[:n_train],
        'y_train': joystick_states[:n_train],
        'X_val': images[n_train:],
        'y_val': joystick_states[n_train:],
    }
    
    print(f"\nDataset prepared:")
    print(f"  Training samples: {len(dataset['X_train'])}")
    print(f"  Validation samples: {len(dataset['X_val'])}")
    print(f"  Image shape: {dataset['X_train'].shape[1:]}")
    print(f"  Joystick features: {dataset['y_train'].shape[1]}")
    
    return dataset


def analyze_joystick_distribution(joystick_states: np.ndarray, column_names: List[str]) -> None:
    """Print statistics about joystick usage."""
    print("\nJoystick Statistics:")
    print("=" * 60)
    
    for i, col_name in enumerate(column_names):
        values = joystick_states[:, i]
        
        if col_name.startswith('button_'):
            # Button statistics (binary)
            press_rate = np.mean(values) * 100
            print(f"{col_name:15s}: Pressed {press_rate:5.2f}% of the time")
        else:
            # Axis/hat statistics (continuous)
            print(f"{col_name:15s}: "
                  f"mean={np.mean(values):7.3f}, "
                  f"std={np.std(values):6.3f}, "
                  f"min={np.min(values):7.3f}, "
                  f"max={np.max(values):7.3f}")


def example_usage():
    """Example showing complete workflow."""
    # Example: Load recordings from a directory
    recordings_dir = Path("./recordings")
    
    if not recordings_dir.exists():
        print("No recordings directory found. Record some data first!")
        print("Run: uv run game-collect -o ./recordings")
        return
    
    parquet_files = list(recordings_dir.glob("*.parquet"))
    
    if not parquet_files:
        print("No recordings found. Record some data first!")
        print("Run: uv run game-collect -o ./recordings")
        return
    
    print(f"Found {len(parquet_files)} recording(s)")
    
    # Load first recording as example
    print(f"\n{'='*60}")
    print("Example: Loading single recording")
    print(f"{'='*60}")
    
    df = pd.read_parquet(parquet_files[0])
    
    # Get joystick column names
    joystick_cols = [c for c in df.columns 
                    if c.startswith(('axis_', 'button_', 'hat_'))]
    
    # Load data
    images, joystick_states = load_dataset([parquet_files[0]])
    
    # Analyze joystick usage
    analyze_joystick_distribution(joystick_states, joystick_cols)
    
    # Prepare for training
    print(f"\n{'='*60}")
    print("Preparing for training")
    print(f"{'='*60}")
    
    dataset = prepare_for_training(
        images,
        joystick_states,
        train_split=0.8,
        resize_to=(224, 224),
    )
    
    print("\nDataset ready for training!")
    print("\nNext steps:")
    print("1. Define your model (e.g., CNN -> Joystick prediction)")
    print("2. Train on X_train, y_train")
    print("3. Validate on X_val, y_val")
    print("\nExample model architectures:")
    print("  - ResNet/EfficientNet backbone -> FC layers -> Joystick outputs")
    print("  - MobileNet (for fast inference) -> Joystick outputs")
    print("  - Custom CNN -> LSTM (for temporal modeling)")


if __name__ == '__main__':
    example_usage()

