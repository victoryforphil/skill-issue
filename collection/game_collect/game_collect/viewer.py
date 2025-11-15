#!/usr/bin/env python3
"""
Recording Viewer - View and analyze recorded game data.

Utility for inspecting parquet recordings and extracting frames.
"""

import argparse
import io
from pathlib import Path
from typing import Optional

import pandas as pd
from PIL import Image


def load_recording(parquet_file: Path) -> pd.DataFrame:
    """Load a recording from parquet file."""
    return pd.read_parquet(parquet_file)


def show_info(parquet_file: Path) -> None:
    """Display information about a recording."""
    df = load_recording(parquet_file)
    
    print(f"\n{'='*60}")
    print(f"Recording: {parquet_file.name}")
    print(f"{'='*60}")
    print(f"Frames: {len(df)}")
    print(f"Duration: {df['timestamp'].max():.2f}s")
    print(f"Average FPS: {len(df) / df['timestamp'].max():.2f}")
    
    # Joystick info
    if 'name' in df.columns:
        print(f"Joystick: {df['name'].iloc[0]}")
    
    # Count axes, buttons, hats
    axis_cols = [c for c in df.columns if c.startswith('axis_')]
    button_cols = [c for c in df.columns if c.startswith('button_')]
    hat_x_cols = [c for c in df.columns if c.startswith('hat_') and c.endswith('_x')]
    
    print(f"Axes: {len(axis_cols)}")
    print(f"Buttons: {len(button_cols)}")
    print(f"Hats: {len(hat_x_cols)}")
    
    # File size
    file_size_mb = parquet_file.stat().st_size / (1024 * 1024)
    print(f"File size: {file_size_mb:.2f} MB")
    
    # Sample data
    print(f"\n{'='*60}")
    print("Sample frame (first):")
    print(f"{'='*60}")
    first_frame = df.iloc[0]
    for col in df.columns:
        if col != 'screenshot':
            value = first_frame[col]
            if isinstance(value, float):
                print(f"  {col:20s}: {value:8.4f}")
            else:
                print(f"  {col:20s}: {value}")
    
    print(f"{'='*60}\n")


def extract_frame(parquet_file: Path, frame_number: int, output_file: Optional[Path] = None) -> Image.Image:
    """
    Extract a specific frame as an image.
    
    Args:
        parquet_file: Path to parquet recording
        frame_number: Frame number to extract
        output_file: Optional path to save the image
    
    Returns:
        PIL Image object
    """
    df = load_recording(parquet_file)
    
    if frame_number >= len(df):
        raise ValueError(f"Frame {frame_number} out of range (max: {len(df) - 1})")
    
    frame_data = df.iloc[frame_number]
    screenshot_bytes = frame_data['screenshot']
    
    image = Image.open(io.BytesIO(screenshot_bytes))
    
    if output_file:
        image.save(output_file)
        print(f"Saved frame {frame_number} to {output_file}")
    
    return image


def export_frames(parquet_file: Path, output_dir: Path, interval: int = 1) -> None:
    """
    Export all frames as individual images.
    
    Args:
        parquet_file: Path to parquet recording
        output_dir: Directory to save images
        interval: Save every Nth frame (default: 1 = all frames)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_recording(parquet_file)
    
    session_name = parquet_file.stem
    
    print(f"Exporting frames from {parquet_file.name}...")
    print(f"Total frames: {len(df)}")
    print(f"Interval: every {interval} frame(s)")
    
    exported = 0
    for i in range(0, len(df), interval):
        frame_data = df.iloc[i]
        screenshot_bytes = frame_data['screenshot']
        image = Image.open(io.BytesIO(screenshot_bytes))
        
        output_file = output_dir / f"{session_name}_frame_{i:06d}.png"
        image.save(output_file)
        exported += 1
        
        if (i + 1) % 100 == 0:
            print(f"  Exported {exported} frames...")
    
    print(f"Done! Exported {exported} frames to {output_dir}")


def main():
    """Main entry point for the recording viewer."""
    parser = argparse.ArgumentParser(
        description="View and analyze game recordings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show recording information
  %(prog)s recordings/session.parquet --info
  
  # Extract a specific frame
  %(prog)s recordings/session.parquet --extract-frame 100 -o frame.png
  
  # Export all frames
  %(prog)s recordings/session.parquet --export-frames ./frames
  
  # Export every 10th frame
  %(prog)s recordings/session.parquet --export-frames ./frames --interval 10
        """
    )
    
    parser.add_argument(
        'parquet_file',
        type=Path,
        help='Path to parquet recording file'
    )
    
    parser.add_argument(
        '--info',
        action='store_true',
        help='Show recording information'
    )
    
    parser.add_argument(
        '--extract-frame',
        type=int,
        metavar='N',
        help='Extract frame N as an image'
    )
    
    parser.add_argument(
        '--export-frames',
        type=Path,
        metavar='DIR',
        help='Export all frames to directory'
    )
    
    parser.add_argument(
        '--interval',
        type=int,
        default=1,
        help='Export every Nth frame (default: 1)'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=Path,
        help='Output file for --extract-frame'
    )
    
    args = parser.parse_args()
    
    if not args.parquet_file.exists():
        print(f"ERROR: File not found: {args.parquet_file}")
        return 1
    
    # Show info
    if args.info:
        show_info(args.parquet_file)
    
    # Extract single frame
    if args.extract_frame is not None:
        try:
            extract_frame(args.parquet_file, args.extract_frame, args.output)
        except ValueError as e:
            print(f"ERROR: {e}")
            return 1
    
    # Export frames
    if args.export_frames:
        export_frames(args.parquet_file, args.export_frames, args.interval)
    
    # If no action specified, show info by default
    if not (args.info or args.extract_frame is not None or args.export_frames):
        show_info(args.parquet_file)
    
    return 0


if __name__ == '__main__':
    exit(main())

