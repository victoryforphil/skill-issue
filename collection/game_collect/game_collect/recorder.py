#!/usr/bin/env python3
"""
Game Recorder - Capture window screenshots and joystick input for ML training.

This module records synchronized game footage and controller input, saving them
to parquet files for training ML models on Image -> Joystick mappings.
"""

import argparse
import time
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
import io

import mss
import pygame
import numpy as np
import pandas as pd
from PIL import Image


class GameRecorder:
    """Records game window screenshots and joystick input at a specified framerate."""
    
    def __init__(
        self,
        output_dir: Path,
        framerate: int = 30,
        monitor_index: int = 1,
        joystick_index: int = 0,
    ):
        """
        Initialize the game recorder.
        
        Args:
            output_dir: Directory to save recordings
            framerate: Capture framerate in Hz (1-50)
            monitor_index: Monitor to capture (0 = all monitors, 1+ = specific monitor)
            joystick_index: Index of joystick to use
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if not 1 <= framerate <= 50:
            raise ValueError("Framerate must be between 1 and 50 Hz")
        self.framerate = framerate
        self.frame_interval = 1.0 / framerate
        
        self.monitor_index = monitor_index
        self.joystick_index = joystick_index
        
        # Initialize pygame for joystick
        pygame.init()
        pygame.joystick.init()
        
        # Storage for captured data
        self.frames: List[Dict[str, Any]] = []
        self.recording = False
        self.session_name: Optional[str] = None
        
    def list_monitors(self) -> None:
        """List available monitors."""
        with mss.mss() as sct:
            print("\nAvailable monitors:")
            for i, monitor in enumerate(sct.monitors):
                print(f"  {i}: {monitor}")
    
    def list_joysticks(self) -> None:
        """List available joysticks."""
        joystick_count = pygame.joystick.get_count()
        print(f"\nFound {joystick_count} joystick(s):")
        
        for i in range(joystick_count):
            joystick = pygame.joystick.Joystick(i)
            joystick.init()
            print(f"  {i}: {joystick.get_name()}")
            print(f"     Axes: {joystick.get_numaxes()}")
            print(f"     Buttons: {joystick.get_numbuttons()}")
            print(f"     Hats: {joystick.get_numhats()}")
            joystick.quit()
    
    def _capture_screenshot(self, sct: mss.mss) -> bytes:
        """Capture screenshot and return as PNG bytes."""
        monitor = sct.monitors[self.monitor_index]
        screenshot = sct.grab(monitor)
        
        # Convert to PIL Image
        img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
        
        # Convert to PNG bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG', optimize=True)
        return img_bytes.getvalue()
    
    def _capture_joystick_state(self, joystick: pygame.joystick.Joystick) -> Dict[str, Any]:
        """Capture current joystick state."""
        # Process pygame events to update joystick state
        pygame.event.pump()
        
        state = {
            'name': joystick.get_name(),
        }
        
        # Capture axes
        num_axes = joystick.get_numaxes()
        for i in range(num_axes):
            state[f'axis_{i}'] = joystick.get_axis(i)
        
        # Capture buttons
        num_buttons = joystick.get_numbuttons()
        for i in range(num_buttons):
            state[f'button_{i}'] = joystick.get_button(i)
        
        # Capture hat switches
        num_hats = joystick.get_numhats()
        for i in range(num_hats):
            hat = joystick.get_hat(i)
            state[f'hat_{i}_x'] = hat[0]
            state[f'hat_{i}_y'] = hat[1]
        
        return state
    
    def start_recording(self, session_name: Optional[str] = None) -> None:
        """
        Start a recording session.
        
        Args:
            session_name: Optional name for the session (default: timestamp)
        """
        if self.recording:
            print("Already recording!")
            return
        
        # Check joystick availability
        joystick_count = pygame.joystick.get_count()
        if joystick_count == 0:
            print("ERROR: No joysticks found!")
            return
        
        if self.joystick_index >= joystick_count:
            print(f"ERROR: Joystick index {self.joystick_index} not found!")
            print(f"Available joysticks: 0-{joystick_count - 1}")
            return
        
        # Initialize joystick
        joystick = pygame.joystick.Joystick(self.joystick_index)
        joystick.init()
        
        self.session_name = session_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.frames = []
        self.recording = True
        
        print(f"\n{'='*60}")
        print(f"Starting recording session: {self.session_name}")
        print(f"Framerate: {self.framerate} Hz")
        print(f"Monitor: {self.monitor_index}")
        print(f"Joystick: {joystick.get_name()}")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*60}")
        print("\nPress Ctrl+C to stop recording...\n")
        
        frame_count = 0
        start_time = time.time()
        last_frame_time = start_time
        
        try:
            with mss.mss() as sct:
                while self.recording:
                    current_time = time.time()
                    
                    # Check if it's time for the next frame
                    if current_time - last_frame_time >= self.frame_interval:
                        # Capture screenshot
                        screenshot_bytes = self._capture_screenshot(sct)
                        
                        # Capture joystick state
                        joystick_state = self._capture_joystick_state(joystick)
                        
                        # Store frame data
                        frame_data = {
                            'timestamp': current_time - start_time,
                            'frame_number': frame_count,
                            'screenshot': screenshot_bytes,
                            **joystick_state,
                        }
                        self.frames.append(frame_data)
                        
                        frame_count += 1
                        last_frame_time = current_time
                        
                        # Print progress
                        if frame_count % self.framerate == 0:
                            elapsed = current_time - start_time
                            actual_fps = frame_count / elapsed if elapsed > 0 else 0
                            print(f"Frames: {frame_count:6d} | "
                                  f"Time: {elapsed:7.2f}s | "
                                  f"Actual FPS: {actual_fps:5.2f}")
                    
                    # Small sleep to prevent busy waiting
                    time.sleep(0.001)
                    
        except KeyboardInterrupt:
            print("\n\nStopping recording...")
        finally:
            self.recording = False
            joystick.quit()
            
            # Save the recording
            if self.frames:
                self._save_recording()
            else:
                print("No frames captured.")
    
    def _save_recording(self) -> None:
        """Save recorded frames to parquet file."""
        if not self.frames:
            print("No frames to save.")
            return
        
        print(f"\nSaving {len(self.frames)} frames...")
        
        # Create DataFrame
        df = pd.DataFrame(self.frames)
        
        # Save to parquet
        output_file = self.output_dir / f"{self.session_name}.parquet"
        df.to_parquet(
            output_file,
            engine='pyarrow',
            compression='snappy',
            index=False,
        )
        
        # Calculate file size
        file_size_mb = output_file.stat().st_size / (1024 * 1024)
        
        print(f"\n{'='*60}")
        print(f"Recording saved successfully!")
        print(f"File: {output_file}")
        print(f"Size: {file_size_mb:.2f} MB")
        print(f"Frames: {len(self.frames)}")
        print(f"Duration: {self.frames[-1]['timestamp']:.2f}s")
        print(f"{'='*60}\n")


def main():
    """Main entry point for the game recorder."""
    parser = argparse.ArgumentParser(
        description="Record game window screenshots and joystick input for ML training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available monitors and joysticks
  %(prog)s --list-devices
  
  # Record at 30 FPS (default)
  %(prog)s -o ./recordings
  
  # Record at 10 FPS with custom session name
  %(prog)s -o ./recordings -f 10 -n my_game_session
  
  # Use specific monitor and joystick
  %(prog)s -o ./recordings -m 2 -j 1
        """
    )
    
    parser.add_argument(
        '-o', '--output-dir',
        type=Path,
        default=Path('./recordings'),
        help='Output directory for recordings (default: ./recordings)'
    )
    
    parser.add_argument(
        '-f', '--framerate',
        type=int,
        default=30,
        help='Capture framerate in Hz, 1-50 (default: 30)'
    )
    
    parser.add_argument(
        '-m', '--monitor',
        type=int,
        default=1,
        help='Monitor index to capture (default: 1, use --list-devices to see available)'
    )
    
    parser.add_argument(
        '-j', '--joystick',
        type=int,
        default=0,
        help='Joystick index to use (default: 0, use --list-devices to see available)'
    )
    
    parser.add_argument(
        '-n', '--session-name',
        type=str,
        help='Session name (default: timestamp)'
    )
    
    parser.add_argument(
        '--list-devices',
        action='store_true',
        help='List available monitors and joysticks, then exit'
    )
    
    args = parser.parse_args()
    
    # Create recorder
    try:
        recorder = GameRecorder(
            output_dir=args.output_dir,
            framerate=args.framerate,
            monitor_index=args.monitor,
            joystick_index=args.joystick,
        )
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    
    # List devices if requested
    if args.list_devices:
        recorder.list_monitors()
        recorder.list_joysticks()
        sys.exit(0)
    
    # Start recording
    recorder.start_recording(session_name=args.session_name)


if __name__ == '__main__':
    main()

