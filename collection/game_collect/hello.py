#!/usr/bin/env python3
"""
Deprecated: Use the game_collect package instead.
This file is kept for backwards compatibility.
"""

import sys

print("=" * 60)
print("This hello.py file is deprecated.")
print("=" * 60)
print("\nPlease use the new game_collect package instead:")
print("\n  # Install dependencies:")
print("  uv pip install -e .")
print("\n  # Record gameplay:")
print("  uv run game-collect -o ./recordings")
print("\n  # View recordings:")
print("  uv run game-viewer recordings/*.parquet --info")
print("\nSee README.md or QUICKSTART.md for more information.")
print("=" * 60)
sys.exit(0)
