#!/usr/bin/env python3
"""
Test script to verify installation and dependencies.
Run this after installing to make sure everything is working.
"""

import sys
import importlib


def test_import(module_name: str) -> bool:
    """Test if a module can be imported."""
    try:
        importlib.import_module(module_name)
        print(f"✓ {module_name:20s} - OK")
        return True
    except ImportError as e:
        print(f"✗ {module_name:20s} - FAILED: {e}")
        return False


def main():
    """Test all required dependencies."""
    print("=" * 60)
    print("Testing Game Collect Installation")
    print("=" * 60)
    print()
    
    required_modules = [
        'mss',
        'pygame',
        'PIL',  # pillow
        'pandas',
        'pyarrow',
        'numpy',
    ]
    
    print("Checking required dependencies:")
    print("-" * 60)
    
    results = []
    for module in required_modules:
        results.append(test_import(module))
    
    print()
    print("Checking game_collect package:")
    print("-" * 60)
    
    results.append(test_import('game_collect'))
    results.append(test_import('game_collect.recorder'))
    results.append(test_import('game_collect.viewer'))
    
    print()
    print("=" * 60)
    
    if all(results):
        print("✓ All tests passed! Installation successful.")
        print()
        print("Next steps:")
        print("  1. Connect your game controller")
        print("  2. Run: uv run game-collect --list-devices")
        print("  3. Run: uv run game-collect -o ./recordings")
        print()
        print("See QUICKSTART.md for more information.")
        return 0
    else:
        print("✗ Some tests failed. Please install missing dependencies:")
        print("  uv pip install -e .")
        return 1


if __name__ == '__main__':
    sys.exit(main())

