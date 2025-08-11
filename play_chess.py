#!/usr/bin/env python3
"""
Chess Engine Launcher

Simple script to launch the graphical chess interface for playing against
your trained AlphaZero models.

Usage:
    python play_chess.py

Requirements:
    pip install pillow  # For GUI support

Author: Francesco Finucci
"""

import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from src.core.chess_engine import ChessEngineGUI, GUI_AVAILABLE
    
    if not GUI_AVAILABLE:
        print("🚨 Missing GUI dependencies!")
        print("\nTo use the graphical chess interface, please install:")
        print("  pip install pillow")
        print("  # cairosvg is optional for advanced features")
        print("\nAlternatively, you can install with uv:")
        print("  uv add pillow")
        sys.exit(1)
    
    print("🚀 Starting AlphaZero Chess Engine...")
    print("\nInstructions:")
    print("1. Click 'Load Model' to select a trained model checkpoint")
    print("2. Choose your color (White/Black)")
    print("3. Adjust AI difficulty (MCTS simulations)")
    print("4. Click 'Start Game' to begin playing!")
    print("5. Click on pieces to select and move them")
    print("\nEnjoy playing against your AI! 🎯")
    
    app = ChessEngineGUI()
    app.run()

except ImportError as e:
    print(f"🚨 Import error: {e}")
    print("\nMake sure you're running from the project root directory and have installed dependencies:")
    print("  uv add pillow")
    sys.exit(1)
except Exception as e:
    print(f"🚨 Error starting chess engine: {e}")
    sys.exit(1)
