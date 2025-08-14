#!/usr/bin/env python3
"""
ELO Evaluation Launcher

Run from project root directory.
Usage: python evaluate_elo.py --model-path checkpoints/model.pth
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Launch the ELO evaluation with proper Python path setup."""
    # Get the project root directory
    project_root = Path(__file__).parent
    
    # Add src to PYTHONPATH environment variable
    env = os.environ.copy()
    src_path = str(project_root / "src")
    
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = f"{src_path}:{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = src_path
    
    # Build the command to run the evaluation script
    cmd = [
        sys.executable, "-m", "evaluation.elo_eval"
    ] + sys.argv[1:]  # Pass through all command line arguments
    
    # Change to src directory and run
    original_cwd = os.getcwd()
    try:
        os.chdir(project_root / "src")
        result = subprocess.run(cmd, env=env)
        return result.returncode
    finally:
        os.chdir(original_cwd)

if __name__ == "__main__":
    sys.exit(main())

