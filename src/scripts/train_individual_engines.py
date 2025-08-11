#!/usr/bin/env python3
"""
Individual Engine Training Script

This script allows you to train individual AlphaZero chess engines with specific
configurations and styles without the complexity of the full parallel training system.

Usage:
    python train_individual_engines.py --training-config fast --iterations 200
    python train_individual_engines.py --training-config prototype --iterations 100 --style tactical
    python train_individual_engines.py --model-size small --training-config standard --iterations 50

Author: Francesco Finucci
"""

import argparse
import json
import sys
import torch
import os
import signal
import time
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.training.trainer import ParallelTrainer
from src.config.model_config import get_model_config
import logging
import multiprocessing as mp


def load_model_config(model_size: str) -> dict:
    """Load model configuration from JSON file."""
    config_path = Path("src/experiments/configs/model_sizes.json")
    
    if not config_path.exists():
        raise FileNotFoundError(f"Model config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        configs = json.load(f)
    
    if model_size not in configs:
        available = ", ".join(configs.keys())
        raise ValueError(f"Model size '{model_size}' not found. Available: {available}")
    
    return configs[model_size]


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train individual AlphaZero chess engines",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--training-config", 
        type=str, 
        default="fast",
        choices=["ultrafast", "fast", "prototype", "standard", "thorough"],
        help="Training configuration to use"
    )
    
    parser.add_argument(
        "--model-size",
        type=str,
        default="micro",
        choices=["nano", "micro", "small", "medium", "large", "xl", "standard"],
        help="Model size to use"
    )
    
    parser.add_argument(
        "--style",
        type=str,
        default="positional",
        choices=["tactical", "positional", "dynamic"],
        help="Playing style to train"
    )
    
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Number of training iterations"
    )
    
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Enable TensorBoard logging"
    )
    
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to save model checkpoints"
    )
    
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory for log files"
    )
    
    return parser.parse_args()


def load_training_config(config_name: str) -> dict:
    """Load training configuration from JSON file."""
    config_path = Path("src/experiments/configs/training_configs.json")
    
    if not config_path.exists():
        raise FileNotFoundError(f"Training config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        configs = json.load(f)
    
    if config_name not in configs:
        available = ", ".join(configs.keys())
        raise ValueError(f"Training config '{config_name}' not found. Available: {available}")
    
    return configs[config_name]


def create_job_config(args) -> dict:
    """Create a job configuration for the training worker."""
    training_config = load_training_config(args.training_config)
    model_config = load_model_config(args.model_size)
    
    job_config = {
        "job_id": f"{args.model_size}_{args.style}_1",
        "worker_id": f"{args.model_size}_{args.style}_1",
        "model_size": args.model_size,
        "style": args.style,
        "iterations": args.iterations,
        "checkpoint_dir": args.checkpoint_dir,
        "log_dir": args.log_dir,
        "tensorboard": args.tensorboard,
        
        # Model configuration
        "filters": model_config["filters"],
        "blocks": model_config["blocks"],
        
        # Training configuration
        **training_config
    }
    
    return job_config


class IndividualTrainer:
    """Individual trainer that directly runs a single training worker."""
    
    def __init__(self, job_config: Dict[str, Any]):
        self.job_config = job_config
        self.trainer = None
        self.interrupted = False
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals gracefully."""
        print(f"\n⚠️ Received signal {signum}, shutting down gracefully...")
        self.interrupted = True
    
    def train(self):
        """Start the training process."""
        try:
            # Create a ParallelTrainer instance
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.trainer = ParallelTrainer(
                config_dir="src/experiments/configs",
                device=device,
                clear_logs=False
            )
            
            # Initialize memory requirements for single worker
            if torch.cuda.is_available():
                available_memory = torch.cuda.get_device_properties(0).total_memory
                self.trainer.memory_per_process = available_memory  # Use all available memory for single worker
            else:
                self.trainer.memory_per_process = 0  # CPU training
            
            # Create dummy queues (since we're not using multiprocessing)
            import queue
            progress_queue = queue.Queue()
            checkpoint_queue = queue.Queue()
            error_queue = queue.Queue()
            
            print("🏁 Starting training...")
            print(f"📋 Job config: {self.job_config}")
            
            # Call the training worker implementation directly
            try:
                self.trainer._training_worker_impl(
                    job=self.job_config,
                    max_iterations=self.job_config['iterations'],
                    save_frequency=50,  # Save every 50 iterations
                    memory_offset=0,    # No memory offset needed
                    progress_queue=progress_queue,
                    checkpoint_queue=checkpoint_queue,
                    error_queue=error_queue
                )
                
                # Check if there were any errors
                if not error_queue.empty():
                    error_info = error_queue.get()
                    print(f"❌ Training failed: {error_info['error']}")
                    return
                    
            except Exception as e:
                print(f"❌ Training failed with exception: {e}")
                import traceback
                traceback.print_exc()
                return
            
            if not self.interrupted:
                print("✅ Training completed successfully!")
            
        except KeyboardInterrupt:
            print("\n⚠️ Training interrupted by user")
        except Exception as e:
            print(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            raise


def main():
    """Main training function."""
    args = parse_arguments()
    
    print(f"🚀 Starting individual engine training")
    print(f"� Config: {args.training_config}")
    print(f"🧠 Model: {args.model_size}")
    print(f"🎯 Style: {args.style}")
    print(f"🔄 Iterations: {args.iterations}")
    print(f"📈 TensorBoard: {'Enabled' if args.tensorboard else 'Disabled'}")
    print("-" * 50)
    
    try:
        # Create job configuration
        job_config = create_job_config(args)
        
        # Create and start trainer
        trainer = IndividualTrainer(job_config)
        trainer.train()
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
