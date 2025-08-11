#!/usr/bin/env python3
"""
Training Entrypoint for FedRL Chess Engine

This script provides a command-line interface to start AlphaZero training
with various configurations and options.

Usage:
    python train.py --help                           # Show all options
    python train.py                                  # Start default training
    python train.py --iterations 2000                # Custom iterations
    python train.py --device cpu                     # Force CPU training
    python train.py --config configs/custom          # Custom config directory
    python train.py --tensorboard                    # Start with TensorBoard
    python train.py --resume checkpoints/            # Resume from checkpoint

Author: Francesco Finucci
"""

import argparse
import sys
import os
import subprocess
import signal
import time
from pathlib import Path
from typing import Optional

# Add src to Python path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from src.training.trainer import ParallelTrainer
    import torch
except ImportError as e:
    print(f"Error importing dependencies: {e}")
    print("Make sure you have installed all required packages:")
    print("  pip install torch torchvision")
    print("  pip install tensorboard")
    print("  pip install chess")
    sys.exit(1)


class TrainingEntrypoint:
    """
    Main entrypoint class for managing AlphaZero training sessions.
    
    Handles command-line arguments, configuration validation,
    and coordination with the ParallelTrainer.
    """
    
    def __init__(self):
        self.trainer: Optional[ParallelTrainer] = None
        self.tensorboard_process: Optional[subprocess.Popen] = None
        
    def parse_arguments(self) -> argparse.Namespace:
        """Parse command-line arguments."""
        parser = argparse.ArgumentParser(
            description="Train AlphaZero chess models with parallel processing",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  %(prog)s                                    # Start default training (6 models)
  %(prog)s --iterations 2000 --save-freq 25  # 2000 iterations, save every 25
  %(prog)s --device cpu --iterations 100     # Quick CPU test run
  %(prog)s --tensorboard --port 6007         # Start with TensorBoard on port 6007
  %(prog)s --resume checkpoints/             # Resume from latest checkpoint
  %(prog)s --config src/experiments/custom   # Use custom configuration files
            """
        )
        
        # Training parameters
        training_group = parser.add_argument_group('Training Parameters')
        training_group.add_argument(
            '--iterations', '-i',
            type=int,
            default=1000,
            help='Maximum training iterations per model (default: 1000)'
        )
        training_group.add_argument(
            '--save-freq', '-s',
            type=int,
            default=50,
            help='Save checkpoints every N iterations (default: 50)'
        )
        training_group.add_argument(
            '--device', '-d',
            type=str,
            default='cuda',
            choices=['cuda', 'cpu', 'auto'],
            help='Training device (default: cuda, auto=detect best)'
        )
        
        # Configuration
        config_group = parser.add_argument_group('Configuration')
        config_group.add_argument(
            '--config', '-c',
            type=str,
            default='src/experiments/configs',
            help='Directory containing configuration files (default: src/experiments/configs)'
        )
        config_group.add_argument(
            '--resume', '-r',
            type=str,
            metavar='CHECKPOINT_DIR',
            help='Resume training from checkpoint directory'
        )
        
        # Monitoring and logging
        monitor_group = parser.add_argument_group('Monitoring and Logging')
        monitor_group.add_argument(
            '--tensorboard', '-t',
            action='store_true',
            help='Start TensorBoard server automatically'
        )
        monitor_group.add_argument(
            '--port', '-p',
            type=int,
            default=6006,
            help='TensorBoard port (default: 6006)'
        )
        monitor_group.add_argument(
            '--log-level',
            type=str,
            default='INFO',
            choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
            help='Logging level (default: INFO)'
        )
        
        # System and debugging
        system_group = parser.add_argument_group('System and Debugging')
        system_group.add_argument(
            '--check-requirements',
            action='store_true',
            help='Check system requirements and exit'
        )
        system_group.add_argument(
            '--dry-run',
            action='store_true',
            help='Initialize trainer but do not start training'
        )
        system_group.add_argument(
            '--verbose', '-v',
            action='store_true',
            help='Enable verbose output'
        )
        
        return parser.parse_args()
    
    def check_requirements(self) -> bool:
        """
        Check system requirements for training.
        
        Returns:
            True if all requirements are met, False otherwise
        """
        print("Checking system requirements...")
        
        # Check Python version
        python_version = sys.version_info
        if python_version < (3, 8):
            print(f"❌ Python 3.8+ required, found {python_version.major}.{python_version.minor}")
            return False
        print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        # Check PyTorch
        try:
            import torch
            print(f"✅ PyTorch {torch.__version__}")
            
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print(f"✅ CUDA available: {gpu_count} GPU(s)")
                print(f"   GPU 0: {gpu_name} ({gpu_memory:.1f}GB)")
                
                if gpu_memory < 8.0:
                    print(f"⚠️  Warning: Low GPU memory ({gpu_memory:.1f}GB), consider smaller models")
            else:
                print("⚠️  CUDA not available - training will use CPU (very slow)")
        except ImportError:
            print("❌ PyTorch not installed")
            return False
        
        # Check TensorBoard
        try:
            import tensorboard
            print("✅ TensorBoard available")
        except ImportError:
            print("⚠️  TensorBoard not available (optional)")
        
        # Check chess library
        try:
            import chess
            print("✅ python-chess library")
        except ImportError:
            print("❌ python-chess library not installed")
            return False
        
        # Check configuration files
        config_dir = Path("src/experiments/configs")
        if config_dir.exists():
            print(f"✅ Configuration directory: {config_dir}")
        else:
            print(f"❌ Configuration directory not found: {config_dir}")
            return False
        
        # Check disk space
        import shutil
        free_space = shutil.disk_usage(".").free / (1024**3)
        if free_space < 5.0:
            print(f"⚠️  Warning: Low disk space ({free_space:.1f}GB)")
        else:
            print(f"✅ Available disk space: {free_space:.1f}GB")
        
        print("\n✅ System requirements check completed")
        return True
    
    def determine_device(self, device_arg: str) -> str:
        """Determine the best device to use for training."""
        if device_arg == 'auto':
            if torch.cuda.is_available():
                return 'cuda'
            else:
                print("CUDA not available, falling back to CPU")
                return 'cpu'
        elif device_arg == 'cuda' and not torch.cuda.is_available():
            print("CUDA requested but not available, falling back to CPU")
            return 'cpu'
        else:
            return device_arg
    
    def start_tensorboard(self, port: int, log_dir: str = "logs/tensorboard") -> Optional[subprocess.Popen]:
        """
        Start TensorBoard server.
        
        Args:
            port: Port number for TensorBoard
            log_dir: Directory containing TensorBoard logs
            
        Returns:
            TensorBoard process or None if failed
        """
        try:
            # Ensure log directory exists
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            
            # Start TensorBoard
            cmd = [
                sys.executable, '-m', 'tensorboard.main',
                '--logdir', log_dir,
                '--port', str(port),
                '--reload_interval', '30',
                '--host', '0.0.0.0'
            ]
            
            print(f"Starting TensorBoard on http://localhost:{port}")
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Give TensorBoard time to start
            time.sleep(3)
            
            if process.poll() is None:
                print(f"✅ TensorBoard started successfully on port {port}")
                print(f"   View training progress at: http://localhost:{port}")
                return process
            else:
                print("❌ Failed to start TensorBoard")
                return None
                
        except Exception as e:
            print(f"❌ Error starting TensorBoard: {e}")
            return None
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            print(f"\n⚠️  Received signal {signum}, shutting down gracefully...")
            self.shutdown()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def shutdown(self):
        """Gracefully shutdown all processes."""
        print("Shutting down training session...")
        
        # Stop TensorBoard
        if self.tensorboard_process:
            print("Stopping TensorBoard...")
            self.tensorboard_process.terminate()
            try:
                self.tensorboard_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.tensorboard_process.kill()
        
        # The ParallelTrainer handles its own cleanup in the context manager
        print("Shutdown completed")
    
    def main(self):
        """Main execution function."""
        # Parse arguments
        args = self.parse_arguments()
        
        # Check requirements if requested
        if args.check_requirements:
            self.check_requirements()
            return
        
        # Determine device
        device = self.determine_device(args.device)
        
        # Check requirements
        if not self.check_requirements():
            print("\n❌ System requirements not met. Please install missing dependencies.")
            sys.exit(1)
        
        # Setup signal handlers
        self.setup_signal_handlers()
        
        # Start TensorBoard if requested
        if args.tensorboard:
            self.tensorboard_process = self.start_tensorboard(args.port)
        
        # Validate configuration directory
        config_dir = Path(args.config)
        if not config_dir.exists():
            print(f"❌ Configuration directory not found: {config_dir}")
            sys.exit(1)
        
        print(f"\n🚀 Starting AlphaZero training session")
        print(f"   Device: {device}")
        print(f"   Max iterations: {args.iterations}")
        print(f"   Save frequency: {args.save_freq}")
        print(f"   Config directory: {config_dir}")
        
        if args.resume:
            print(f"   Resume from: {args.resume}")
        
        try:
            # Initialize trainer
            print("\n📋 Initializing ParallelTrainer...")
            self.trainer = ParallelTrainer(
                config_dir=str(config_dir),
                device=device
            )
            
            # Handle resume logic (would need to be implemented in ParallelTrainer)
            if args.resume:
                print(f"⚠️  Resume functionality not yet implemented")
                # TODO: Implement resume functionality
                # self.trainer.resume_from_checkpoint(args.resume)
            
            if args.dry_run:
                print("✅ Dry run completed - trainer initialized successfully")
                return
            
            # Start training
            print("\n🏁 Starting parallel training...")
            self.trainer.start_parallel_training(
                max_iterations=args.iterations,
                save_frequency=args.save_freq
            )
            
            print("\n🎉 Training completed successfully!")
            
        except KeyboardInterrupt:
            print("\n⚠️  Training interrupted by user")
        except Exception as e:
            print(f"\n❌ Training failed: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)
        finally:
            self.shutdown()


def main():
    """Entry point for the training script."""
    entrypoint = TrainingEntrypoint()
    entrypoint.main()


if __name__ == "__main__":
    main()
