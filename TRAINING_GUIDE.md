# AlphaZero Chess Engine Training

This document explains how to train the AlphaZero chess models using the training entrypoint.

## Quick Start

```bash
# Basic training with default settings
python train.py

# Training with custom parameters
python train.py --iterations 2000 --save-freq 25

# Check system requirements
python train.py --check-requirements

# Start with TensorBoard monitoring
python train.py --tensorboard --port 6007
```

## Training Script Usage

### Basic Commands

```bash
# Show help and all options
python train.py --help

# Start default training (6 models: 3 small + 3 micro)
python train.py

# Quick test run on CPU
python train.py --device cpu --iterations 10 --dry-run

# Production training with monitoring
python train.py --iterations 5000 --tensorboard --verbose
```

### Parameters

#### Training Parameters
- `--iterations, -i`: Maximum training iterations per model (default: 1000)
- `--save-freq, -s`: Save checkpoints every N iterations (default: 50)
- `--device, -d`: Training device - cuda/cpu/auto (default: cuda)

#### Configuration
- `--config, -c`: Directory with configuration files (default: src/experiments/configs)
- `--resume, -r`: Resume from checkpoint directory (not yet implemented)

#### Monitoring
- `--tensorboard, -t`: Start TensorBoard server automatically
- `--port, -p`: TensorBoard port (default: 6006)
- `--log-level`: Logging level - DEBUG/INFO/WARNING/ERROR (default: INFO)

#### System
- `--check-requirements`: Check system requirements and exit
- `--dry-run`: Initialize trainer but don't start training
- `--verbose, -v`: Enable verbose output

## Training Process

### What Happens During Training

1. **Initialization**: 
   - Loads model configurations from JSON files
   - Creates 6 training jobs (3 small tactical/positional/dynamic + 3 micro)
   - Allocates GPU memory across processes
   - Sets up TensorBoard logging

2. **Parallel Training**:
   - Each model trains in its own process
   - Self-play games generate training data
   - Neural networks learn from game outcomes
   - Progress is monitored and logged

3. **Monitoring**:
   - Real-time progress logs in console
   - Detailed metrics in TensorBoard
   - Automatic checkpoint saving
   - Error handling and recovery

### Default Training Configuration

The trainer starts 6 models simultaneously:

**Small Models (256 filters, 10 blocks) with Standard Training:**
- `small_tactical_1`: Focuses on tactical play
- `small_positional_1`: Emphasizes positional understanding  
- `small_dynamic_1`: Adapts playing style dynamically

**Micro Models (128 filters, 6 blocks) with Prototype Training:**
- `micro_tactical_1`: Lightweight tactical model
- `micro_positional_1`: Lightweight positional model
- `micro_dynamic_1`: Lightweight dynamic model

## System Requirements

### Minimum Requirements
- Python 3.8+
- 8GB RAM
- 5GB free disk space
- PyTorch 1.8+
- python-chess library

### Recommended for GPU Training
- NVIDIA GPU with 8GB+ VRAM
- 24GB+ system RAM
- CUDA-compatible PyTorch installation
- 50GB+ free disk space

### Required Python Packages
```bash
pip install torch torchvision
pip install chess
pip install tensorboard  # optional, for monitoring
```

## Monitoring Training

### TensorBoard Dashboard

Start training with TensorBoard:
```bash
python train.py --tensorboard
```

Then open http://localhost:6006 to view:

- **Loss metrics**: Total, policy, and value losses
- **Training speed**: Games per second, iteration times
- **Learning progress**: Learning rate changes, best loss tracking
- **System metrics**: GPU memory usage, parameter counts
- **Model comparison**: Compare different model sizes and styles

### Console Output

The trainer provides real-time console output:
```
🚀 Starting AlphaZero training session
   Device: cuda
   Max iterations: 1000
   Save frequency: 50

📋 Initializing ParallelTrainer...
Started process for small_tactical_1 (PID: 12345)
Started process for small_positional_1 (PID: 12346)
...

=== OVERALL TRAINING PROGRESS ===
Processes: 0 completed, 6 training, 0 failed (total: 6)
  small_tactical_1: Iteration 45, Loss 2.456789, LR 0.00100000
  small_positional_1: Iteration 44, Loss 2.534567, LR 0.00100000
  ...
```

## Checkpoints and Resume

### Automatic Checkpointing
- Models are saved every N iterations (default: 50)
- Checkpoints include model state, optimizer state, and training progress
- Saved to `checkpoints/{job_id}/` directory

### Manual Checkpointing
Training automatically saves on:
- Regular intervals (every `--save-freq` iterations)
- Training completion
- Graceful shutdown (Ctrl+C)

### Resume Training (Coming Soon)
```bash
# Resume from specific checkpoint
python train.py --resume checkpoints/small_tactical_1/

# Resume all models from latest checkpoints
python train.py --resume checkpoints/
```

## Troubleshooting

### Common Issues

**CUDA Out of Memory:**
```bash
# Use smaller models or fewer parallel processes
python train.py --config configs/micro_only
```

**Slow Training:**
```bash
# Check if using CPU instead of GPU
python train.py --check-requirements

# Reduce batch size or MCTS simulations in config files
```

**TensorBoard Not Starting:**
```bash
# Install TensorBoard
pip install tensorboard

# Use different port
python train.py --tensorboard --port 6007
```

### Performance Tips

1. **GPU Memory**: Ensure sufficient VRAM for parallel training
2. **Disk Space**: Training generates large checkpoint files
3. **CPU Cores**: More cores = faster self-play game generation
4. **RAM**: Each process needs ~2-4GB system RAM

## Configuration Files

Training configurations are stored in `src/experiments/configs/`:

- `model_sizes.json`: Model architectures (filters, blocks, descriptions)
- `training_configs.json`: Training parameters (learning rates, batch sizes, etc.)
- `style_configs.json`: Playing style configurations

Edit these files to customize training behavior.

## Getting Help

- Use `python train.py --help` for command-line options
- Use `python train.py --check-requirements` to diagnose issues
- Use `--verbose` flag for detailed error messages
- Check TensorBoard for training progress visualization
