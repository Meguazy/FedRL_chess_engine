"""
Parallel Trainer for Multi-Model AlphaZero Training - DEDUPLICATED VERSION

This module provides parallel training capabilities for multiple AlphaZero models
with all shared utilities centralized in training_utils.py to eliminate duplication.

Key improvements:
1. All shared logic moved to training_utils.py
2. Consistent use of centralized temperature schedules
3. Unified action sampling and Dirichlet noise
4. Centralized game outcome analysis
5. Proper separation of concerns

Author: Francesco Finucci
"""

import json
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
import queue
import time

from ..core.alphazero_net import AlphaZeroNet

# DEDUPLICATED: Import centralized utilities
from ..training.training_utils import (
    TrainingUtilities,
    TemperatureSchedules,
    TrainingExampleFilters,
    GameOutcomeAnalyzer,
    GameResult
)

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: TensorBoard not available. Install with: pip install tensorboard")


def _global_training_worker(trainer_config: Dict[str, Any], job: Dict[str, Any], 
                           max_iterations: int, save_frequency: int,
                           memory_offset: int, progress_queue: mp.Queue, 
                           checkpoint_queue: mp.Queue, error_queue: mp.Queue) -> None:
    """
    Global worker function for multiprocessing.
    
    This function recreates a trainer instance in the worker process and calls
    the actual training logic. This avoids pickling issues with instance methods.
    """
    try:
        # Create a minimal trainer instance in the worker process
        trainer = ParallelTrainer(
            config_dir=trainer_config['config_dir'],
            device=trainer_config['device'],
            clear_logs=False,  # Don't clear logs in worker processes
            setup_logging=False  # Don't setup parallel trainer logging in workers
        )
        
        # Set additional attributes from the config
        trainer.memory_per_process = trainer_config['memory_per_process']
        trainer.tensorboard_dir = Path(trainer_config['tensorboard_dir'])
        
        # Call the actual worker method
        trainer._training_worker_impl(
            job, max_iterations, save_frequency, memory_offset,
            progress_queue, checkpoint_queue, error_queue
        )
    except Exception as e:
        error_queue.put({'job_id': job.get('job_id', 'unknown'), 'error': str(e)})


class ParallelTrainer:
    """
    Manages parallel training of multiple AlphaZero models - DEDUPLICATED VERSION.
    
    All shared training utilities have been moved to training_utils.py to eliminate
    duplication between this module, MCTS, and self-play modules.
    """
    
    def __init__(self, config_dir: str = "src/experiments/configs", device: str = "cuda", 
                 clear_logs: bool = True, setup_logging: bool = True):
        """
        Initialize the ParallelTrainer for multi-model training.
        """
        self.config_dir = Path(config_dir)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Clear existing logs only if requested (main process only)
        if clear_logs:
            self._clear_existing_logs()
        
        # Setup logging only if requested
        if setup_logging:
            self.logger = self._setup_logging()
            self.logger.info(f"Initializing ParallelTrainer on device: {self.device}")
        else:
            # For worker processes, create a minimal logger that doesn't create files
            self.logger = logging.getLogger(f"ParallelTrainer_Worker_{id(self)}")
            self.logger.setLevel(logging.INFO)
        
        # Load configurations automatically
        self.configs = self._load_experiment_configs()
        self.logger.info(f"Loaded {len(self.configs['model_sizes'])} model sizes and {len(self.configs['training_configs'])} training configs")
        
        # Initialize containers for training components
        self.models: Dict[str, AlphaZeroNet] = {}
        self.optimizers: Dict[str, torch.optim.Optimizer] = {}
        self.schedulers: Dict[str, torch.optim.lr_scheduler.ReduceLROnPlateau] = {}
        self.training_states: Dict[str, Dict[str, Any]] = {}
        
        # Resource tracking
        self.gpu_memory_allocated = 0
        self.max_gpu_memory = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
        
        # Training tracking
        self.training_started = False
        self.checkpoint_dir = Path("checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # TensorBoard setup
        self.tensorboard_dir = Path("logs/tensorboard")
        self.tensorboard_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"TensorBoard logs will be saved to: {self.tensorboard_dir}")
        
        if not TENSORBOARD_AVAILABLE:
            self.logger.warning("TensorBoard not available - install with: pip install tensorboard")
        else:
            self.logger.info("TensorBoard integration enabled")
        
        # Setup default training combination
        self.default_training_jobs = self._create_default_training_jobs()
        self.logger.info(f"Created {len(self.default_training_jobs)} default training jobs")
    
    def _load_experiment_configs(self) -> Dict[str, Any]:
        """Load all experiment configurations from JSON files."""
        configs = {
            'model_sizes': {},
            'training_configs': {},
            'default_combinations': []
        }
        
        # Load model sizes configuration
        model_sizes_path = self.config_dir / "model_sizes.json"
        if not model_sizes_path.exists():
            raise FileNotFoundError(f"Model sizes config not found: {model_sizes_path}")
        
        try:
            with open(model_sizes_path, 'r') as f:
                configs['model_sizes'] = json.load(f)
            self.logger.info(f"Loaded model sizes: {list(configs['model_sizes'].keys())}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in model sizes config: {e}")
        
        # Load training configurations
        training_configs_path = self.config_dir / "training_configs.json"
        if not training_configs_path.exists():
            raise FileNotFoundError(f"Training configs not found: {training_configs_path}")
        
        try:
            with open(training_configs_path, 'r') as f:
                configs['training_configs'] = json.load(f)
            self.logger.info(f"Loaded training configs: {list(configs['training_configs'].keys())}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in training configs: {e}")
        
        # Validate configurations
        self._validate_configs(configs)
        
        # Create default combinations
        configs['default_combinations'] = self._create_default_combinations(configs)
        
        return configs
    
    def _validate_configs(self, configs: Dict[str, Any]) -> None:
        """Validate that configurations have required fields and valid values."""
        # Validate model sizes
        for model_name, model_config in configs['model_sizes'].items():
            required_fields = ['filters', 'blocks', 'description']
            for field in required_fields:
                if field not in model_config:
                    raise ValueError(f"Model {model_name} missing required field: {field}")
            
            # Validate numeric values
            if not isinstance(model_config['filters'], int) or model_config['filters'] <= 0:
                raise ValueError(f"Model {model_name} has invalid filters value: {model_config['filters']}")
            
            if not isinstance(model_config['blocks'], int) or model_config['blocks'] <= 0:
                raise ValueError(f"Model {model_name} has invalid blocks value: {model_config['blocks']}")
        
        # Validate training configs
        for training_name, training_config in configs['training_configs'].items():
            required_fields = ['initial_learning_rate', 'lr_scheduler', 'batch_size', 'mcts_simulations']
            for field in required_fields:
                if field not in training_config:
                    raise ValueError(f"Training config {training_name} missing required field: {field}")
            
            # Validate learning rate
            lr = training_config['initial_learning_rate']
            if not isinstance(lr, (int, float)) or lr <= 0:
                raise ValueError(f"Training config {training_name} has invalid learning rate: {lr}")
            
            # Validate batch size
            batch_size = training_config['batch_size']
            if not isinstance(batch_size, int) or batch_size <= 0:
                raise ValueError(f"Training config {training_name} has invalid batch size: {batch_size}")
            
            # Validate MCTS simulations
            sims = training_config['mcts_simulations']
            if not isinstance(sims, int) or sims <= 0:
                raise ValueError(f"Training config {training_name} has invalid MCTS simulations: {sims}")
        
        self.logger.info("Configuration validation completed successfully")
    
    def _create_default_combinations(self, configs: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create default model+training combinations for parallel training.
        """
        combinations = []
        
        # Check if required model sizes and training configs exist
        required_models = ['micro']
        required_training = ['fast']
        
        for model in required_models:
            if model not in configs['model_sizes']:
                raise ValueError(f"Required model size '{model}' not found in configuration")
        
        for training in required_training:
            if training not in configs['training_configs']:
                raise ValueError(f"Required training config '{training}' not found in configuration")
        
        # Create micro models with fast training  
        styles = ['positional']
        for i, style in enumerate(styles):
            combinations.append({
                'job_id': f'micro_{style}_{i+1}',
                'model_size': 'micro', 
                'training_config': 'fast',
                'style': style,
                'description': f'Micro model with {style} style using fast training'
            })
        
        self.logger.info(f"Created {len(combinations)} default training combinations")
        return combinations
    
    def _create_default_training_jobs(self) -> List[Dict[str, Any]]:
        """Create training job specifications from default combinations."""
        training_jobs = []
        
        for combo in self.configs['default_combinations']:
            # Get model and training configurations
            model_config = self.configs['model_sizes'][combo['model_size']]
            training_config = self.configs['training_configs'][combo['training_config']]
            
            # Create detailed job specification
            job = {
                'job_id': combo['job_id'],
                'model_size': combo['model_size'],
                'training_config': combo['training_config'],
                'style': combo['style'],
                'description': combo['description'],
                
                # Model architecture
                'filters': model_config['filters'],
                'blocks': model_config['blocks'],
                
                # Training parameters
                'initial_learning_rate': training_config['initial_learning_rate'],
                'batch_size': training_config['batch_size'],
                'mcts_simulations': training_config['mcts_simulations'],
                'games_per_iteration': training_config['games_per_iteration'],
                'dirichlet_alpha': training_config['dirichlet_alpha'],
                'temperature_moves': training_config['temperature_moves'],
                
                # Learning rate scheduler
                'lr_scheduler_type': training_config['lr_scheduler'],
                'lr_factor': training_config['lr_factor'],
                'lr_patience': training_config['lr_patience'],
                
                # Status tracking
                'status': 'initialized',
                'epoch': 0,
                'games_played': 0,
                'current_loss': None,
                'best_loss': float('inf'),
                'last_checkpoint': None
            }
            
            training_jobs.append(job)
        
        return training_jobs
    
    def _clear_existing_logs(self) -> None:
        """Clear existing log files to start fresh."""
        log_dir = Path("logs")
        if log_dir.exists():
            # Remove worker logs
            for log_file in log_dir.glob("worker_*.log"):
                try:
                    log_file.unlink()
                except Exception:
                    pass
            
            # Remove parallel trainer logs
            for log_file in log_dir.glob("parallel_trainer_*.log"):
                try:
                    log_file.unlink()
                except Exception:
                    pass
        
        # Clear TensorBoard logs
        tensorboard_dir = Path("logs/tensorboard")
        if tensorboard_dir.exists():
            import shutil
            try:
                shutil.rmtree(tensorboard_dir)
                tensorboard_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the parallel trainer."""
        logger = logging.getLogger(f"ParallelTrainer_{id(self)}")
        logger.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler with immediate flushing
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"parallel_trainer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        class FlushingFileHandler(logging.FileHandler):
            def emit(self, record):
                super().emit(record)
                self.flush()
        
        file_handler = FlushingFileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging initialized - log file: {log_file}")
        return logger
    
    def start_parallel_training(self, max_iterations: int = 1000, save_frequency: int = 50) -> None:
        """
        Start parallel training with deduplicated utilities.
        """
        self.logger.info(f"Starting parallel training for {len(self.default_training_jobs)} models")
        self.logger.info(f"Max iterations: {max_iterations}, Save frequency: {save_frequency}")
        
        # Set multiprocessing start method
        if mp.get_start_method(allow_none=True) != 'spawn':
            mp.set_start_method('spawn', force=True)
        
        # Create inter-process communication queues
        self.progress_queue = mp.Queue()
        self.checkpoint_queue = mp.Queue() 
        self.error_queue = mp.Queue()
        
        # Calculate GPU memory allocation per process
        self._calculate_gpu_memory_allocation()
        
        # Start all training processes
        self.processes = []
        self.process_status = {}
        
        try:
            self.logger.info("Starting parallel training processes...")
            
            for i, job in enumerate(self.default_training_jobs):
                job_id = job['job_id']
                
                # Calculate GPU memory offset for this process
                memory_offset = i * self.memory_per_process
                
                # Create process for this training job
                trainer_config = {
                    'config_dir': str(self.config_dir),
                    'device': str(self.device),
                    'memory_per_process': self.memory_per_process,
                    'tensorboard_dir': str(self.tensorboard_dir)
                }
                
                process = mp.Process(
                    target=_global_training_worker,
                    args=(
                        trainer_config,
                        job,
                        max_iterations,
                        save_frequency,
                        memory_offset,
                        self.progress_queue,
                        self.checkpoint_queue,
                        self.error_queue
                    ),
                    name=f"TrainingWorker-{job_id}"
                )
                
                process.start()
                self.processes.append(process)
                self.process_status[job_id] = {
                    'process': process,
                    'status': 'starting',
                    'iteration': 0,
                    'loss': None,
                    'start_time': datetime.now()
                }
                
                self.logger.info(f"Started process for {job_id} (PID: {process.pid})")
                time.sleep(2)  # Small delay to stagger GPU memory allocation
            
            self.logger.info(f"All {len(self.processes)} training processes started successfully")
            
            # Main coordination loop
            self._coordinate_parallel_training(max_iterations, save_frequency)
            
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user - stopping all processes...")
            self._stop_all_processes()
            
        except Exception as e:
            self.logger.error(f"Parallel training failed: {e}")
            self._stop_all_processes()
            raise
            
        finally:
            self._cleanup_parallel_training()
    
    def _calculate_gpu_memory_allocation(self) -> None:
        """Calculate GPU memory allocation per training process."""
        if not torch.cuda.is_available():
            self.logger.warning("CUDA not available - training will use CPU")
            self.memory_per_process = 0
            return
        
        # Get total GPU memory
        total_memory = torch.cuda.get_device_properties(0).total_memory
        total_memory_gb = total_memory / (1024**3)
        
        # Reserve memory for system and coordination (4GB safety margin)
        available_memory = total_memory - (4 * 1024**3)
        
        # Calculate memory per process
        num_processes = len(self.default_training_jobs)
        self.memory_per_process = available_memory // num_processes
        memory_per_process_gb = self.memory_per_process / (1024**3)
        
        self.logger.info(f"GPU Memory allocation:")
        self.logger.info(f"  Total GPU memory: {total_memory_gb:.2f}GB")
        self.logger.info(f"  Available for training: {available_memory / (1024**3):.2f}GB")
        self.logger.info(f"  Memory per process: {memory_per_process_gb:.2f}GB")
        self.logger.info(f"  Number of processes: {num_processes}")
        
        if memory_per_process_gb < 1.0:
            self.logger.warning(f"Low memory per process ({memory_per_process_gb:.2f}GB) - consider reducing model sizes")
    
    def _training_worker_impl(self, job: Dict[str, Any], max_iterations: int, save_frequency: int,
                             memory_offset: int, progress_queue: mp.Queue, 
                             checkpoint_queue: mp.Queue, error_queue: mp.Queue) -> None:
        """
        Implementation of the training worker logic - DEDUPLICATED VERSION.
        
        Now uses centralized utilities from training_utils.py.
        """
        job_id = job['job_id']
        
        try:
            # Set up process-specific logging
            worker_logger = self._setup_worker_logging(job_id)
            worker_logger.info(f"Training worker started for {job_id}")
            
            # Set up TensorBoard logging for this worker
            tensorboard_writer = None
            if TENSORBOARD_AVAILABLE:
                tensorboard_log_dir = self.tensorboard_dir / job_id
                tensorboard_writer = SummaryWriter(log_dir=str(tensorboard_log_dir))
                worker_logger.info(f"TensorBoard logging to: {tensorboard_log_dir}")
                
                # Log job configuration to TensorBoard
                config_text = f"""
                **Model Configuration:**
                - Model Size: {job['model_size']}
                - Style: {job['style']}
                - Filters: {job['filters']}
                - Blocks: {job['blocks']}
                
                **Training Configuration:**
                - Initial LR: {job['initial_learning_rate']}
                - Batch Size: {job['batch_size']}
                - MCTS Simulations: {job['mcts_simulations']}
                - Games per Iteration: {job['games_per_iteration']}
                - Temperature Moves: {job['temperature_moves']}
                """
                tensorboard_writer.add_text("Configuration", config_text, 0)
            
            # Set GPU memory allocation for this process
            if torch.cuda.is_available():
                torch.cuda.set_per_process_memory_fraction(
                    self.memory_per_process / torch.cuda.get_device_properties(0).total_memory
                )
                torch.cuda.empty_cache()
            
            # Initialize model and training components for this worker
            model, optimizer, scheduler = self._initialize_worker_components(job)
            
            # DEDUPLICATED: Import self-play functionality
            from .self_play import StyleSpecificSelfPlay
            self_play_coordinator = StyleSpecificSelfPlay(device=str(self.device), logger=worker_logger)
            
            worker_logger.info(f"Worker {job_id} initialized - starting training loop")
            
            # Track metrics for TensorBoard
            total_games_played = 0
            best_loss = float('inf')
            games_per_second_history = []
            
            # Independent training loop for this model
            for iteration in range(max_iterations):
                iteration_start = time.time()
                worker_logger.info(f"Starting iteration {iteration + 1}/{max_iterations}")
                
                try:
                    # Generate self-play training data
                    worker_logger.info(f"Starting self-play data generation...")
                    worker_logger.info(f"Target: {job['games_per_iteration']} games with {job['mcts_simulations']} MCTS sims each")
                    games_start = time.time()
                    training_examples = self_play_coordinator.generate_training_examples(
                        model=model,
                        style=job['style'],
                        num_games=job['games_per_iteration'],
                        mcts_simulations=job['mcts_simulations'],
                        dirichlet_alpha=job['dirichlet_alpha'],  # Still passed but handled centrally
                        temperature_moves=job['temperature_moves'],
                        save_board_images=job.get('save_board_images', False),
                        board_images_dir=job.get('board_images_dir', 'board_images'),
                        enable_resignation=job.get('enable_resignation', True),
                        resignation_threshold=job.get('resignation_threshold', -0.9),
                        max_moves=job.get('max_moves', 150)
                    )
                    games_time = time.time() - games_start
                    games_per_second = job['games_per_iteration'] / games_time if games_time > 0 else 0
                    worker_logger.info(f"Self-play completed in {games_time:.2f}s ({games_per_second:.2f} games/sec)")
                    worker_logger.info(f"Generated {len(training_examples)} training examples")
                    games_per_second_history.append(games_per_second)
                    
                    # Check if we have any training examples
                    if len(training_examples) == 0:
                        worker_logger.warning("No training examples generated - all games were draws!")
                        worker_logger.warning("Skipping this iteration and continuing...")
                        continue
                    
                    # Train model on generated data
                    worker_logger.info(f"Starting neural network training on {len(training_examples)} examples...")
                    training_start = time.time()
                    loss_dict = self._worker_train_model_detailed(model, optimizer, training_examples, job, worker_logger)
                    training_time = time.time() - training_start
                    worker_logger.info(f"Neural network training completed in {training_time:.2f}s")
                    
                    total_loss = loss_dict['total_loss']
                    policy_loss = loss_dict['policy_loss']
                    value_loss = loss_dict['value_loss']
                    worker_logger.info(f"Training losses - Total: {total_loss:.6f}, Policy: {policy_loss:.6f}, Value: {value_loss:.6f}")
                    
                    # Check for problematic value loss
                    if value_loss < 1e-8:
                        worker_logger.warning(f"⚠️  Very low value loss ({value_loss:.8f}) - possible double-tanh issue!")
                    
                    # Update learning rate scheduler
                    old_lr = optimizer.param_groups[0]['lr']
                    scheduler.step(total_loss)
                    new_lr = optimizer.param_groups[0]['lr']
                    if old_lr != new_lr:
                        worker_logger.info(f"Learning rate updated: {old_lr:.2e} -> {new_lr:.2e}")
                    
                    # Track best loss
                    if total_loss < best_loss:
                        best_loss = total_loss
                    
                    iteration_time = time.time() - iteration_start
                    total_games_played += job['games_per_iteration']
                    
                    # TensorBoard logging
                    if tensorboard_writer is not None:
                        global_step = iteration + 1
                        
                        # Loss metrics
                        tensorboard_writer.add_scalar('Loss/Total', total_loss, global_step)
                        tensorboard_writer.add_scalar('Loss/Policy', policy_loss, global_step)
                        tensorboard_writer.add_scalar('Loss/Value', value_loss, global_step)
                        tensorboard_writer.add_scalar('Loss/Best', best_loss, global_step)
                        
                        # Learning rate
                        tensorboard_writer.add_scalar('Training/Learning_Rate', new_lr, global_step)
                        
                        # Training speed metrics
                        tensorboard_writer.add_scalar('Performance/Games_Per_Second', games_per_second, global_step)
                        tensorboard_writer.add_scalar('Performance/Iteration_Time_Seconds', iteration_time, global_step)
                        tensorboard_writer.add_scalar('Performance/Training_Time_Seconds', training_time, global_step)
                        tensorboard_writer.add_scalar('Performance/SelfPlay_Time_Seconds', games_time, global_step)
                        tensorboard_writer.add_scalar('Performance/Total_Games_Played', total_games_played, global_step)
                        
                        # Enhanced training data tracking
                        tensorboard_writer.add_scalar('Data/Training_Examples_Count', len(training_examples), global_step)
                        tensorboard_writer.add_scalar('Data/Decisive_Games_Ratio', len(training_examples) / max(job['games_per_iteration'], 1), global_step)
                        
                        # Model architecture info (logged once)
                        if iteration == 0:
                            total_params = sum(p.numel() for p in model.parameters())
                            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                            tensorboard_writer.add_scalar('Model/Total_Parameters', total_params, 0)
                            tensorboard_writer.add_scalar('Model/Trainable_Parameters', trainable_params, 0)
                        
                        # GPU memory usage (if available)
                        if torch.cuda.is_available():
                            gpu_memory_allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
                            gpu_memory_reserved = torch.cuda.memory_reserved() / (1024**3)    # GB
                            tensorboard_writer.add_scalar('System/GPU_Memory_Allocated_GB', gpu_memory_allocated, global_step)
                            tensorboard_writer.add_scalar('System/GPU_Memory_Reserved_GB', gpu_memory_reserved, global_step)
                        
                        # Moving averages
                        if len(games_per_second_history) >= 10:
                            avg_games_per_second = sum(games_per_second_history[-10:]) / 10
                            tensorboard_writer.add_scalar('Performance/Games_Per_Second_10MA', avg_games_per_second, global_step)
                        
                        # LR change detection
                        if old_lr != new_lr:
                            tensorboard_writer.add_scalar('Training/LR_Reduction_Event', 1.0, global_step)
                            worker_logger.info(f"Learning rate reduced: {old_lr:.8f} → {new_lr:.8f}")
                    
                    # Send progress update to main process
                    progress_update = {
                        'job_id': job_id,
                        'iteration': iteration + 1,
                        'loss': total_loss,
                        'policy_loss': policy_loss,
                        'value_loss': value_loss,
                        'learning_rate': new_lr,
                        'iteration_time': iteration_time,
                        'training_examples': len(training_examples),
                        'games_per_second': games_per_second,
                        'total_games_played': total_games_played,
                        'best_loss': best_loss,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    progress_queue.put(progress_update)
                    
                    worker_logger.info(
                        f"Iter {iteration + 1}: Loss={total_loss:.6f} "
                        f"(Policy={policy_loss:.6f}, Value={value_loss:.6f}), "
                        f"LR={new_lr:.8f}, Time={iteration_time:.1f}s, "
                        f"Examples={len(training_examples)}, GPS={games_per_second:.2f}"
                    )
                    
                    # Save checkpoint periodically
                    if (iteration + 1) % save_frequency == 0:
                        checkpoint_path = self._worker_save_checkpoint(
                            job_id, model, optimizer, scheduler, iteration + 1, job
                        )
                        
                        checkpoint_queue.put({
                            'job_id': job_id,
                            'iteration': iteration + 1,
                            'checkpoint_path': str(checkpoint_path),
                            'timestamp': datetime.now().isoformat()
                        })
                        
                        worker_logger.info(f"Saved checkpoint at iteration {iteration + 1}")
                        
                        # Log checkpoint event to TensorBoard
                        if tensorboard_writer is not None:
                            tensorboard_writer.add_scalar('Training/Checkpoint_Saved', 1.0, iteration + 1)
                
                except Exception as e:
                    error_msg = f"Error in training iteration {iteration + 1} for {job_id}: {str(e)}"
                    worker_logger.error(error_msg)
                    import traceback
                    worker_logger.error(f"Full traceback: {traceback.format_exc()}")
                    
                    error_queue.put({
                        'job_id': job_id,
                        'iteration': iteration + 1,
                        'error': error_msg,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    # Log error to TensorBoard
                    if tensorboard_writer is not None:
                        tensorboard_writer.add_scalar('Training/Error_Count', 1.0, iteration + 1)
                    
                    # Continue training despite iteration error
                    continue
            
            # Final checkpoint
            final_checkpoint = self._worker_save_checkpoint(
                job_id, model, optimizer, scheduler, max_iterations, job
            )
            
            worker_logger.info(f"Training completed for {job_id} - final checkpoint: {final_checkpoint}")
            
            # Final TensorBoard logging
            if tensorboard_writer is not None:
                tensorboard_writer.add_scalar('Training/Training_Completed', 1.0, max_iterations)
                
                # Final summary
                final_summary = f"""
                **Training Completed Successfully**
                
                - Total Iterations: {max_iterations}
                - Total Games Played: {total_games_played}
                - Final Loss: {total_loss:.6f}
                - Best Loss: {best_loss:.6f}
                - Final Learning Rate: {new_lr:.8f}
                - Average Games/Second: {sum(games_per_second_history)/len(games_per_second_history):.2f}
                """
                tensorboard_writer.add_text("Training_Summary", final_summary, max_iterations)
                tensorboard_writer.close()
            
            # Send completion notification
            progress_queue.put({
                'job_id': job_id,
                'status': 'completed',
                'final_checkpoint': str(final_checkpoint),
                'total_games_played': total_games_played,
                'best_loss': best_loss,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            error_msg = f"Fatal error in worker {job_id}: {str(e)}"
            error_queue.put({
                'job_id': job_id,
                'error': error_msg,
                'fatal': True,
                'timestamp': datetime.now().isoformat()
            })
            
            # Close TensorBoard writer on fatal error
            if 'tensorboard_writer' in locals() and tensorboard_writer is not None:
                tensorboard_writer.close()
    
    def _coordinate_parallel_training(self, max_iterations: int, save_frequency: int) -> None:
        """Main coordination loop for monitoring parallel training processes."""
        self.logger.info("Starting coordination of parallel training processes")
        
        completed_processes = set()
        last_progress_log = time.time()
        progress_log_interval = 60  # Log overall progress every 60 seconds
        
        try:
            while len(completed_processes) < len(self.processes):
                # Check for progress updates
                while True:
                    try:
                        progress = self.progress_queue.get(timeout=1.0)
                        
                        job_id = progress['job_id']
                        
                        if progress.get('status') == 'completed':
                            completed_processes.add(job_id)
                            self.process_status[job_id]['status'] = 'completed'
                            self.logger.info(f"Training completed for {job_id}")
                        else:
                            # Update process status
                            self.process_status[job_id].update({
                                'status': 'training',
                                'iteration': progress['iteration'],
                                'loss': progress['loss'],
                                'learning_rate': progress['learning_rate'],
                                'last_update': datetime.now()
                            })
                            
                    except queue.Empty:
                        break
                
                # Check for checkpoint notifications
                while True:
                    try:
                        checkpoint_info = self.checkpoint_queue.get(timeout=0.1)
                        job_id = checkpoint_info['job_id']
                        self.logger.info(f"Checkpoint saved for {job_id}: {checkpoint_info['checkpoint_path']}")
                    except queue.Empty:
                        break
                
                # Check for errors
                while True:
                    try:
                        error_info = self.error_queue.get(timeout=0.1)
                        job_id = error_info['job_id']
                        
                        if error_info.get('fatal'):
                            self.logger.error(f"FATAL ERROR in {job_id}: {error_info['error']}")
                            self.process_status[job_id]['status'] = 'failed'
                        else:
                            self.logger.warning(f"Error in {job_id}: {error_info['error']}")
                            
                    except queue.Empty:
                        break
                
                # Log overall progress periodically
                current_time = time.time()
                if current_time - last_progress_log > progress_log_interval:
                    self._log_overall_progress()
                    last_progress_log = current_time
                
                # Check if all processes are still alive
                for job_id, status_info in self.process_status.items():
                    process = status_info['process']
                    if not process.is_alive() and status_info['status'] not in ['completed', 'failed']:
                        self.logger.error(f"Process {job_id} died unexpectedly!")
                        status_info['status'] = 'failed'
                
                time.sleep(5)  # Brief pause before next coordination cycle
                
        except KeyboardInterrupt:
            self.logger.info("Coordination interrupted by user")
            raise
            
        self.logger.info("All training processes completed - coordination finished")
    
    def _log_overall_progress(self) -> None:
        """Log overall progress across all training processes."""
        self.logger.info("=== OVERALL TRAINING PROGRESS ===")
        
        total_processes = len(self.process_status)
        completed = sum(1 for status in self.process_status.values() if status['status'] == 'completed')
        failed = sum(1 for status in self.process_status.values() if status['status'] == 'failed')
        training = total_processes - completed - failed
        
        self.logger.info(f"Processes: {completed} completed, {training} training, {failed} failed (total: {total_processes})")
        
        for job_id, status in self.process_status.items():
            if status['status'] == 'training':
                iteration = status.get('iteration', 0)
                loss = status.get('loss', 'N/A')
                lr = status.get('learning_rate', 'N/A')
                
                if isinstance(loss, float):
                    loss_str = f"{loss:.6f}"
                else:
                    loss_str = str(loss)
                    
                if isinstance(lr, float):
                    lr_str = f"{lr:.8f}"
                else:
                    lr_str = str(lr)
                
                self.logger.info(f"  {job_id}: Iteration {iteration}, Loss {loss_str}, LR {lr_str}")
        
        # GPU memory usage
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / (1024**3)
            memory_cached = torch.cuda.memory_reserved() / (1024**3)
            self.logger.info(f"GPU Memory: {memory_used:.2f}GB used, {memory_cached:.2f}GB cached")
    
    def _stop_all_processes(self) -> None:
        """Stop all training processes gracefully."""
        self.logger.info("Stopping all training processes...")
        
        for job_id, status_info in self.process_status.items():
            process = status_info['process']
            if process.is_alive():
                self.logger.info(f"Terminating process {job_id} (PID: {process.pid})")
                process.terminate()
                
                # Wait for graceful termination
                process.join(timeout=10)
                
                # Force kill if necessary
                if process.is_alive():
                    self.logger.warning(f"Force killing process {job_id}")
                    process.kill()
                    process.join()
        
        self.logger.info("All processes stopped")
    
    def _cleanup_parallel_training(self) -> None:
        """Clean up resources after parallel training."""
        self.logger.info("Cleaning up parallel training resources...")
        
        # Close queues
        try:
            self.progress_queue.close()
            self.checkpoint_queue.close()
            self.error_queue.close()
        except:
            pass
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Final summary
        self._training_summary()
        
        self.logger.info("Parallel training cleanup completed")
    
    def _setup_worker_logging(self, job_id: str) -> logging.Logger:
        """Setup logging for a worker process."""
        logger = logging.getLogger(f"Worker-{job_id}")
        logger.setLevel(logging.INFO)
        
        # Clear any existing handlers to avoid duplicates
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Create custom handler that flushes immediately
        class FlushingFileHandler(logging.FileHandler):
            def emit(self, record):
                super().emit(record)
                self.flush()
        
        # File handler for this worker with immediate flushing
        log_file = Path("logs") / f"worker_{job_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_file.parent.mkdir(exist_ok=True)
        
        file_handler = FlushingFileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Store the log file path in the logger for reference
        logger.log_file_path = log_file
        
        return logger
    
    def _initialize_worker_components(self, job: Dict[str, Any]) -> Tuple[AlphaZeroNet, torch.optim.Optimizer, torch.optim.lr_scheduler.ReduceLROnPlateau]:
        """Initialize model, optimizer, and scheduler for a worker process."""
        
        # Create model
        model = AlphaZeroNet(
            board_size=8,
            action_size=4672,
            num_filters=job['filters'],
            num_res_blocks=job['blocks'],
            input_channels=119
        )
        model.to(self.device)
        
        # Create optimizer
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=job['initial_learning_rate'],
            weight_decay=1e-4
        )
        
        # Create scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=job['lr_factor'],
            patience=job['lr_patience'],
            threshold=1e-4,
            min_lr=1e-8
        )
        
        return model, optimizer, scheduler
    
    def _worker_train_model_detailed(self, model: AlphaZeroNet, optimizer: torch.optim.Optimizer, 
                                   training_examples: List[Any], job: Dict[str, Any], 
                                   worker_logger: logging.Logger) -> Dict[str, float]:
        """
        Train model in worker process with detailed loss breakdown - DEDUPLICATED VERSION.
        
        Now uses centralized training utilities where applicable.
        """
        if not training_examples:
            return {'total_loss': 0.0, 'policy_loss': 0.0, 'value_loss': 0.0}
            
        model.train()
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_batches = 0
        batch_size = job['batch_size']
        
        # Analyze training data before training
        outcomes = [ex.outcome for ex in training_examples]
        outcome_stats = {
            'wins': sum(1 for o in outcomes if o > 0.5),
            'losses': sum(1 for o in outcomes if o < -0.5),
            'draws': sum(1 for o in outcomes if abs(o) <= 0.5),
            'avg_outcome': sum(outcomes) / len(outcomes),
            'outcome_range': (min(outcomes), max(outcomes))
        }
        
        worker_logger.info(f"Training data analysis: {outcome_stats}")
        
        if outcome_stats['draws'] == len(training_examples):
            worker_logger.warning("⚠️  ALL TRAINING EXAMPLES ARE DRAWS - This will prevent value learning!")
        
        # Create batches
        for i in range(0, len(training_examples), batch_size):
            batch = training_examples[i:i + batch_size]
            
            try:
                # Prepare batch tensors
                states = torch.stack([ex.state_tensor for ex in batch]).to(self.device)
                
                # Policy targets
                policy_targets = torch.zeros(len(batch), 4672).to(self.device)
                for j, ex in enumerate(batch):
                    for action, prob in ex.action_probs.items():
                        try:
                            action_idx = self._action_to_index(action)
                            if 0 <= action_idx < 4672:
                                policy_targets[j, action_idx] = prob
                        except:
                            continue
                
                # Value targets
                value_targets = torch.tensor([ex.outcome for ex in batch], 
                                           dtype=torch.float32).to(self.device)
                
                # Forward pass
                policy_logits, values = model(states)
                
                # Apply tanh to values since we removed it from forward()
                values = torch.tanh(values).squeeze()
                
                # Debug: Log value predictions vs targets for first batch
                if num_batches == 0:
                    value_targets_np = value_targets.detach().cpu().numpy()
                    values_np = values.detach().cpu().numpy()
                    unique_targets = set(value_targets_np.round(3))
                    worker_logger.info(f"Batch 0: Value targets range [{value_targets_np.min():.3f}, {value_targets_np.max():.3f}], unique values: {unique_targets}")
                    worker_logger.info(f"Batch 0: Value predictions range [{values_np.min():.3f}, {values_np.max():.3f}]")
                    
                    # Check for tanh squashing issues
                    if abs(values_np.max()) < 0.8 and abs(values_np.min()) < 0.8:
                        worker_logger.warning("⚠️  Value predictions seem squashed - possible double-tanh issue!")
                
                # Compute losses separately
                policy_loss = F.cross_entropy(policy_logits, policy_targets)
                value_loss = F.mse_loss(values, value_targets)
                
                # Weighted loss to emphasize value learning
                value_weight = 3.0  # Increased from 1.0 to emphasize value learning
                total_loss_batch = policy_loss + value_weight * value_loss
                
                # Backward pass
                optimizer.zero_grad()
                total_loss_batch.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Accumulate losses
                total_loss += total_loss_batch.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                num_batches += 1
                
            except Exception as e:
                worker_logger.warning(f"Error processing batch: {e}")
                continue  # Skip problematic batches
        
        if num_batches > 0:
            return {
                'total_loss': total_loss / num_batches,
                'policy_loss': total_policy_loss / num_batches,
                'value_loss': total_value_loss / num_batches
            }
        else:
            return {'total_loss': 0.0, 'policy_loss': 0.0, 'value_loss': 0.0}
    
    def _worker_save_checkpoint(self, job_id: str, model: AlphaZeroNet, 
                               optimizer: torch.optim.Optimizer, 
                               scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
                               iteration: int, job: Dict[str, Any]) -> Path:
        """Save checkpoint in worker process."""
        checkpoint_dir = self.checkpoint_dir / job_id
        checkpoint_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = checkpoint_dir / f"{job_id}_iter_{iteration}_{timestamp}.pth"
        
        torch.save({
            'iteration': iteration,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'job_config': job,
            'timestamp': timestamp
        }, checkpoint_path)
        
        return checkpoint_path
    
    def _action_to_index(self, action) -> int:
        """
        Convert chess move to action index using ChessPosition implementation.
        """
        from ..core.game_utils import ChessPosition
        temp_position = ChessPosition()
        return temp_position.action_to_index(action)
    
    def _training_summary(self) -> None:
        """Print final training summary and statistics."""
        self.logger.info("=== FINAL TRAINING SUMMARY ===")
        
        for job in self.default_training_jobs:
            job_id = job['job_id']
            
            self.logger.info(f"\n{job_id} ({job['style']} style, {job['model_size']} model):")
            self.logger.info(f"  Total epochs: {job['epoch']}")
            self.logger.info(f"  Games played: {job['games_played']}")
            self.logger.info(f"  Final loss: {job['current_loss']:.6f}")
            self.logger.info(f"  Best loss: {job['best_loss']:.6f}")
            
            if job['last_checkpoint']:
                self.logger.info(f"  Last checkpoint: {job['last_checkpoint']}")
        
        # GPU memory summary
        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated() / 1024**3
            max_memory = torch.cuda.max_memory_allocated() / 1024**3
            self.logger.info(f"\nGPU Memory - Final: {final_memory:.2f}GB, Peak: {max_memory:.2f}GB")
        
        self.logger.info("Training session completed.")