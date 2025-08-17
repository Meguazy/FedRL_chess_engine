#!/usr/bin/env python3
"""
Test script to verify resignation configuration parameters are properly passed through the system.
"""

import sys
import json
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.scripts.train_individual_engines import load_training_config, create_job_config
from src.training.self_play import StyleSpecificSelfPlay
from src.core.alphazero_net import create_alphazero_chess_net
import argparse


def test_config_parameter_flow():
    """Test that resignation parameters flow correctly from config to implementation."""
    print("🔍 Testing configuration parameter flow...")
    
    # Test each configuration 
    configs_to_test = ['ultrafast', 'fast', 'tactical_optimized', 'thorough']
    
    for config_name in configs_to_test:
        print(f"\n📋 Testing config: {config_name}")
        
        # Load the training config
        training_config = load_training_config(config_name)
        print(f"  Config resignation_threshold: {training_config.get('resignation_threshold', 'NOT FOUND')}")
        print(f"  Config enable_resignation: {training_config.get('enable_resignation', 'NOT FOUND')}")
        print(f"  Config max_moves: {training_config.get('max_moves', 'NOT FOUND')}")
        
        # Create a mock args object
        class MockArgs:
            training_config = config_name
            model_size = 'small'
            style = 'tactical'
            iterations = 1
            checkpoint_dir = 'test_checkpoints'
            log_dir = 'test_logs'
            tensorboard = False
            debug_outcomes = False
            save_board_images = False
            board_images_dir = 'test_images'
        
        # Create job config to see what happens to parameters
        job_config = create_job_config(MockArgs())
        print(f"  Job resignation_threshold: {job_config.get('resignation_threshold', 'NOT FOUND')}")
        print(f"  Job enable_resignation: {job_config.get('enable_resignation', 'NOT FOUND')}")
        print(f"  Job max_moves: {job_config.get('max_moves', 'NOT FOUND')}")
        
        # Verify parameters are preserved
        expected_resignation_threshold = training_config.get('resignation_threshold')
        expected_enable_resignation = training_config.get('enable_resignation')
        expected_max_moves = training_config.get('max_moves')
        
        actual_resignation_threshold = job_config.get('resignation_threshold')
        actual_enable_resignation = job_config.get('enable_resignation')
        actual_max_moves = job_config.get('max_moves')
        
        print(f"  ✅ Resignation threshold preserved: {expected_resignation_threshold == actual_resignation_threshold}")
        print(f"  ✅ Enable resignation preserved: {expected_enable_resignation == actual_enable_resignation}")
        print(f"  ✅ Max moves preserved: {expected_max_moves == actual_max_moves}")


def test_self_play_parameter_usage():
    """Test that StyleSpecificSelfPlay correctly uses the resignation parameters."""
    print("\n🎯 Testing self-play parameter usage...")
    
    # Create a small test model
    model = create_alphazero_chess_net(num_filters=32, num_blocks=2)
    
    # Create self-play generator
    generator = StyleSpecificSelfPlay()
    
    # Test different resignation configurations
    test_configs = [
        {"enable_resignation": True, "resignation_threshold": -0.8, "max_moves": 100},
        {"enable_resignation": False, "resignation_threshold": -0.9, "max_moves": 120},
        {"enable_resignation": True, "resignation_threshold": -0.95, "max_moves": 150},
    ]
    
    for i, config in enumerate(test_configs):
        print(f"\n  Test {i+1}: {config}")
        
        try:
            # This should work without errors and respect the parameters
            training_examples = generator.generate_training_examples(
                model=model,
                style='tactical',
                num_games=1,  # Just one game for testing
                mcts_simulations=25,  # Fast for testing
                temperature_moves=10,
                **config
            )
            
            print(f"    ✅ Successfully generated {len(training_examples)} training examples")
            
        except Exception as e:
            print(f"    ❌ Error: {e}")


if __name__ == "__main__":
    print("🚀 Starting configuration flow test...\n")
    
    try:
        test_config_parameter_flow()
        test_self_play_parameter_usage()
        print("\n🎉 All tests completed!")
        
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        import traceback
        traceback.print_exc()
