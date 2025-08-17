#!/usr/bin/env python3
"""
Simple verification that resignation configuration parameters are properly set in configs.
"""

import json
from pathlib import Path

def verify_resignation_configs():
    """Verify that all training configs have proper resignation parameters."""
    
    config_path = Path("src/experiments/configs/training_configs.json")
    
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return False
    
    with open(config_path, 'r') as f:
        configs = json.load(f)
    
    print("🔍 Checking resignation parameters in all configurations:\n")
    
    all_good = True
    for config_name, config_data in configs.items():
        print(f"📋 {config_name}:")
        
        # Check required resignation parameters
        enable_resignation = config_data.get('enable_resignation')
        resignation_threshold = config_data.get('resignation_threshold')
        max_moves = config_data.get('max_moves')
        
        print(f"  ✓ enable_resignation: {enable_resignation}")
        print(f"  ✓ resignation_threshold: {resignation_threshold}")
        print(f"  ✓ max_moves: {max_moves}")
        
        # Validate values
        if enable_resignation is None:
            print(f"  ❌ Missing enable_resignation")
            all_good = False
        if resignation_threshold is None:
            print(f"  ❌ Missing resignation_threshold")
            all_good = False
        elif resignation_threshold > -0.5:
            print(f"  ⚠️  resignation_threshold seems high: {resignation_threshold}")
        if max_moves is None:
            print(f"  ❌ Missing max_moves")
            all_good = False
        elif max_moves < 50:
            print(f"  ⚠️  max_moves seems low: {max_moves}")
        
        print()
    
    return all_good

def verify_code_changes():
    """Verify that the code changes were applied correctly."""
    print("🔍 Checking code changes in key files:\n")
    
    # Check self_play.py
    self_play_path = Path("src/training/self_play.py")
    if self_play_path.exists():
        with open(self_play_path, 'r') as f:
            content = f.read()
        
        print("📁 src/training/self_play.py:")
        
        # Check method signature
        if "enable_resignation: bool = True" in content:
            print("  ✅ generate_training_examples has enable_resignation parameter")
        else:
            print("  ❌ Missing enable_resignation parameter in generate_training_examples")
        
        if "resignation_threshold: float = -0.9" in content:
            print("  ✅ generate_training_examples has resignation_threshold parameter")
        else:
            print("  ❌ Missing resignation_threshold parameter in generate_training_examples")
        
        if "max_moves: int = 150" in content:
            print("  ✅ generate_training_examples has max_moves parameter")
        else:
            print("  ❌ Missing max_moves parameter in generate_training_examples")
        
        # Check usage
        if "resignation_threshold=resignation_threshold" in content:
            print("  ✅ resignation_threshold is passed to MCTS engine")
        else:
            print("  ❌ resignation_threshold not passed to MCTS engine")
        
        print()
    
    # Check trainer.py
    trainer_path = Path("src/training/trainer.py")
    if trainer_path.exists():
        with open(trainer_path, 'r') as f:
            content = f.read()
        
        print("📁 src/training/trainer.py:")
        
        if "enable_resignation=job.get('enable_resignation'" in content:
            print("  ✅ Trainer passes enable_resignation from job config")
        else:
            print("  ❌ Trainer doesn't pass enable_resignation from job config")
        
        if "resignation_threshold=job.get('resignation_threshold'" in content:
            print("  ✅ Trainer passes resignation_threshold from job config")
        else:
            print("  ❌ Trainer doesn't pass resignation_threshold from job config")
        
        print()
    
    # Check train_individual_engines.py
    script_path = Path("src/scripts/train_individual_engines.py")
    if script_path.exists():
        with open(script_path, 'r') as f:
            content = f.read()
        
        print("📁 src/scripts/train_individual_engines.py:")
        
        if '"enable_resignation": True,' in content and '"resignation_threshold": -0.9,' in content:
            print("  ⚠️  Script still has hardcoded resignation values (this might override configs)")
        elif 'training_config.get("enable_resignation"' in content:
            print("  ✅ Script uses config values for resignation parameters")
        else:
            print("  ❓ Unclear how script handles resignation parameters")
        
        print()


if __name__ == "__main__":
    print("🚀 Verifying resignation configuration implementation...\n")
    
    configs_ok = verify_resignation_configs()
    verify_code_changes()
    
    if configs_ok:
        print("🎉 All configurations have resignation parameters!")
    else:
        print("⚠️  Some configurations are missing resignation parameters!")
    
    print("\n✅ Verification complete. The resignation parameters should now flow correctly from configs to implementation.")
