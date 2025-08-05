#!/bin/bash

# Create main project directory
mkdir -p diversity_preserving_chess_fl
cd diversity_preserving_chess_fl

# Create core directories
mkdir -p core training federated explainability evaluation
mkdir -p data/{openings,test_positions,models/{individual,clustered,final}}
mkdir -p experiments config utils tests scripts notebooks docs

# Create __init__.py files
touch core/__init__.py training/__init__.py federated/__init__.py
touch explainability/__init__.py evaluation/__init__.py experiments/__init__.py
touch config/__init__.py utils/__init__.py tests/__init__.py

# Create main files for Week 1-2
touch core/alphazero_net.py core/mcts.py core/chess_engine.py core/game_utils.py
touch training/self_play.py training/trainer.py training/reward_shaping.py
touch config/model_config.py config/training_config.py
touch utils/chess_utils.py utils/logging_utils.py
touch scripts/train_individual_engines.py
touch tests/test_alphazero.py

# Create requirements.txt
cat > requirements.txt << EOF
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0
python-chess>=1.999
flwr>=1.5.0
shap>=0.42.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
jupyter>=1.0.0
pytest>=7.0.0
tqdm>=4.64.0
tensorboard>=2.13.0
pandas>=2.0.0
plotly>=5.15.0
EOF

# Create basic setup.py
cat > setup.py << EOF
from setuptools import setup, find_packages

setup(
    name="diversity-preserving-chess-fl",
    version="0.1.0",
    description="Diversity-Preserving Federated Reinforcement Learning for Chess",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "python-chess>=1.999",
        "flwr>=1.5.0",
        "shap>=0.42.0",
        "scikit-learn>=1.3.0",
        "numpy>=1.21.0",
    ],
    python_requires=">=3.8",
)
EOF

# Create .gitignore
cat > .gitignore << EOF
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# PyTorch
*.pth
*.pt

# Jupyter Notebook
.ipynb_checkpoints

# Environment
.env
.venv
env/
venv/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Data
data/models/*.pth
data/models/*.pt
*.pgn
*.log

# OS
.DS_Store
Thumbs.db
EOF

# Create basic README
cat > README.md << EOF
# Diversity-Preserving Federated Reinforcement Learning for Chess

This project implements a novel federated learning framework that preserves beneficial diversity in chess playing styles while enabling collaborative learning.

## Key Innovation
- Uses reward shaping to create distinct chess playing styles (tactical, positional, dynamic)
- Employs SHAP-based clustering to automatically discover behavioral patterns
- Preserves style diversity during federated learning through selective knowledge transfer

## Quick Start
\`\`\`bash
# Install dependencies
pip install -r requirements.txt

# Train individual chess engines with different styles
python scripts/train_individual_engines.py

# Run federated learning experiment
python scripts/run_federated_learning.py
\`\`\`

## Project Structure
See the complete directory structure in the docs/ folder.

## Week 1-2 Goals
- Implement working AlphaZero architecture
- Create self-play training loop
- Validate chess playing ability
- Set foundation for style differentiation
EOF

echo "✅ Project structure created successfully!"
echo "📁 Navigate to diversity_preserving_chess_fl/ to start coding"
echo "🚀 Next: Implement core/alphazero_net.py"