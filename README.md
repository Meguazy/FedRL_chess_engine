# Diversity-Preserving Federated Reinforcement Learning for Chess

This project implements a novel federated learning framework that preserves beneficial diversity in chess playing styles while enabling collaborative learning.

## Key Innovation
- Uses reward shaping to create distinct chess playing styles (tactical, positional, dynamic)
- Employs SHAP-based clustering to automatically discover behavioral patterns
- Preserves style diversity during federated learning through selective knowledge transfer

## Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Train individual chess engines with different styles
python scripts/train_individual_engines.py

# Run federated learning experiment
python scripts/run_federated_learning.py
```

## Project Structure
See the complete directory structure in the docs/ folder.

## Week 1-2 Goals
- Implement working AlphaZero architecture
- Create self-play training loop
- Validate chess playing ability
- Set foundation for style differentiation
