"""
Style-Specific Self-Play for AlphaZero Training

This module provides self-play game generation with style-specific opening selection
using the ECO opening database to imprint tactical, positional, or dynamic playing styles.

Author: Francesco Finucci
"""

import torch
import torch.nn.functional as F
import chess
import numpy as np
import logging
import time
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
import random

from ..core.alphazero_mcts import AlphaZeroMCTS, AlphaZeroTrainingExample
from ..core.alphazero_net import AlphaZeroNet
from ..core.game_utils import ChessPosition
from ..data.openings.openings import ECO_OpeningDatabase, OpeningTemplate


@dataclass
class SelfPlayGameResult:
    """Result from a single self-play game."""
    training_examples: List[AlphaZeroTrainingExample]
    game_length: int
    final_result: float  # 1.0 = white wins, -1.0 = black wins, 0.0 = draw
    opening_used: Optional[str] = None
    style_adherence: float = 0.0  # How well the game followed the target style


class StyleSpecificSelfPlay:
    """
    Generates self-play training data with style-specific opening selection.
    
    Uses the ECO opening database to force specific playing styles during
    the opening phase, then allows AlphaZero MCTS to take over naturally.
    This creates training data biased toward tactical, positional, or dynamic styles.
    """
    
    def __init__(self, device: str = 'cuda', logger: logging.Logger = None):
        """
        Initialize style-specific self-play generator.
        
        Args:
            device: PyTorch device for neural network inference
            logger: Logger instance to use (if None, creates its own)
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Initialize ECO opening database
        self.opening_db = ECO_OpeningDatabase()
        
        # Setup logging - use provided logger or create own
        if logger is not None:
            self.logger = logger
        else:
            self.logger = logging.getLogger(f"StyleSpecificSelfPlay_{id(self)}")
        
        # Statistics tracking
        self.games_generated = 0
        self.openings_used = {}
        self.style_stats = {
            'tactical': {'games': 0, 'avg_length': 0},
            'positional': {'games': 0, 'avg_length': 0},
            'dynamic': {'games': 0, 'avg_length': 0}
        }
        
        self.logger.info("StyleSpecificSelfPlay initialized")
        self.logger.info(f"Available openings: Tactical={len(self.opening_db.get_openings_for_style('tactical'))}, "
                        f"Positional={len(self.opening_db.get_openings_for_style('positional'))}, "
                        f"Dynamic={len(self.opening_db.get_openings_for_style('dynamic'))}")
    
    def generate_training_examples(self, model: AlphaZeroNet, style: str, num_games: int,
                                 mcts_simulations: int = 800, dirichlet_alpha: float = 0.3,
                                 temperature_moves: int = 30) -> List[AlphaZeroTrainingExample]:
        """
        Generate training examples through style-specific self-play.
        
        Args:
            model: AlphaZero neural network for move evaluation
            style: Target playing style ('tactical', 'positional', 'dynamic')
            num_games: Number of self-play games to generate
            mcts_simulations: Number of MCTS simulations per move
            dirichlet_alpha: Dirichlet noise parameter for exploration
            temperature_moves: Number of moves to use temperature > 0
            
        Returns:
            List of training examples from all generated games
        """
        if style not in ['tactical', 'positional', 'dynamic']:
            raise ValueError(f"Unknown style: {style}. Must be 'tactical', 'positional', or 'dynamic'")
        
        self.logger.info(f"Generating {num_games} {style} style games with {mcts_simulations} MCTS sims")
        
        # Create MCTS engine for this model
        self.logger.info(f"Creating MCTS engine for {style} style games...")
        mcts_engine = AlphaZeroMCTS(model, c_puct=1.0, device=self.device)
        self.logger.info(f"MCTS engine created successfully")
        
        all_training_examples = []
        successful_games = 0
        
        for game_idx in range(num_games):
            try:
                self.logger.info(f"Starting game {game_idx + 1}/{num_games} ({style} style)")
                # Generate single self-play game with style-specific opening
                game_result = self._generate_single_game(
                    mcts_engine=mcts_engine,
                    style=style,
                    mcts_simulations=mcts_simulations,
                    dirichlet_alpha=dirichlet_alpha,
                    temperature_moves=temperature_moves,
                    game_id=f"{style}_{game_idx}"
                )
                
                if game_result and game_result.training_examples:
                    all_training_examples.extend(game_result.training_examples)
                    successful_games += 1
                    self.logger.info(f"Game {game_idx + 1} completed successfully - {len(game_result.training_examples)} examples, {game_result.game_length} moves")
                    
                    # Update statistics
                    self.style_stats[style]['games'] += 1
                    current_avg = self.style_stats[style]['avg_length']
                    games_count = self.style_stats[style]['games']
                    new_avg = ((current_avg * (games_count - 1)) + game_result.game_length) / games_count
                    self.style_stats[style]['avg_length'] = new_avg
                else:
                    self.logger.warning(f"Game {game_idx + 1} failed to generate valid training examples")
                    
                    if game_result.opening_used:
                        self.openings_used[game_result.opening_used] = self.openings_used.get(game_result.opening_used, 0) + 1
                
            except Exception as e:
                self.logger.warning(f"Failed to generate game {game_idx} for style {style}: {e}")
                continue
        
        self.games_generated += successful_games
        
        self.logger.info(f"Generated {successful_games}/{num_games} successful {style} games, "
                        f"{len(all_training_examples)} training examples")
        
        return all_training_examples
    
    def _generate_single_game(self, mcts_engine: AlphaZeroMCTS, style: str,
                             mcts_simulations: int, dirichlet_alpha: float,
                             temperature_moves: int, game_id: str) -> Optional[SelfPlayGameResult]:
        """
        Generate a single self-play game with style-specific opening.
        
        Args:
            mcts_engine: MCTS engine for move selection
            style: Target playing style
            mcts_simulations: MCTS simulations per move
            dirichlet_alpha: Dirichlet noise parameter
            temperature_moves: Moves with temperature > 0
            game_id: Unique identifier for this game
            
        Returns:
            SelfPlayGameResult with training examples and metadata
        """
        # Select style-specific opening
        opening = self._select_opening_for_style(style)
        
        # Create initial position from opening
        initial_position = self._create_position_from_opening(opening)
        
        # Generate training examples through self-play
        training_examples = []
        current_position = initial_position.clone()
        move_count = 0
        max_moves = 80  # Prevent infinite games - increased from 200 to allow more natural endings
        
        # Resignation parameters
        resign_threshold = 0.15  # Resign if win probability < 15% (less aggressive)
        resign_earliest_move = 20  # Don't resign before move 20
        consecutive_bad_evals = 0  # Track consecutive bad evaluations
        resign_consistency_required = 5  # Need 5 consecutive bad evals to resign (more conservative)
        
        # Track game state for style adherence calculation
        opening_moves_played = len(opening.moves) if opening else 0
        
        while not current_position.is_terminal() and move_count < max_moves:
            # Calculate temperature based on move number
            if move_count < temperature_moves:
                temperature = 1.0
            else:
                temperature = 0.0  # Deterministic play after temperature_moves
            
            # Get move probabilities from MCTS
            try:
                move_start_time = time.time()
                action_probs = mcts_engine.get_action_probabilities(
                    current_position,
                    num_simulations=mcts_simulations,
                    temperature=temperature
                )
                move_time = time.time() - move_start_time
                
                # Log timing every 10 moves to avoid spam
                if move_count % 10 == 0:
                    self.logger.info(f"Move {move_count}: {move_time:.2f}s ({mcts_simulations} sims)")
                
                if not action_probs:
                    break  # No legal moves
                
                # Add Dirichlet noise to root node for exploration
                if move_count < temperature_moves:
                    action_probs = self._add_dirichlet_noise(action_probs, dirichlet_alpha)
                
                # Check for resignation (if enabled and past earliest move)
                if move_count >= resign_earliest_move:
                    # Get current position evaluation from MCTS value
                    # We can estimate this from the neural network or use a simple heuristic
                    root_node = mcts_engine.search(current_position, mcts_simulations)
                    position_value = root_node.value_sum / max(root_node.visit_count, 1)
                    
                    # Convert neural network value to win probability (roughly)
                    # Neural network outputs are in [-1, 1], convert to [0, 1]
                    win_probability = (position_value + 1.0) / 2.0
                    
                    # Check if position is resignable
                    if win_probability < resign_threshold or win_probability > (1.0 - resign_threshold):
                        consecutive_bad_evals += 1
                        if consecutive_bad_evals >= resign_consistency_required:
                            # Resign the game
                            final_reward = -1.0 if win_probability < resign_threshold else 1.0
                            # Adjust for current player perspective
                            if current_position.get_current_player() == -1:  # Black to move
                                final_reward = -final_reward
                            
                            termination_reason = f"resignation(eval={win_probability:.3f},threshold={resign_threshold})"
                            self.logger.info(f"Game {game_id} resigned after {move_count} moves: {termination_reason}")
                            
                            # Update training examples with resignation outcome
                            for i, example in enumerate(training_examples):
                                if example.current_player == current_position.get_current_player():
                                    example.outcome = final_reward
                                else:
                                    example.outcome = -final_reward
                            
                            return SelfPlayGameResult(
                                training_examples=training_examples,
                                game_length=move_count,
                                final_result=final_reward,
                                opening_used=opening.name if opening else None,
                                style_adherence=min(opening_moves_played / max(move_count, 1), 1.0) if opening else 0.0
                            )
                    else:
                        consecutive_bad_evals = 0  # Reset counter
                
                # Create training example
                training_example = AlphaZeroTrainingExample(
                    state_tensor=current_position.to_tensor().clone(),
                    action_probs=action_probs.copy(),
                    outcome=0.0,  # Will be updated with final game result
                    current_player=current_position.get_current_player()
                )
                training_examples.append(training_example)
                
                # Sample action from probabilities
                if temperature == 0.0:
                    # Deterministic: choose best action
                    action = max(action_probs.keys(), key=lambda a: action_probs[a])
                else:
                    # Stochastic: sample from distribution
                    actions = list(action_probs.keys())
                    probabilities = list(action_probs.values())
                    action = np.random.choice(actions, p=probabilities)
                
                # Apply the selected action
                current_position = current_position.apply_action(action)
                move_count += 1
                
            except Exception as e:
                self.logger.warning(f"Error during move {move_count} in game {game_id}: {e}")
                break
        
        # Determine final game result
        if current_position.is_terminal():
            final_reward = current_position.get_reward()
            termination_reason = "natural"
        else:
            # Game didn't finish naturally - treat as draw
            final_reward = 0.0
            termination_reason = f"move_limit({max_moves})"
        
        # Update all training examples with final outcome
        for i, example in enumerate(training_examples):
            # Outcome is from perspective of the player who made that move
            if example.current_player == current_position.get_current_player():
                example.outcome = final_reward
            else:
                example.outcome = -final_reward
        
        # Debug: Log game outcome for analysis
        outcome_str = "White wins" if final_reward == 1.0 else "Black wins" if final_reward == -1.0 else "Draw"
        self.logger.info(f"Game outcome: {outcome_str} (final_reward={final_reward}) - {termination_reason}")
        
        # Calculate style adherence (simplified - based on opening usage)
        style_adherence = min(opening_moves_played / max(move_count, 1), 1.0) if opening else 0.0
        
        return SelfPlayGameResult(
            training_examples=training_examples,
            game_length=move_count,
            final_result=final_reward,
            opening_used=opening.name if opening else None,
            style_adherence=style_adherence
        )
    
    def _select_opening_for_style(self, style: str) -> Optional[OpeningTemplate]:
        """
        Select an opening template based on the target style.
        
        Args:
            style: Target playing style
            
        Returns:
            Selected opening template or None if no openings available
        """
        try:
            return self.opening_db.sample_opening_for_style(style)
        except Exception as e:
            self.logger.warning(f"Failed to select opening for style {style}: {e}")
            return None
    
    def _create_position_from_opening(self, opening: Optional[OpeningTemplate]) -> ChessPosition:
        """
        Create a chess position from an opening template.
        
        Args:
            opening: Opening template with move sequence
            
        Returns:
            ChessPosition after playing the opening moves
        """
        board = chess.Board()
        
        if opening is not None:
            try:
                # Play opening moves up to a random depth for variation
                max_depth = min(len(opening.moves), opening.continuation_depth)
                
                # Sometimes play partial opening for more variation
                if max_depth > 4:
                    depth = random.randint(max(4, max_depth - 3), max_depth)
                else:
                    depth = max_depth
                
                for i in range(depth):
                    if i < len(opening.moves):
                        move_san = opening.moves[i]
                        try:
                            move = board.parse_san(move_san)
                            if move in board.legal_moves:
                                board.push(move)
                            else:
                                self.logger.warning(f"Illegal move in opening {opening.name}: {move_san}")
                                break
                        except Exception as e:
                            self.logger.warning(f"Failed to parse move {move_san} in opening {opening.name}: {e}")
                            break
                    else:
                        break
                        
            except Exception as e:
                self.logger.warning(f"Error applying opening {opening.name}: {e}")
                # Fall back to starting position
                board = chess.Board()
        
        return ChessPosition(board)
    
    def _add_dirichlet_noise(self, action_probs: Dict[chess.Move, float], 
                            alpha: float) -> Dict[chess.Move, float]:
        """
        Add Dirichlet noise to action probabilities for exploration.
        
        Args:
            action_probs: Original action probabilities
            alpha: Dirichlet distribution parameter
            
        Returns:
            Action probabilities with added noise
        """
        if not action_probs or alpha <= 0:
            return action_probs
        
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        
        try:
            # Generate Dirichlet noise
            noise = np.random.dirichlet([alpha] * len(actions))
            
            # Mix original probabilities with noise (0.75 original, 0.25 noise)
            noisy_probs = 0.75 * np.array(probs) + 0.25 * noise
            
            # Normalize to ensure probabilities sum to 1
            noisy_probs = noisy_probs / np.sum(noisy_probs)
            
            return {action: prob for action, prob in zip(actions, noisy_probs)}
            
        except Exception as e:
            self.logger.warning(f"Failed to add Dirichlet noise: {e}")
            return action_probs
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about generated games and opening usage.
        
        Returns:
            Dictionary with detailed statistics
        """
        return {
            'total_games_generated': self.games_generated,
            'style_statistics': self.style_stats.copy(),
            'opening_usage': self.openings_used.copy(),
            'most_used_openings': sorted(
                self.openings_used.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10],
            'average_game_lengths': {
                style: stats['avg_length'] 
                for style, stats in self.style_stats.items()
            }
        }
    
    def reset_statistics(self) -> None:
        """Reset all statistics tracking."""
        self.games_generated = 0
        self.openings_used.clear()
        self.style_stats = {
            'tactical': {'games': 0, 'avg_length': 0},
            'positional': {'games': 0, 'avg_length': 0},
            'dynamic': {'games': 0, 'avg_length': 0}
        }
        self.logger.info("Statistics reset")


# Utility functions for testing and validation
def validate_style_specific_generation(style: str, num_test_games: int = 5) -> None:
    """
    Validate that style-specific self-play generation works correctly.
    
    Args:
        style: Style to test ('tactical', 'positional', 'dynamic')
        num_test_games: Number of test games to generate
    """
    print(f"Validating {style} style generation with {num_test_games} games...")
    
    # Create a dummy model for testing
    from ..core.alphazero_net import create_alphazero_chess_net
    test_model = create_alphazero_chess_net()
    
    # Create self-play generator
    generator = StyleSpecificSelfPlay()
    
    # Generate test games
    training_examples = generator.generate_training_examples(
        model=test_model,
        style=style,
        num_games=num_test_games,
        mcts_simulations=100,  # Reduced for testing
        temperature_moves=10
    )
    
    # Display results
    stats = generator.get_statistics()
    print(f"Generated {len(training_examples)} training examples")
    print(f"Style statistics: {stats['style_statistics'][style]}")
    print(f"Openings used: {stats['opening_usage']}")
    
    return len(training_examples) > 0


def test_all_styles() -> None:
    """Test self-play generation for all three styles."""
    styles = ['tactical', 'positional', 'dynamic']
    
    for style in styles:
        print(f"\n{'='*50}")
        print(f"Testing {style.upper()} style")
        print(f"{'='*50}")
        
        try:
            success = validate_style_specific_generation(style, num_test_games=2)
            print(f"✅ {style} style test: {'PASSED' if success else 'FAILED'}")
        except Exception as e:
            print(f"❌ {style} style test FAILED: {e}")


if __name__ == "__main__":
    # Run validation tests
    test_all_styles()