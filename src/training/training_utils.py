"""
Consolidated Training Utilities - DEDUPLICATION VERSION

This module consolidates all shared training utilities to eliminate duplication:
- Dirichlet noise application
- Temperature scheduling functions  
- Action selection utilities
- Training example filtering
- Game outcome determination

This eliminates duplication between AlphaZeroMCTS, StyleSpecificSelfPlay, and other modules.

Author: Francesco Finucci
"""

import math
import torch
import torch.nn.functional as F
import numpy as np
import chess
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum


class GameOutcome(Enum):
    """Enumeration for game outcomes."""
    WHITE_WINS = 1
    BLACK_WINS = -1
    DRAW = 0


@dataclass
class GameResult:
    """Consolidated game result information."""
    winner: Optional[int]  # 1 = white, -1 = black, 0 = draw, None = incomplete
    outcome_type: str  # "checkmate", "resignation", "draw", "stalemate", "repetition", etc.
    move_count: int
    final_fen: str
    resigned: bool = False
    resigning_player: Optional[int] = None


class TrainingUtilities:
    """
    Consolidated utilities for AlphaZero training to eliminate duplication.
    
    All shared functionality between MCTS, self-play, and training modules
    should be centralized here.
    """
    
    @staticmethod
    def add_dirichlet_noise(action_probs: Dict[Any, float], 
                           alpha: float = 0.3, 
                           epsilon: float = 0.25) -> Dict[Any, float]:
        """
        CENTRALIZED: Add Dirichlet noise to action probabilities for exploration.
        
        This replaces all scattered implementations of Dirichlet noise.
        
        Args:
            action_probs: Original action probabilities
            alpha: Dirichlet concentration parameter (0.3 for chess)
            epsilon: Noise mixing parameter (0.25 = 25% noise, 75% original)
            
        Returns:
            Action probabilities with added noise
        """
        if not action_probs or alpha <= 0:
            return action_probs
        
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        
        if len(actions) == 0:
            return action_probs
        
        try:
            # Generate Dirichlet noise
            noise = np.random.dirichlet([alpha] * len(actions))
            
            # Mix original probabilities with noise
            noisy_probs = {}
            for i, action in enumerate(actions):
                noisy_probs[action] = (1 - epsilon) * probs[i] + epsilon * noise[i]
            
            # Normalize to ensure probabilities sum to 1
            total_prob = sum(noisy_probs.values())
            if total_prob > 0:
                for action in noisy_probs:
                    noisy_probs[action] /= total_prob
            
            return noisy_probs
            
        except Exception as e:
            print(f"Warning: Failed to add Dirichlet noise: {e}")
            return action_probs
    
    @staticmethod
    def sample_action_from_probabilities(action_probs: Dict[Any, float], 
                                       temperature: float = 1.0) -> Any:
        """
        CENTRALIZED: Sample action from probability distribution.
        
        This replaces scattered action sampling logic.
        
        Args:
            action_probs: Action probability distribution
            temperature: Temperature for sampling (0.0 = deterministic)
            
        Returns:
            Sampled action
        """
        if not action_probs:
            return None
        
        actions = list(action_probs.keys())
        probabilities = list(action_probs.values())
        
        if len(actions) == 1:
            return actions[0]
        
        if temperature == 0.0:
            # Deterministic: choose action with highest probability
            return max(action_probs.keys(), key=lambda a: action_probs[a])
        else:
            # Stochastic: sample from distribution
            if sum(probabilities) > 0:
                # Normalize probabilities
                prob_tensor = torch.tensor(probabilities, dtype=torch.float32)
                prob_tensor = prob_tensor / prob_tensor.sum()
                
                # Sample action
                action_idx = torch.multinomial(prob_tensor, 1).item()
                return actions[action_idx]
            else:
                # Fallback if all probabilities are zero
                return actions[0]


class TemperatureSchedules:
    """
    CENTRALIZED: All temperature scheduling functions.
    
    This replaces scattered temperature schedule implementations.
    """
    
    @staticmethod
    def tactical_schedule(move_num: int) -> float:
        """
        Temperature schedule optimized for tactical play.
        More exploration in tactical phases.
        
        Args:
            move_num: Current move number (0-indexed)
            
        Returns:
            Temperature value for this move
        """
        if move_num < 10:
            return 1.8    # High exploration in opening
        elif move_num < 40:
            return 1.2    # Medium exploration in middlegame  
        elif move_num < 60:
            return 0.7    # Reduced exploration in late middlegame
        else:
            return 0.2    # Low exploration in endgame
    
    @staticmethod
    def positional_schedule(move_num: int) -> float:
        """
        Temperature schedule optimized for positional play.
        More consistent exploration for long-term planning.
        
        Args:
            move_num: Current move number (0-indexed)
            
        Returns:
            Temperature value for this move
        """
        if move_num < 15:
            return 1.5    # Moderate exploration in opening
        elif move_num < 50:
            return 1.0    # Consistent exploration in middlegame
        elif move_num < 80:
            return 0.6    # Reduced exploration in late middlegame
        else:
            return 0.3    # Some exploration in endgame
    
    @staticmethod
    def dynamic_schedule(move_num: int) -> float:
        """
        Temperature schedule optimized for dynamic play.
        Variable exploration based on game phase.
        
        Args:
            move_num: Current move number (0-indexed)
            
        Returns:
            Temperature value for this move
        """
        if move_num < 8:
            return 1.6    # High exploration early
        elif move_num < 25:
            return 1.4    # High exploration continues
        elif move_num < 45:
            return 0.9    # Medium exploration in complex middlegame
        elif move_num < 70:
            return 0.5    # Reduced exploration late middlegame
        else:
            return 0.1    # Minimal exploration in endgame
    
    @staticmethod
    def standard_schedule(move_num: int) -> float:
        """
        Standard temperature schedule for general play.
        
        Args:
            move_num: Current move number (0-indexed)
            
        Returns:
            Temperature value for this move
        """
        if move_num < 30:
            return 1.0    # Exploration in opening/early middlegame
        else:
            return 0.1    # Minimal exploration afterwards
    
    @staticmethod
    def get_schedule_for_style(style: str) -> Callable[[int], float]:
        """
        Get temperature schedule function for a given style.
        
        Args:
            style: Playing style ('tactical', 'positional', 'dynamic', or 'standard')
            
        Returns:
            Temperature schedule function
        """
        schedules = {
            'tactical': TemperatureSchedules.tactical_schedule,
            'positional': TemperatureSchedules.positional_schedule,
            'dynamic': TemperatureSchedules.dynamic_schedule,
            'standard': TemperatureSchedules.standard_schedule
        }
        
        return schedules.get(style, TemperatureSchedules.standard_schedule)


class TrainingExampleFilters:
    """
    CENTRALIZED: Training example filtering utilities.
    
    This replaces scattered filtering logic.
    """
    
    @staticmethod
    def filter_decisive_games(training_examples: List[Any]) -> List[Any]:
        """
        CENTRALIZED: Filter training examples to focus on decisive games.
        
        Args:
            training_examples: List of training examples from a game
            
        Returns:
            Filtered examples (empty list if game was drawn)
        """
        if not training_examples:
            return []
        
        # Check if any example has non-zero outcome (decisive game)
        has_decisive_outcome = any(abs(ex.outcome) > 0.1 for ex in training_examples)
        
        if has_decisive_outcome:
            return training_examples
        else:
            # Filter out drawn games during training to encourage decisive play
            return []
    
    @staticmethod
    def filter_by_game_length(training_examples: List[Any], 
                            min_length: int = 10, 
                            max_length: int = 200) -> List[Any]:
        """
        Filter games by length to avoid too short or too long games.
        
        Args:
            training_examples: List of training examples
            min_length: Minimum game length
            max_length: Maximum game length
            
        Returns:
            Filtered examples or empty list if game length is invalid
        """
        if min_length <= len(training_examples) <= max_length:
            return training_examples
        else:
            return []
    
    @staticmethod
    def filter_by_outcome_diversity(training_examples: List[Any], 
                                  min_outcome_ratio: float = 0.1) -> List[Any]:
        """
        Filter games to ensure outcome diversity in training data.
        
        Args:
            training_examples: List of training examples
            min_outcome_ratio: Minimum ratio of non-neutral outcomes
            
        Returns:
            Filtered examples or empty list if outcomes are too uniform
        """
        if not training_examples:
            return []
        
        # Count outcome types
        outcomes = [ex.outcome for ex in training_examples]
        wins = sum(1 for o in outcomes if o > 0.5)
        losses = sum(1 for o in outcomes if o < -0.5)
        decisive = wins + losses
        
        # Check if game has sufficient outcome diversity
        decisive_ratio = decisive / len(training_examples)
        
        if decisive_ratio >= min_outcome_ratio:
            return training_examples
        else:
            return []


class ResignationLogic:
    """
    CENTRALIZED: Resignation decision logic.
    
    This replaces scattered resignation implementations.
    """
    
    @staticmethod
    def should_resign_by_evaluation(value: float, 
                                  threshold: float = -0.9,
                                  move_count: int = 0,
                                  min_moves: int = 20) -> bool:
        """
        Determine resignation based on neural network evaluation.
        
        Args:
            value: Current position evaluation from neural network
            threshold: Resignation threshold (default -0.9 = 90% loss probability)
            move_count: Current move number
            min_moves: Minimum moves before considering resignation
            
        Returns:
            True if should resign, False otherwise
        """
        # Only consider resignation after opening
        if move_count < min_moves:
            return False
        
        return value < threshold
    
    @staticmethod
    def should_resign_by_material(board: chess.Board, 
                                threshold_centipawns: int = 500) -> bool:
        """
        Determine resignation based on material imbalance.
        
        Args:
            board: Current chess position
            threshold_centipawns: Material threshold in centipawns
            
        Returns:
            True if should resign due to material disadvantage
        """
        # Import here to avoid circular imports
        from ..core.game_utils import should_resign_material
        return should_resign_material(board, threshold_centipawns)
    
    @staticmethod
    def should_resign_combined(value: float, 
                             board: chess.Board,
                             move_count: int,
                             eval_threshold: float = -0.9,
                             material_threshold: int = 500,
                             min_moves: int = 20) -> tuple[bool, str]:
        """
        Combined resignation logic using both evaluation and material.
        
        Args:
            value: Neural network evaluation
            board: Current chess position
            move_count: Current move number
            eval_threshold: Neural network resignation threshold
            material_threshold: Material resignation threshold in centipawns
            min_moves: Minimum moves before considering resignation
            
        Returns:
            Tuple of (should_resign, reason)
        """
        if move_count < min_moves:
            return False, "too_early"
        
        # Check neural network evaluation
        if ResignationLogic.should_resign_by_evaluation(
            value, eval_threshold, move_count, min_moves
        ):
            return True, "evaluation"
        
        # Check material imbalance
        if ResignationLogic.should_resign_by_material(board, material_threshold):
            return True, "material"
        
        return False, "continue"


class GameOutcomeAnalyzer:
    """
    CENTRALIZED: Game outcome analysis and assignment.
    
    This replaces scattered outcome determination logic.
    """
    
    @staticmethod
    def analyze_terminal_position(board: chess.Board) -> GameResult:
        """
        Analyze a terminal chess position to determine the outcome.
        
        Args:
            board: Terminal chess position
            
        Returns:
            GameResult with detailed outcome information
        """
        if not board.is_game_over():
            raise ValueError("Position is not terminal")
        
        outcome = board.outcome()
        move_count = board.fullmove_number
        final_fen = board.fen()
        
        if outcome is None:
            # Shouldn't happen if is_game_over() is True
            return GameResult(
                winner=None,
                outcome_type="unknown",
                move_count=move_count,
                final_fen=final_fen
            )
        
        # Determine winner
        if outcome.winner is None:
            winner = 0  # Draw
        elif outcome.winner == chess.WHITE:
            winner = 1  # White wins
        else:
            winner = -1  # Black wins
        
        # Determine outcome type
        if outcome.termination == chess.Termination.CHECKMATE:
            outcome_type = "checkmate"
        elif outcome.termination == chess.Termination.STALEMATE:
            outcome_type = "stalemate"
        elif outcome.termination == chess.Termination.INSUFFICIENT_MATERIAL:
            outcome_type = "insufficient_material"
        elif outcome.termination == chess.Termination.SEVENTYFIVE_MOVES:
            outcome_type = "75_move_rule"
        elif outcome.termination == chess.Termination.FIVEFOLD_REPETITION:
            outcome_type = "fivefold_repetition"
        elif outcome.termination == chess.Termination.FIFTY_MOVES:
            outcome_type = "50_move_rule"
        elif outcome.termination == chess.Termination.THREEFOLD_REPETITION:
            outcome_type = "threefold_repetition"
        else:
            outcome_type = "other"
        
        return GameResult(
            winner=winner,
            outcome_type=outcome_type,
            move_count=move_count,
            final_fen=final_fen,
            resigned=False
        )
    
    @staticmethod
    def create_resignation_result(resigning_player: int, 
                                move_count: int, 
                                final_fen: str) -> GameResult:
        """
        Create a GameResult for a resignation.
        
        Args:
            resigning_player: Player who resigned (1 = white, -1 = black)
            move_count: Move when resignation occurred
            final_fen: Final position FEN
            
        Returns:
            GameResult for resignation
        """
        return GameResult(
            winner=-resigning_player,  # Opponent wins
            outcome_type="resignation",
            move_count=move_count,
            final_fen=final_fen,
            resigned=True,
            resigning_player=resigning_player
        )
    
    @staticmethod
    def assign_training_outcomes(training_examples: List[Any], 
                               game_result: GameResult) -> None:
        """
        CENTRALIZED: Assign outcomes to training examples based on game result.
        
        This replaces scattered outcome assignment logic.
        
        Args:
            training_examples: List of training examples to update
            game_result: Result of the completed game
        """
        if not training_examples or game_result.winner is None:
            return
        
        final_outcome = float(game_result.winner)  # 1.0, -1.0, or 0.0
        
        for example in training_examples:
            if game_result.winner == 0:
                # Draw
                example.outcome = 0.0
            else:
                # Assign outcome from the perspective of the player who made the move
                if example.current_player == game_result.winner:
                    example.outcome = 1.0  # This player won
                else:
                    example.outcome = -1.0  # This player lost


# Convenience functions for backward compatibility
def add_dirichlet_noise(action_probs: Dict[Any, float], alpha: float = 0.3) -> Dict[Any, float]:
    """Convenience function for backward compatibility."""
    return TrainingUtilities.add_dirichlet_noise(action_probs, alpha)

def tactical_temperature_schedule(move_num: int) -> float:
    """Convenience function for backward compatibility."""
    return TemperatureSchedules.tactical_schedule(move_num)

def filter_decisive_games(training_examples: List[Any]) -> List[Any]:
    """Convenience function for backward compatibility."""
    return TrainingExampleFilters.filter_decisive_games(training_examples)