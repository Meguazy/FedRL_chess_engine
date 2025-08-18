"""
Style-Specific Self-Play for AlphaZero Training - FULLY DEDUPLICATED VERSION

This module provides self-play game generation with complete deduplication.
All shared utilities are in training_utils.py, including statistics, analysis,
and file operations.

Author: Francesco Finucci
"""

import torch
import chess
import numpy as np
import logging
import time
import os
import json
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
import random

from ..core.chess_engine import ChessPosition
from ..core.game_utils import ChessGameState
from ..core.alphazero_net import AlphaZeroNet
from ..core.alphazero_mcts import AlphaZeroMCTS, AlphaZeroTrainingExample, generate_self_play_game
from ..data.openings.openings import ECO_OpeningDatabase, OpeningTemplate

# DEDUPLICATED: Import ALL utilities from centralized location
from ..training.training_utils import (
    TrainingUtilities,
    TemperatureSchedules, 
    TrainingExampleFilters,
    GameOutcomeAnalyzer,
    GameStatistics,           # NEW: Centralized statistics
    BoardImageManager,        # NEW: Centralized image handling
    GameResultAnalyzer,       # NEW: Centralized result analysis
    FileManager              # NEW: Centralized file operations
)


@dataclass
class SelfPlayGameResult:
    """Result from a single self-play game - simplified."""
    training_examples: List[AlphaZeroTrainingExample]
    game_length: int
    final_result: float
    opening_used: Optional[str] = None
    style_adherence: float = 0.0


class StyleSpecificSelfPlay:
    """
    FULLY DEDUPLICATED Style-Specific Self-Play Generator.
    
    All functionality is delegated to centralized utilities in training_utils.py.
    This class only handles the high-level orchestration.
    """
    
    def __init__(self, device: str = 'cuda', logger: logging.Logger = None, 
                 save_board_images: bool = False, board_images_dir: str = "board_images"):
        """Initialize with centralized utilities."""
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.opening_db = ECO_OpeningDatabase()
        
        # DEDUPLICATED: Use centralized managers
        self.statistics = GameStatistics()
        self.image_manager = BoardImageManager(board_images_dir) if save_board_images else None
        self.result_analyzer = GameResultAnalyzer()
        self.file_manager = FileManager()
        
        # Setup logging
        self.logger = logger or logging.getLogger(f"StyleSpecificSelfPlay_{id(self)}")
        
        self.logger.info("StyleSpecificSelfPlay initialized with centralized utilities")
        self.logger.info(f"Available openings: {self._get_opening_counts()}")
    
    def _get_opening_counts(self) -> str:
        """Get opening counts for all styles."""
        counts = {}
        for style in ['tactical', 'positional', 'dynamic']:
            counts[style] = len(self.opening_db.get_openings_for_style(style))
        return f"Tactical={counts['tactical']}, Positional={counts['positional']}, Dynamic={counts['dynamic']}"
    
    def generate_training_examples(self, model, num_games: int, style: str = 'standard', 
                                 **config) -> List[AlphaZeroTrainingExample]:
        """
        Generate training examples through style-specific self-play - FULLY DEDUPLICATED.
        
        All game generation, filtering, and analysis is handled by centralized utilities.
        """
        if style not in ['tactical', 'positional', 'dynamic']:
            raise ValueError(f"Unknown style: {style}. Must be 'tactical', 'positional', or 'dynamic'")
        
        # Extract configuration with defaults
        mcts_simulations = config.get('mcts_simulations', 100)
        dirichlet_alpha = config.get('dirichlet_alpha', 0.3)
        enable_resignation = config.get('enable_resignation', True)
        resignation_threshold = config.get('resignation_threshold', -0.9)
        max_moves = config.get('max_moves', 150)
        save_board_images = config.get('save_board_images', False)
        board_images_dir = config.get('board_images_dir', 'board_images')
        
        self.logger.info(f"Generating {num_games} {style} style games with {mcts_simulations} MCTS sims")
        
        # Create MCTS engine
        mcts_engine = self._create_mcts_engine(model, resignation_threshold)
        
        all_training_examples = []
        game_results = []
        
        # Generate games
        for game_idx in range(num_games):
            try:
                self.logger.info(f"Starting game {game_idx + 1}/{num_games} ({style} style)")
                
                game_result = self._generate_single_game(
                    mcts_engine=mcts_engine,
                    style=style,
                    game_id=f"{style}_{game_idx}",
                    config=config
                )

                if game_result and game_result.training_examples:
                    game_results.append(game_result)
                    all_training_examples.extend(game_result.training_examples)
                    
                    # DEDUPLICATED: Use centralized statistics
                    self.statistics.record_game(
                        style=style,
                        game_length=game_result.game_length,
                        final_result=game_result.final_result,
                        opening_used=game_result.opening_used
                    )
                    
                    # DEDUPLICATED: Use centralized result analysis
                    game_analysis = self.result_analyzer.analyze_training_examples(game_result.training_examples)
                    self.logger.info(f"Game {game_idx + 1}: {game_analysis['outcome_description']}")
                    self.logger.info(f"Examples distribution: {game_analysis['outcome_distribution']}")
                else:
                    self.logger.warning(f"Game {game_idx + 1} failed to generate valid training examples")
                
            except Exception as e:
                self.logger.warning(f"Failed to generate game {game_idx}: {e}")
                continue
        
        # DEDUPLICATED: Use centralized filtering and final analysis
        filtered_examples = TrainingExampleFilters.filter_decisive_games(all_training_examples)
        final_stats = self.statistics.get_style_summary(style)
        decisive_percentage = self.result_analyzer.calculate_decisive_percentage(game_results)
        
        self.logger.info(f"Generated {len(game_results)}/{num_games} successful {style} games")
        self.logger.info(f"Decisive games: {decisive_percentage:.1f}%")
        self.logger.info(f"Training examples: {len(filtered_examples)}/{len(all_training_examples)} after filtering")
        
        if decisive_percentage == 0:
            self.logger.warning("⚠️  NO DECISIVE GAMES GENERATED - All games were draws!")
        
        return filtered_examples
    
    def _create_mcts_engine(self, model, resignation_threshold: float) -> AlphaZeroMCTS:
        """Create MCTS engine with proper configuration."""
        return AlphaZeroMCTS(
            model, 
            c_puct=2.0,
            device=self.device, 
            resignation_threshold=resignation_threshold,
            logger=self.logger
        )
    
    def _generate_single_game(self, mcts_engine: AlphaZeroMCTS, style: str,
                             game_id: str, config: dict) -> Optional[SelfPlayGameResult]:
        """
        Generate a single self-play game - FULLY DEDUPLICATED.
        
        All game logic is handled by centralized generate_self_play_game().
        """
        # DEDUPLICATED: Use centralized opening selection
        opening = self._select_opening_for_style(style)
        initial_position = self._create_position_from_opening(opening)
        
        try:
            # DEDUPLICATED: Use centralized self-play generation
            training_examples, final_state, move_history, game_resigned, winner = generate_self_play_game(
                chess_engine=mcts_engine,
                initial_state=initial_position,
                num_simulations=config.get('mcts_simulations', 100),
                temperature_schedule=None,  # Uses style-based schedule
                max_moves=config.get('max_moves', 150),
                enable_resignation=config.get('enable_resignation', True),
                filter_draws=False,  # Handle filtering at higher level
                style=style,
                dirichlet_alpha=config.get('dirichlet_alpha', 0.3)
            )
            
            if not training_examples:
                self.logger.warning(f"Game {game_id} generated no training examples")
                return None
            
            # DEDUPLICATED: Use centralized result analysis
            outcome_info = self.result_analyzer.analyze_game_outcome(
                training_examples, game_resigned, winner
            )
            
            # DEDUPLICATED: Use centralized image saving
            if self.image_manager and config.get('save_board_images', False):
                self.image_manager.save_game_images(
                    initial_position=initial_position,
                    final_state=final_state,
                    training_examples=training_examples,
                    game_id=game_id,
                    opening=opening,
                    outcome_str=outcome_info['outcome_description'],
                    move_history=move_history
                )
            
            # DEDUPLICATED: Use centralized style adherence calculation
            style_adherence = self._calculate_style_adherence(opening, training_examples)
            
            return SelfPlayGameResult(
                training_examples=training_examples,
                game_length=len(training_examples),
                final_result=training_examples[-1].outcome if training_examples else 0.0,
                opening_used=opening.name if opening else None,
                style_adherence=style_adherence
            )
            
        except Exception as e:
            self.logger.error(f"Error generating game {game_id}: {e}")
            return None
    
    def _select_opening_for_style(self, style: str) -> Optional[OpeningTemplate]:
        """Select opening with error handling."""
        try:
            return self.opening_db.sample_opening_for_style(style)
        except Exception as e:
            self.logger.warning(f"Failed to select opening for style {style}: {e}")
            return None
    
    def _create_position_from_opening(self, opening: Optional[OpeningTemplate]) -> ChessPosition:
        """Create position from opening with variation."""
        board = chess.Board()
        
        if opening is not None:
            try:
                # DEDUPLICATED: Use centralized opening application logic
                applied_moves = TrainingUtilities.apply_opening_moves(
                    board, opening.moves, opening.continuation_depth
                )
                self.logger.debug(f"Applied {applied_moves} moves from opening {opening.name}")
                        
            except Exception as e:
                self.logger.warning(f"Error applying opening {opening.name}: {e}")
                board = chess.Board()  # Fall back to starting position
        
        return ChessPosition(board)
    
    def _calculate_style_adherence(self, opening: Optional[OpeningTemplate], 
                                 training_examples: List[AlphaZeroTrainingExample]) -> float:
        """Calculate how well the game followed the target style."""
        if not opening:
            return 0.0
        
        opening_moves_played = len(opening.moves)
        total_moves = len(training_examples)
        
        # Simple adherence metric: ratio of opening moves to total moves
        return min(opening_moves_played / max(total_moves, 1), 1.0)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics - DEDUPLICATED.
        
        Uses centralized statistics manager.
        """
        return self.statistics.get_comprehensive_report()
    
    def reset_statistics(self) -> None:
        """Reset all statistics."""
        self.statistics.reset()
        self.logger.info("Statistics reset")


# DEDUPLICATED: Simplified utility functions using centralized testing
def validate_style_specific_generation(style: str, num_test_games: int = 5) -> bool:
    """Validate style generation using centralized test utilities."""
    print(f"Validating {style} style generation with {num_test_games} games...")
    
    # Use centralized test model creation
    from ..training.training_utils import TestUtilities
    test_model = TestUtilities.create_test_model()
    
    # Create self-play generator
    generator = StyleSpecificSelfPlay()
    
    # Generate test games with minimal configuration
    test_config = {
        'mcts_simulations': 50,  # Reduced for testing
        'max_moves': 120,
        'enable_resignation': True,
        'resignation_threshold': -0.9
    }
    
    try:
        training_examples = generator.generate_training_examples(
            model=test_model,
            style=style,
            num_games=num_test_games,
            **test_config
        )
        
        # Use centralized result validation
        validation_result = TestUtilities.validate_training_examples(training_examples)
        
        print(f"Generated {len(training_examples)} training examples")
        print(f"Validation: {validation_result}")
        
        return len(training_examples) > 0
        
    except Exception as e:
        print(f"Validation failed: {e}")
        return False


def test_all_styles() -> None:
    """Test all styles using centralized test framework."""
    from ..training.training_utils import TestUtilities
    
    styles = ['tactical', 'positional', 'dynamic']
    results = {}
    
    for style in styles:
        print(f"\n{'='*50}")
        print(f"Testing {style.upper()} style")
        print(f"{'='*50}")
        
        try:
            success = validate_style_specific_generation(style, num_test_games=2)
            results[style] = success
            print(f"✅ {style} style test: {'PASSED' if success else 'FAILED'}")
        except Exception as e:
            results[style] = False
            print(f"❌ {style} style test FAILED: {e}")
    
    # Use centralized test reporting
    TestUtilities.generate_test_report(results, "Style-Specific Self-Play Tests")


if __name__ == "__main__":
    test_all_styles()