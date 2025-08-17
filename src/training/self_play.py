"""
Style-Specific Self-Play for AlphaZero Training - DEDUPLICATED VERSION

This module provides self-play game generation with style-specific opening selection
using the ECO opening database. All shared utilities have been moved to training_utils.py
to eliminate duplication.

Author: Francesco Finucci
"""

import torch
import torch.nn.functional as F
import chess
import chess.svg
import numpy as np
import logging
import time
import os
import json
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
import random

# For board image generation
try:
    import cairosvg
    CAIROSVG_AVAILABLE = True
except ImportError:
    CAIROSVG_AVAILABLE = False
    print("⚠️  cairosvg not available - board images will be saved as SVG only")

try:
    from PIL import Image
    import io
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("⚠️  PIL not available - board images will be saved as SVG only")

from ..core.chess_engine import ChessPosition
from ..core.game_utils import ChessGameState
from ..core.alphazero_net import AlphaZeroNet
from ..core.alphazero_mcts import AlphaZeroMCTS, AlphaZeroTrainingExample, generate_self_play_game
from ..data.openings.openings import ECO_OpeningDatabase, OpeningTemplate

# DEDUPLICATED: Import centralized utilities instead of local implementations
from ..training.training_utils import (
    TrainingUtilities,
    TemperatureSchedules, 
    TrainingExampleFilters,
    GameOutcomeAnalyzer
)


def save_board_image(board: chess.Board, filepath: str, title: str = "") -> bool:
    """
    Save a chess board position as an image.
    
    Args:
        board: Chess board position to save
        filepath: Path where to save the image (without extension)
        title: Optional title to add to the image
        
    Returns:
        True if saved successfully, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Generate SVG of the board
        board_svg = chess.svg.board(
            board=board,
            size=400,
            coordinates=True,
            lastmove=board.peek() if board.move_stack else None
        )
        
        # Save as SVG first (always works)
        svg_path = f"{filepath}.svg"
        with open(svg_path, 'w') as f:
            f.write(board_svg)
        
        # Try to convert to PNG if libraries are available
        png_path = f"{filepath}.png"
        
        if CAIROSVG_AVAILABLE:
            # Method 1: Use cairosvg directly
            try:
                cairosvg.svg2png(bytestring=board_svg.encode('utf-8'), 
                               write_to=png_path)
                return True
            except Exception as e:
                print(f"⚠️  cairosvg conversion failed: {e}")
        
        if PIL_AVAILABLE:
            # Method 2: Use PIL with SVG (requires additional setup)
            try:
                # This requires rsvg or similar - might not work without additional setup
                pass
            except Exception:
                pass
        
        # If PNG conversion failed, at least we have SVG
        print(f"📷 Board saved as SVG: {svg_path}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to save board image: {e}")
        return False


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
    Generates self-play training data with style-specific opening selection - DEDUPLICATED VERSION.
    
    Uses the ECO opening database to force specific playing styles during
    the opening phase, then allows AlphaZero MCTS to take over naturally.
    All shared utilities have been moved to training_utils.py.
    """
    
    def __init__(self, device: str = 'cuda', logger: logging.Logger = None, 
                 save_board_images: bool = False, board_images_dir: str = "board_images"):
        """
        Initialize style-specific self-play generator.
        
        Args:
            device: PyTorch device for neural network inference
            logger: Logger instance to use (if None, creates its own)
            save_board_images: Whether to save starting and ending board positions as images
            board_images_dir: Directory to save board images
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Initialize ECO opening database
        self.opening_db = ECO_OpeningDatabase()
        
        # Board image saving configuration
        self.save_board_images = save_board_images
        self.board_images_dir = board_images_dir
        
        if self.save_board_images:
            os.makedirs(self.board_images_dir, exist_ok=True)
            if logger:
                logger.info(f"Board images will be saved to: {self.board_images_dir}")
        
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
    
    def generate_training_examples(self, model, num_games: int, style: str = 'standard', 
                                 mcts_simulations: int = 100, temperature_moves: int = 30,
                                 save_board_images: bool = False, board_images_dir: Optional[str] = None,
                                 dirichlet_alpha: float = 0.3, enable_resignation: bool = True,
                                 resignation_threshold: float = -0.9, max_moves: int = 150) -> List[AlphaZeroTrainingExample]:
        """
        Generate training examples through style-specific self-play - DEDUPLICATED VERSION.
        
        Args:
            model: AlphaZero neural network for move evaluation
            style: Target playing style ('tactical', 'positional', 'dynamic')
            num_games: Number of self-play games to generate
            mcts_simulations: Number of MCTS simulations per move
            dirichlet_alpha: Dirichlet noise parameter for exploration (now handled centrally)
            temperature_moves: Number of moves to use temperature > 0
            save_board_images: Whether to save starting and ending board positions as images
            board_images_dir: Directory to save board images
            
        Returns:
            List of training examples from all generated games
        """
        if style not in ['tactical', 'positional', 'dynamic']:
            raise ValueError(f"Unknown style: {style}. Must be 'tactical', 'positional', or 'dynamic'")
        
        self.logger.info(f"Generating {num_games} {style} style games with {mcts_simulations} MCTS sims")
        
        # Create MCTS engine with config parameters for proper resignation handling
        self.logger.info(f"Creating MCTS engine for {style} style games...")
        mcts_engine = AlphaZeroMCTS(
            model, 
            c_puct=2.0,  # Increased exploration
            device=self.device, 
            resignation_threshold=resignation_threshold,  # Use config parameter
            logger=self.logger
        )
        self.logger.info(f"MCTS engine created successfully")
        
        all_training_examples = []
        successful_games = 0
        decisive_games = 0  # Track games with actual winners
        
        for game_idx in range(num_games):
            try:
                self.logger.info(f"Starting game {game_idx + 1}/{num_games} ({style} style)")
                # Generate single self-play game with style-specific opening
                game_result = self._generate_single_game(
                    mcts_engine=mcts_engine,
                    style=style,
                    mcts_simulations=mcts_simulations,
                    temperature_moves=temperature_moves,
                    game_id=f"{style}_{game_idx}",
                    save_board_images=save_board_images,
                    board_images_dir=board_images_dir,
                    dirichlet_alpha=dirichlet_alpha,
                    enable_resignation=enable_resignation,
                    max_moves=max_moves
                )
                
                if game_result and game_result.training_examples:
                    # DEDUPLICATED: Use centralized filtering
                    filtered_examples = TrainingExampleFilters.filter_decisive_games(game_result.training_examples)
                    
                    if filtered_examples:
                        all_training_examples.extend(filtered_examples)
                        decisive_games += 1
                        self.logger.info(f"Game {game_idx + 1} completed successfully - DECISIVE game with {len(filtered_examples)} examples, {game_result.game_length} moves")
                    else:
                        self.logger.info(f"Game {game_idx + 1} was a draw - filtered out from training data")
                    
                    successful_games += 1
                    
                    # Update statistics
                    self.style_stats[style]['games'] += 1
                    current_avg = self.style_stats[style]['avg_length']
                    games_count = self.style_stats[style]['games']
                    new_avg = ((current_avg * (games_count - 1)) + game_result.game_length) / games_count
                    self.style_stats[style]['avg_length'] = new_avg
                    
                    if game_result.opening_used:
                        self.openings_used[game_result.opening_used] = self.openings_used.get(game_result.opening_used, 0) + 1
                else:
                    self.logger.warning(f"Game {game_idx + 1} failed to generate valid training examples")
                
            except Exception as e:
                self.logger.warning(f"Failed to generate game {game_idx} for style {style}: {e}")
                import traceback
                self.logger.warning(f"Exception details: {traceback.format_exc()}")
                continue
        
        self.games_generated += successful_games
        
        # Enhanced logging with decisive game statistics
        self.logger.info(f"Generated {successful_games}/{num_games} successful {style} games")
        self.logger.info(f"Decisive games: {decisive_games}/{successful_games} ({decisive_games/max(successful_games,1)*100:.1f}%)")
        self.logger.info(f"Total training examples: {len(all_training_examples)}")
        
        if decisive_games == 0:
            self.logger.warning("⚠️  NO DECISIVE GAMES GENERATED - All games were draws!")
            self.logger.warning("Consider adjusting MCTS parameters or resignation thresholds")
        
        return all_training_examples
    
    def _generate_single_game(self, mcts_engine: AlphaZeroMCTS, style: str,
                             mcts_simulations: int, temperature_moves: int, 
                             game_id: str, save_board_images: bool = False,
                             board_images_dir: str = "board_images",
                             dirichlet_alpha: float = 0.3, enable_resignation: bool = True,
                             max_moves: int = 150) -> Optional[SelfPlayGameResult]:
        """
        Generate a single self-play game using the enhanced MCTS implementation - DEDUPLICATED VERSION.
        
        Args:
            mcts_engine: MCTS engine for move selection
            style: Target playing style
            mcts_simulations: MCTS simulations per move
            temperature_moves: Moves with temperature > 0
            game_id: Unique identifier for this game
            
        Returns:
            SelfPlayGameResult with training examples and metadata
        """
        # Select style-specific opening
        opening = self._select_opening_for_style(style)
        
        # Create initial position from opening
        initial_position = self._create_position_from_opening(opening)
        
        try:
            # DEDUPLICATED: Use centralized self-play generation with style-specific temperature
            training_examples, final_state, move_history, game_resigned, winner = generate_self_play_game(
                chess_engine=mcts_engine,
                initial_state=initial_position,
                num_simulations=mcts_simulations,
                temperature_schedule=None,  # Will use style-based schedule
                max_moves=max_moves,  # Use config parameter
                enable_resignation=enable_resignation,  # Use config parameter
                filter_draws=False,  # Don't filter here, we'll filter in the calling function
                style=style,  # Pass style for temperature schedule selection
                dirichlet_alpha=dirichlet_alpha  # Pass through the alpha parameter
            )
            
            if not training_examples:
                self.logger.warning(f"Game {game_id} generated no training examples")
                return None
            
            # Calculate final reward from the last example
            final_reward = training_examples[-1].outcome if training_examples else 0.0
            
            # Enhanced outcome logging with resignation info
            if game_resigned:
                if winner == 1:
                    outcome_str = "White wins by resignation"
                elif winner == -1:
                    outcome_str = "Black wins by resignation"
                else:
                    outcome_str = "Draw by resignation"  # Shouldn't happen but handle gracefully
            else:
                if winner == 1:
                    outcome_str = "White wins"
                elif winner == -1:
                    outcome_str = "Black wins"
                else:
                    outcome_str = "Draw"
            
            # Log game outcome with additional statistics
            outcome_distribution = {}
            for ex in training_examples:
                outcome_key = "win" if ex.outcome > 0.5 else "loss" if ex.outcome < -0.5 else "draw"
                outcome_distribution[outcome_key] = outcome_distribution.get(outcome_key, 0) + 1
            
            self.logger.info(f"Game {game_id}: {outcome_str} after {len(training_examples)} moves")
            self.logger.info(f"Training examples distribution: {outcome_distribution}")
            
            # Save board images if enabled
            if save_board_images:
                self._save_game_board_images(initial_position, final_state, training_examples, game_id, opening, outcome_str, board_images_dir, move_history)
            
            # Calculate style adherence (simplified - based on opening usage)
            opening_moves_played = len(opening.moves) if opening else 0
            style_adherence = min(opening_moves_played / max(len(training_examples), 1), 1.0) if opening else 0.0
            
            return SelfPlayGameResult(
                training_examples=training_examples,
                game_length=len(training_examples),
                final_result=final_reward,
                opening_used=opening.name if opening else None,
                style_adherence=style_adherence
            )
            
        except Exception as e:
            self.logger.error(f"Error generating game {game_id}: {e}")
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            return None
    
    def _save_game_board_images(self, initial_position: ChessPosition, 
                               final_state: ChessGameState,
                               training_examples: List[AlphaZeroTrainingExample],
                               game_id: str, opening: Optional[OpeningTemplate], 
                               outcome_str: str, board_images_dir: str = "board_images",
                               move_history: List[dict] = None) -> None:
        """
        Save starting and ending board positions as images, plus complete move history as JSON.
        """
        try:
            # Create a subdirectory for this game
            game_dir = os.path.join(board_images_dir, game_id)
            os.makedirs(game_dir, exist_ok=True)
            
            # Save starting position
            opening_name = opening.name if opening else "random"
            start_filename = os.path.join(game_dir, f"start_{opening_name}")
            start_title = f"Game {game_id} - Start Position\nOpening: {opening_name}"
            
            if save_board_image(initial_position.board, start_filename, start_title):
                self.logger.debug(f"Saved starting position for game {game_id}")
            
            # Save final position using the actual final state
            end_filename = os.path.join(game_dir, f"end_{outcome_str.replace(' ', '_')}")
            end_title = f"Game {game_id} - End Position\nOutcome: {outcome_str}\nMoves: {len(training_examples)}"
            
            if save_board_image(final_state.board, end_filename, end_title):
                self.logger.debug(f"Saved ending position for game {game_id}")
            
            # Save complete move history as JSON
            if move_history:
                game_data = {
                    "game_id": game_id,
                    "opening": {
                        "name": opening_name,
                        "moves": opening.moves if opening else []
                    },
                    "outcome": outcome_str,
                    "total_moves": len(training_examples),
                    "starting_fen": initial_position.board.fen(),
                    "final_fen": final_state.board.fen(),
                    "move_history": move_history,
                    "metadata": {
                        "game_length": len(move_history) - 1,  # -1 because first entry is starting position
                        "opening_moves_played": len(opening.moves) if opening else 0,
                        "resignation": "resignation" in outcome_str.lower()
                    }
                }
                
                json_file = os.path.join(game_dir, "game_moves.json")
                with open(json_file, 'w') as f:
                    json.dump(game_data, f, indent=2)
                self.logger.debug(f"Saved move history JSON for game {game_id}")
            
            # Create a summary file with game info
            summary_file = os.path.join(game_dir, "game_info.txt")
            with open(summary_file, 'w') as f:
                f.write(f"Game ID: {game_id}\n")
                f.write(f"Opening: {opening_name}\n")
                f.write(f"Outcome: {outcome_str}\n")
                f.write(f"Total Moves: {len(training_examples)}\n")
                f.write(f"Starting FEN: {initial_position.board.fen()}\n")
                f.write(f"Final FEN: {final_state.board.fen()}\n")
                if opening and opening.moves:
                    f.write(f"Opening Moves: {' '.join(opening.moves)}\n")
            
            self.logger.info(f"📷 Saved board images for game {game_id} to {game_dir}")
        
        except Exception as e:
            self.logger.error(f"Failed to save board images for game {game_id}: {e}")
    
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
        temperature_moves=10,
        enable_resignation=True,  # Default for testing
        resignation_threshold=-0.9,  # Default for testing
        max_moves=120  # Shorter for testing
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