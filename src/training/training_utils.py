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

from collections import defaultdict
import json
import math
import time
import torch
import torch.nn.functional as F
import numpy as np
import chess
import chess.svg  # Add chess.svg import for board image generation
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, asdict  # Add asdict import
from enum import Enum
from pathlib import Path
import logging

# For board image generation
try:
    import cairosvg
    CAIROSVG_AVAILABLE = True
except ImportError:
    CAIROSVG_AVAILABLE = False

try:
    from PIL import Image
    import io
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

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


def tactical_temperature_schedule(move_num: int) -> float:
    """Convenience function for backward compatibility."""
    return TemperatureSchedules.tactical_schedule(move_num)

def filter_decisive_games(training_examples: List[Any]) -> List[Any]:
    """Convenience function for backward compatibility."""
    return TrainingExampleFilters.filter_decisive_games(training_examples)

@dataclass
class GameStats:
    """Statistics for a single game or style."""
    games_played: int = 0
    total_moves: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    resignations: int = 0
    average_length: float = 0.0
    openings_used: Dict[str, int] = None
    
    def __post_init__(self):
        if self.openings_used is None:
            self.openings_used = {}
    
    @property
    def win_rate(self) -> float:
        """Calculate win rate."""
        total_decisive = self.wins + self.losses
        return self.wins / max(total_decisive, 1)
    
    @property
    def draw_rate(self) -> float:
        """Calculate draw rate."""
        return self.draws / max(self.games_played, 1)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class GameStatistics:
    """
    Centralized game statistics tracking and analysis.
    
    Tracks detailed statistics for different playing styles, openings,
    and overall training progress.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
        # Style-specific statistics
        self.style_stats: Dict[str, GameStats] = {
            'tactical': GameStats(),
            'positional': GameStats(),
            'dynamic': GameStats(),
            'standard': GameStats()
        }
        
        # Overall statistics
        self.overall_stats = GameStats()
        
        # Detailed tracking
        self.game_history: List[Dict[str, Any]] = []
        self.opening_performance: Dict[str, GameStats] = defaultdict(GameStats)
        
        # Performance metrics over time
        self.performance_timeline: List[Dict[str, Any]] = []
        
        self.logger.info("GameStatistics initialized")
    
    def record_game(self, style: str, game_length: int, final_result: float,
                   opening_used: Optional[str] = None, resigned: bool = False,
                   additional_data: Optional[Dict[str, Any]] = None) -> None:
        """
        Record statistics for a completed game.
        
        Args:
            style: Playing style used
            game_length: Number of moves in the game
            final_result: Game outcome (-1.0 = loss, 0.0 = draw, 1.0 = win)
            opening_used: Name of opening used
            resigned: Whether the game ended by resignation
            additional_data: Additional metadata to store
        """
        # Update style statistics
        if style in self.style_stats:
            self._update_game_stats(self.style_stats[style], game_length, final_result, 
                                  opening_used, resigned)
        
        # Update overall statistics
        self._update_game_stats(self.overall_stats, game_length, final_result, 
                              opening_used, resigned)
        
        # Update opening-specific statistics
        if opening_used:
            self._update_game_stats(self.opening_performance[opening_used], 
                                  game_length, final_result, opening_used, resigned)
        
        # Record in game history
        game_record = {
            'timestamp': time.time(),
            'style': style,
            'game_length': game_length,
            'final_result': final_result,
            'opening_used': opening_used,
            'resigned': resigned,
            'game_number': len(self.game_history) + 1
        }
        
        if additional_data:
            game_record.update(additional_data)
        
        self.game_history.append(game_record)
        
        # Update performance timeline (every 10 games)
        if len(self.game_history) % 10 == 0:
            self._update_performance_timeline()
        
        self.logger.debug(f"Recorded game: {style} style, {game_length} moves, "
                         f"result: {final_result}, opening: {opening_used}")
    
    def _update_game_stats(self, stats: GameStats, game_length: int, 
                          final_result: float, opening_used: Optional[str],
                          resigned: bool) -> None:
        """Update a GameStats object with new game data."""
        # Update counters
        stats.games_played += 1
        stats.total_moves += game_length
        
        if resigned:
            stats.resignations += 1
        
        # Update win/loss/draw counters
        if final_result > 0.5:
            stats.wins += 1
        elif final_result < -0.5:
            stats.losses += 1
        else:
            stats.draws += 1
        
        # Update average length
        stats.average_length = stats.total_moves / stats.games_played
        
        # Update opening usage
        if opening_used:
            stats.openings_used[opening_used] = stats.openings_used.get(opening_used, 0) + 1
    
    def _update_performance_timeline(self) -> None:
        """Update performance metrics timeline."""
        recent_games = self.game_history[-10:] if len(self.game_history) >= 10 else self.game_history
        
        wins = sum(1 for g in recent_games if g['final_result'] > 0.5)
        draws = sum(1 for g in recent_games if abs(g['final_result']) <= 0.5)
        avg_length = sum(g['game_length'] for g in recent_games) / len(recent_games)
        resignations = sum(1 for g in recent_games if g['resigned'])
        
        timeline_entry = {
            'games_completed': len(self.game_history),
            'recent_win_rate': wins / len(recent_games),
            'recent_draw_rate': draws / len(recent_games),
            'recent_avg_length': avg_length,
            'recent_resignations': resignations,
            'timestamp': time.time()
        }
        
        self.performance_timeline.append(timeline_entry)
    
    def get_style_summary(self, style: str) -> Dict[str, Any]:
        """Get comprehensive summary for a specific style."""
        if style not in self.style_stats:
            return {}
        
        stats = self.style_stats[style]
        
        return {
            'style': style,
            'games_played': stats.games_played,
            'win_rate': stats.win_rate,
            'draw_rate': stats.draw_rate,
            'average_length': stats.average_length,
            'total_moves': stats.total_moves,
            'resignations': stats.resignations,
            'most_used_openings': sorted(stats.openings_used.items(), 
                                       key=lambda x: x[1], reverse=True)[:5],
            'performance_trend': self._calculate_performance_trend(style)
        }
    
    def _calculate_performance_trend(self, style: str) -> str:
        """Calculate performance trend for a style."""
        style_games = [g for g in self.game_history if g['style'] == style]
        
        if len(style_games) < 10:
            return "insufficient_data"
        
        # Compare recent 10 games with previous 10 games
        recent_games = style_games[-10:]
        previous_games = style_games[-20:-10] if len(style_games) >= 20 else []
        
        if not previous_games:
            return "improving"  # Default for early games
        
        recent_win_rate = sum(1 for g in recent_games if g['final_result'] > 0.5) / len(recent_games)
        previous_win_rate = sum(1 for g in previous_games if g['final_result'] > 0.5) / len(previous_games)
        
        diff = recent_win_rate - previous_win_rate
        
        if diff > 0.1:
            return "improving"
        elif diff < -0.1:
            return "declining"
        else:
            return "stable"
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Get comprehensive statistics report."""
        return {
            'overall_statistics': self.overall_stats.to_dict(),
            'style_statistics': {style: stats.to_dict() 
                               for style, stats in self.style_stats.items()},
            'opening_performance': {opening: stats.to_dict() 
                                  for opening, stats in self.opening_performance.items()},
            'performance_timeline': self.performance_timeline[-20:],  # Last 20 entries
            'total_games': len(self.game_history),
            'top_openings': self._get_top_openings(),
            'style_comparison': self._get_style_comparison()
        }
    
    def _get_top_openings(self, limit: int = 10) -> List[Tuple[str, Dict[str, Any]]]:
        """Get top performing openings."""
        opening_summary = []
        
        for opening, stats in self.opening_performance.items():
            if stats.games_played >= 3:  # Minimum games for meaningful statistics
                summary = {
                    'games': stats.games_played,
                    'win_rate': stats.win_rate,
                    'avg_length': stats.average_length,
                    'performance_score': stats.win_rate * stats.games_played  # Weighted by usage
                }
                opening_summary.append((opening, summary))
        
        return sorted(opening_summary, key=lambda x: x[1]['performance_score'], reverse=True)[:limit]
    
    def _get_style_comparison(self) -> Dict[str, Any]:
        """Compare performance across different styles."""
        comparison = {}
        
        for style, stats in self.style_stats.items():
            if stats.games_played > 0:
                comparison[style] = {
                    'win_rate': stats.win_rate,
                    'avg_length': stats.average_length,
                    'games_played': stats.games_played,
                    'efficiency': stats.wins / max(stats.total_moves, 1) * 1000  # Wins per 1000 moves
                }
        
        return comparison
    
    def reset(self) -> None:
        """Reset all statistics."""
        self.style_stats = {style: GameStats() for style in self.style_stats.keys()}
        self.overall_stats = GameStats()
        self.game_history.clear()
        self.opening_performance.clear()
        self.performance_timeline.clear()
        
        self.logger.info("All statistics reset")
    
    def export_to_file(self, filepath: str) -> bool:
        """Export statistics to JSON file."""
        try:
            report = self.get_comprehensive_report()
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"Statistics exported to {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to export statistics: {e}")
            return False


class BoardImageManager:
    """
    Centralized board image saving and management.
    
    Handles saving chess board positions as images with proper organization
    and metadata.
    """
    
    def __init__(self, base_dir: str = "board_images", logger: Optional[logging.Logger] = None):
        self.base_dir = Path(base_dir)
        self.logger = logger or logging.getLogger(__name__)
        
        # Create base directory
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Check availability of image libraries
        self.can_create_png = CAIROSVG_AVAILABLE
        self.can_create_advanced = PIL_AVAILABLE
        
        if not self.can_create_png:
            self.logger.warning("cairosvg not available - only SVG images will be saved")
        
        self.logger.info(f"BoardImageManager initialized with base directory: {self.base_dir}")
    
    def save_board_image(self, board: chess.Board, filepath: str, 
                        title: str = "", size: int = 400) -> bool:
        """
        Save a chess board position as an image.
        
        Args:
            board: Chess board position to save
            filepath: Path where to save the image (without extension)
            title: Optional title to add to the image
            size: Size of the board image in pixels
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Ensure directory exists
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            # Generate SVG of the board
            board_svg = chess.svg.board(
                board=board,
                size=size,
                coordinates=True,
                lastmove=board.peek() if board.move_stack else None
            )
            
            # Save as SVG first (always works)
            svg_path = f"{filepath}.svg"
            with open(svg_path, 'w') as f:
                f.write(board_svg)
            
            # Try to convert to PNG if possible
            if self.can_create_png:
                png_path = f"{filepath}.png"
                try:
                    cairosvg.svg2png(bytestring=board_svg.encode('utf-8'), 
                                   write_to=png_path)
                    self.logger.debug(f"Board saved as PNG: {png_path}")
                    return True
                except Exception as e:
                    self.logger.warning(f"PNG conversion failed: {e}")
            
            self.logger.debug(f"Board saved as SVG: {svg_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save board image: {e}")
            return False
    
    def save_game_images(self, initial_position, final_state, training_examples: List,
                        game_id: str, opening, outcome_str: str, 
                        move_history: Optional[List[Dict]] = None) -> bool:
        """
        Save complete game image set with metadata.
        
        Args:
            initial_position: Starting chess position
            final_state: Final game state
            training_examples: List of training examples
            game_id: Unique game identifier
            opening: Opening template used
            outcome_str: Description of game outcome
            move_history: Optional move history
            
        Returns:
            True if saved successfully
        """
        try:
            # Create game directory
            game_dir = self.base_dir / game_id
            game_dir.mkdir(parents=True, exist_ok=True)
            
            # Save starting position
            opening_name = opening.name if opening else "standard"
            start_path = game_dir / f"start_{opening_name.replace(' ', '_')}"
            start_title = f"Game {game_id} - Start\nOpening: {opening_name}"
            
            self.save_board_image(initial_position.board, str(start_path), start_title)
            
            # Save final position
            end_path = game_dir / f"end_{outcome_str.replace(' ', '_')}"
            end_title = f"Game {game_id} - End\nOutcome: {outcome_str}\nMoves: {len(training_examples)}"
            
            self.save_board_image(final_state.board, str(end_path), end_title)
            
            # Save metadata
            self._save_game_metadata(game_dir, game_id, opening, outcome_str, 
                                   training_examples, initial_position, final_state, move_history)
            
            self.logger.info(f"Game images saved for {game_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save game images for {game_id}: {e}")
            return False
    
    def _save_game_metadata(self, game_dir: Path, game_id: str, opening, 
                           outcome_str: str, training_examples: List,
                           initial_position, final_state, move_history: Optional[List[Dict]]) -> None:
        """Save game metadata as JSON and text files."""
        # JSON metadata
        metadata = {
            "game_id": game_id,
            "opening": {
                "name": opening.name if opening else "standard",
                "eco_code": opening.eco_code if opening else "N/A",
                "moves": opening.moves if opening else []
            },
            "outcome": outcome_str,
            "total_moves": len(training_examples),
            "starting_fen": initial_position.board.fen(),
            "final_fen": final_state.board.fen(),
            "move_history": move_history or [],
            "timestamp": time.time(),
            "metadata": {
                "game_length": len(move_history) - 1 if move_history else 0,
                "opening_moves_played": len(opening.moves) if opening else 0,
                "resignation": "resignation" in outcome_str.lower(),
                "training_examples_count": len(training_examples)
            }
        }
        
        json_file = game_dir / "game_data.json"
        with open(json_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Human-readable summary
        summary_file = game_dir / "game_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Chess Game Summary\n")
            f.write(f"================\n\n")
            f.write(f"Game ID: {game_id}\n")
            f.write(f"Opening: {metadata['opening']['name']}\n")
            f.write(f"ECO Code: {metadata['opening']['eco_code']}\n")
            f.write(f"Outcome: {outcome_str}\n")
            f.write(f"Total Moves: {len(training_examples)}\n")
            f.write(f"Starting FEN: {initial_position.board.fen()}\n")
            f.write(f"Final FEN: {final_state.board.fen()}\n")
            
            if opening and opening.moves:
                f.write(f"\nOpening Moves: {' '.join(opening.moves)}\n")
            
            if move_history:
                f.write(f"\nMove History:\n")
                for i, move_data in enumerate(move_history[1:], 1):  # Skip initial position
                    f.write(f"{i}. {move_data.get('move_san', 'N/A')}\n")


class GameResultAnalyzer:
    """
    Centralized game result analysis and outcome processing.
    
    Provides consistent analysis of game outcomes, training examples,
    and performance metrics.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def analyze_training_examples(self, training_examples: List) -> Dict[str, Any]:
        """
        Analyze a list of training examples for outcome distribution.
        
        Args:
            training_examples: List of AlphaZeroTrainingExample objects
            
        Returns:
            Dictionary with analysis results
        """
        if not training_examples:
            return {
                'total_examples': 0,
                'outcome_distribution': {},
                'outcome_description': 'No examples'
            }
        
        # Count outcomes
        wins = sum(1 for ex in training_examples if ex.outcome > 0.5)
        losses = sum(1 for ex in training_examples if ex.outcome < -0.5)
        draws = sum(1 for ex in training_examples if abs(ex.outcome) <= 0.5)
        
        outcome_distribution = {
            'wins': wins,
            'losses': losses,
            'draws': draws,
            'win_percentage': wins / len(training_examples) * 100,
            'loss_percentage': losses / len(training_examples) * 100,
            'draw_percentage': draws / len(training_examples) * 100
        }
        
        # Determine primary outcome
        if wins > losses and wins > draws:
            primary_outcome = 'win'
        elif losses > wins and losses > draws:
            primary_outcome = 'loss'
        else:
            primary_outcome = 'draw'
        
        return {
            'total_examples': len(training_examples),
            'outcome_distribution': outcome_distribution,
            'primary_outcome': primary_outcome,
            'outcome_description': self._format_outcome_description(outcome_distribution, primary_outcome),
            'is_decisive': draws < len(training_examples) // 2  # More than half non-draw
        }
    
    def _format_outcome_description(self, distribution: Dict[str, Any], primary: str) -> str:
        """Format a human-readable outcome description."""
        if primary == 'win':
            return f"White wins ({distribution['win_percentage']:.1f}% winning positions)"
        elif primary == 'loss':
            return f"Black wins ({distribution['loss_percentage']:.1f}% losing positions)"
        else:
            return f"Draw ({distribution['draw_percentage']:.1f}% drawn positions)"
    
    def analyze_game_outcome(self, training_examples: List, game_resigned: bool, 
                           winner: Optional[int]) -> Dict[str, Any]:
        """
        Analyze the outcome of a complete game.
        
        Args:
            training_examples: List of training examples from the game
            game_resigned: Whether the game ended by resignation
            winner: Winner of the game (1 = white, -1 = black, 0 = draw)
            
        Returns:
            Dictionary with outcome analysis
        """
        base_analysis = self.analyze_training_examples(training_examples)
        
        # Determine outcome description
        if game_resigned:
            if winner == 1:
                outcome_desc = "White wins by resignation"
            elif winner == -1:
                outcome_desc = "Black wins by resignation"
            else:
                outcome_desc = "Draw by resignation"
        else:
            if winner == 1:
                outcome_desc = "White wins"
            elif winner == -1:
                outcome_desc = "Black wins"
            else:
                outcome_desc = "Draw"
        
        return {
            **base_analysis,
            'game_resigned': game_resigned,
            'winner': winner,
            'outcome_description': outcome_desc,
            'resignation_type': self._classify_resignation(training_examples) if game_resigned else None
        }
    
    def _classify_resignation(self, training_examples: List) -> str:
        """Classify the type of resignation based on final positions."""
        if not training_examples:
            return "unknown"
        
        final_outcomes = [ex.outcome for ex in training_examples[-5:]]  # Last 5 positions
        avg_final_outcome = sum(final_outcomes) / len(final_outcomes)
        
        if avg_final_outcome < -0.8:
            return "hopeless_position"
        elif avg_final_outcome > 0.8:
            return "winning_position"  # Shouldn't happen but handle it
        else:
            return "unclear_position"
    
    def calculate_decisive_percentage(self, game_results: List) -> float:
        """
        Calculate the percentage of games that had decisive outcomes.
        
        Args:
            game_results: List of game result objects
            
        Returns:
            Percentage of decisive games (0-100)
        """
        if not game_results:
            return 0.0
        
        decisive_games = 0
        for result in game_results:
            if hasattr(result, 'final_result'):
                if abs(result.final_result) > 0.5:  # Win or loss, not draw
                    decisive_games += 1
            elif hasattr(result, 'training_examples'):
                # Analyze training examples if final_result not available
                analysis = self.analyze_training_examples(result.training_examples)
                if analysis['is_decisive']:
                    decisive_games += 1
        
        return (decisive_games / len(game_results)) * 100


class FileManager:
    """
    Centralized file operations for training data and results.
    
    Handles saving, loading, and organizing training files with
    consistent naming and structure.
    """
    
    def __init__(self, base_dir: str = "training_data", logger: Optional[logging.Logger] = None):
        self.base_dir = Path(base_dir)
        self.logger = logger or logging.getLogger(__name__)
        
        # Create directory structure
        self.models_dir = self.base_dir / "models"
        self.games_dir = self.base_dir / "games"
        self.stats_dir = self.base_dir / "statistics"
        self.logs_dir = self.base_dir / "logs"
        
        for directory in [self.models_dir, self.games_dir, self.stats_dir, self.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"FileManager initialized with base directory: {self.base_dir}")
    
    def save_training_examples(self, examples: List, filename: str, 
                              metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Save training examples to file with metadata.
        
        Args:
            examples: List of training examples
            filename: Name for the output file
            metadata: Optional metadata to save alongside
            
        Returns:
            True if saved successfully
        """
        try:
            filepath = self.games_dir / f"{filename}.pt"
            
            # Prepare data for saving
            save_data = {
                'training_examples': examples,
                'metadata': metadata or {},
                'timestamp': time.time(),
                'count': len(examples)
            }
            
            torch.save(save_data, filepath)
            
            # Also save metadata as JSON for easy inspection
            if metadata:
                json_path = self.games_dir / f"{filename}_metadata.json"
                with open(json_path, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
            
            self.logger.info(f"Saved {len(examples)} training examples to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save training examples: {e}")
            return False
    
    def load_training_examples(self, filename: str) -> Tuple[Optional[List], Optional[Dict]]:
        """
        Load training examples and metadata from file.
        
        Args:
            filename: Name of file to load
            
        Returns:
            Tuple of (examples, metadata) or (None, None) if failed
        """
        try:
            filepath = self.games_dir / f"{filename}.pt"
            
            if not filepath.exists():
                self.logger.warning(f"Training file not found: {filepath}")
                return None, None
            
            data = torch.load(filepath, map_location='cpu')
            
            examples = data.get('training_examples', [])
            metadata = data.get('metadata', {})
            
            self.logger.info(f"Loaded {len(examples)} training examples from {filepath}")
            return examples, metadata
            
        except Exception as e:
            self.logger.error(f"Failed to load training examples: {e}")
            return None, None
    
    def save_model_checkpoint(self, model, optimizer, epoch: int, 
                             loss: float, additional_data: Optional[Dict] = None) -> bool:
        """
        Save model checkpoint with training state.
        
        Args:
            model: PyTorch model to save
            optimizer: Optimizer state
            epoch: Current epoch number
            loss: Current loss value
            additional_data: Additional data to save
            
        Returns:
            True if saved successfully
        """
        try:
            checkpoint_data = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': loss,
                'timestamp': time.time()
            }
            
            if additional_data:
                checkpoint_data.update(additional_data)
            
            # Save with epoch number for versioning
            filepath = self.models_dir / f"checkpoint_epoch_{epoch:04d}.pt"
            torch.save(checkpoint_data, filepath)
            
            # Also save as "latest" for easy loading
            latest_path = self.models_dir / "latest_checkpoint.pt"
            torch.save(checkpoint_data, latest_path)
            
            self.logger.info(f"Saved model checkpoint: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
            return False
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get path to the latest model checkpoint."""
        latest_path = self.models_dir / "latest_checkpoint.pt"
        
        if latest_path.exists():
            return str(latest_path)
        
        # Look for numbered checkpoints
        checkpoint_files = list(self.models_dir.glob("checkpoint_epoch_*.pt"))
        if checkpoint_files:
            latest = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
            return str(latest)
        
        return None
    
    def organize_files_by_date(self) -> Dict[str, List[str]]:
        """Organize files by creation date for cleanup."""
        file_organization = defaultdict(list)
        
        for directory in [self.games_dir, self.models_dir, self.stats_dir]:
            for filepath in directory.rglob("*"):
                if filepath.is_file():
                    # Get file creation date
                    creation_time = filepath.stat().st_ctime
                    date_str = time.strftime("%Y-%m-%d", time.localtime(creation_time))
                    file_organization[date_str].append(str(filepath))
        
        return dict(file_organization)
    
    def cleanup_old_files(self, days_to_keep: int = 30) -> int:
        """
        Clean up files older than specified days.
        
        Args:
            days_to_keep: Number of days of files to keep
            
        Returns:
            Number of files deleted
        """
        cutoff_time = time.time() - (days_to_keep * 24 * 60 * 60)
        deleted_count = 0
        
        for directory in [self.games_dir, self.stats_dir]:
            for filepath in directory.rglob("*"):
                if filepath.is_file() and filepath.stat().st_ctime < cutoff_time:
                    try:
                        filepath.unlink()
                        deleted_count += 1
                        self.logger.debug(f"Deleted old file: {filepath}")
                    except Exception as e:
                        self.logger.warning(f"Failed to delete {filepath}: {e}")
        
        self.logger.info(f"Cleanup completed: {deleted_count} old files deleted")
        return deleted_count


class TestUtilities:
    """
    Centralized testing utilities for AlphaZero training components.
    
    Provides consistent test models, validation functions, and test reporting
    for all training components.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    @staticmethod
    def create_test_model():
        """Create a small test model for validation purposes."""
        from ..core.alphazero_net import AlphaZeroNet
        
        # Create a smaller model for testing
        test_model = AlphaZeroNet(
            board_size=8,
            action_size=4672,
            num_filters=64,      # Reduced from 256
            num_res_blocks=4,    # Reduced from 19
            input_channels=119
        )
        
        # Initialize with random weights
        test_model.eval()
        return test_model
    
    @staticmethod
    def create_dummy_training_examples(count: int = 10) -> List:
        """Create dummy training examples for testing."""
        import torch
        from ..core.alphazero_mcts import AlphaZeroTrainingExample
        
        examples = []
        for i in range(count):
            # Create dummy state tensor
            state_tensor = torch.randn(119, 8, 8)
            
            # Create dummy action probabilities
            action_probs = {f"move_{j}": 1.0/10 for j in range(10)}
            
            # Alternate between win/loss/draw outcomes
            if i % 3 == 0:
                outcome = 1.0  # Win
            elif i % 3 == 1:
                outcome = -1.0  # Loss
            else:
                outcome = 0.0  # Draw
            
            current_player = 1 if i % 2 == 0 else -1
            
            example = AlphaZeroTrainingExample(
                state_tensor=state_tensor,
                action_probs=action_probs,
                outcome=outcome,
                current_player=current_player
            )
            examples.append(example)
        
        return examples
    
    @staticmethod
    def validate_training_examples(examples: List) -> Dict[str, Any]:
        """
        Validate a list of training examples for consistency.
        
        Args:
            examples: List of training examples to validate
            
        Returns:
            Dictionary with validation results
        """
        if not examples:
            return {
                'valid': False,
                'error': 'No examples provided',
                'count': 0
            }
        
        validation_results = {
            'valid': True,
            'count': len(examples),
            'errors': [],
            'warnings': []
        }
        
        for i, example in enumerate(examples):
            # Check state tensor shape
            if not hasattr(example, 'state_tensor'):
                validation_results['errors'].append(f"Example {i}: Missing state_tensor")
                continue
            
            if example.state_tensor.shape != (119, 8, 8):
                validation_results['errors'].append(
                    f"Example {i}: Invalid state tensor shape {example.state_tensor.shape}, expected (119, 8, 8)"
                )
            
            # Check action probabilities
            if not hasattr(example, 'action_probs') or not example.action_probs:
                validation_results['errors'].append(f"Example {i}: Missing or empty action_probs")
            
            # Check outcome range
            if not hasattr(example, 'outcome'):
                validation_results['errors'].append(f"Example {i}: Missing outcome")
            elif not (-1.0 <= example.outcome <= 1.0):
                validation_results['warnings'].append(
                    f"Example {i}: Outcome {example.outcome} outside expected range [-1, 1]"
                )
            
            # Check current player
            if not hasattr(example, 'current_player'):
                validation_results['errors'].append(f"Example {i}: Missing current_player")
            elif example.current_player not in [-1, 1]:
                validation_results['errors'].append(
                    f"Example {i}: Invalid current_player {example.current_player}, expected -1 or 1"
                )
        
        validation_results['valid'] = len(validation_results['errors']) == 0
        
        return validation_results
    
    @staticmethod
    def validate_model_output(model, input_tensor: torch.Tensor) -> Dict[str, Any]:
        """
        Validate model output shapes and ranges.
        
        Args:
            model: Model to test
            input_tensor: Input tensor for testing
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'valid': True,
            'errors': [],
            'policy_shape': None,
            'value_shape': None,
            'policy_range': None,
            'value_range': None
        }
        
        try:
            with torch.no_grad():
                policy_logits, value = model(input_tensor)
            
            # Check policy output
            validation_results['policy_shape'] = policy_logits.shape
            if policy_logits.shape[-1] != 4672:
                validation_results['errors'].append(
                    f"Policy output size {policy_logits.shape[-1]}, expected 4672"
                )
            
            # Check value output
            validation_results['value_shape'] = value.shape
            if len(value.shape) != 2 or value.shape[-1] != 1:
                validation_results['errors'].append(
                    f"Value output shape {value.shape}, expected (batch_size, 1)"
                )
            
            # Check ranges after applying activations
            policy_probs = torch.softmax(policy_logits, dim=-1)
            value_activated = torch.tanh(value)
            
            validation_results['policy_range'] = (policy_probs.min().item(), policy_probs.max().item())
            validation_results['value_range'] = (value_activated.min().item(), value_activated.max().item())
            
            # Validate probability sum
            prob_sum = policy_probs.sum(dim=-1).item()
            if abs(prob_sum - 1.0) > 1e-6:
                validation_results['errors'].append(
                    f"Policy probabilities sum to {prob_sum}, expected ~1.0"
                )
            
        except Exception as e:
            validation_results['errors'].append(f"Model forward pass failed: {e}")
        
        validation_results['valid'] = len(validation_results['errors']) == 0
        return validation_results
    
    @staticmethod
    def run_model_inference_test(model, num_tests: int = 5) -> Dict[str, Any]:
        """
        Run multiple inference tests to check model consistency.
        
        Args:
            model: Model to test
            num_tests: Number of test runs
            
        Returns:
            Dictionary with test results
        """
        test_results = {
            'passed': True,
            'num_tests': num_tests,
            'inference_times': [],
            'errors': []
        }
        
        for i in range(num_tests):
            try:
                # Create random input
                input_tensor = torch.randn(1, 119, 8, 8)
                
                # Time the inference
                start_time = time.time()
                with torch.no_grad():
                    policy_logits, value = model(input_tensor)
                inference_time = time.time() - start_time
                
                test_results['inference_times'].append(inference_time)
                
                # Validate output
                validation = TestUtilities.validate_model_output(model, input_tensor)
                if not validation['valid']:
                    test_results['errors'].extend(validation['errors'])
                
            except Exception as e:
                test_results['errors'].append(f"Test {i} failed: {e}")
        
        test_results['passed'] = len(test_results['errors']) == 0
        test_results['avg_inference_time'] = np.mean(test_results['inference_times']) if test_results['inference_times'] else 0
        
        return test_results
    
    @staticmethod
    def generate_test_report(test_results: Dict[str, Any], test_name: str = "AlphaZero Tests") -> str:
        """
        Generate a formatted test report.
        
        Args:
            test_results: Dictionary of test results
            test_name: Name of the test suite
            
        Returns:
            Formatted test report string
        """
        report_lines = [
            f"\n{'='*60}",
            f"TEST REPORT: {test_name}",
            f"{'='*60}",
            f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]
        
        total_tests = 0
        passed_tests = 0
        
        for test_category, result in test_results.items():
            total_tests += 1
            
            if isinstance(result, bool):
                status = "PASS" if result else "FAIL"
                if result:
                    passed_tests += 1
                report_lines.append(f"{test_category}: {status}")
            
            elif isinstance(result, dict):
                if 'passed' in result or 'valid' in result:
                    passed = result.get('passed', result.get('valid', False))
                    status = "PASS" if passed else "FAIL"
                    if passed:
                        passed_tests += 1
                    
                    report_lines.append(f"{test_category}: {status}")
                    
                    # Add details for failed tests
                    if not passed and 'errors' in result:
                        for error in result['errors'][:3]:  # Show first 3 errors
                            report_lines.append(f"  - {error}")
                        if len(result['errors']) > 3:
                            report_lines.append(f"  - ... and {len(result['errors']) - 3} more errors")
        
        # Summary
        report_lines.extend([
            "",
            f"SUMMARY: {passed_tests}/{total_tests} tests passed",
            f"Success rate: {passed_tests/max(total_tests,1)*100:.1f}%",
            ""
        ])
        
        if passed_tests == total_tests:
            report_lines.append("🎉 ALL TESTS PASSED!")
        else:
            report_lines.append(f"❌ {total_tests - passed_tests} test(s) failed")
        
        report_lines.append(f"{'='*60}\n")
        
        return "\n".join(report_lines)
    
    @staticmethod
    def benchmark_training_pipeline(model, num_examples: int = 100) -> Dict[str, Any]:
        """
        Benchmark the training pipeline performance.
        
        Args:
            model: Model to benchmark
            num_examples: Number of training examples to use
            
        Returns:
            Dictionary with benchmark results
        """
        benchmark_results = {
            'model_parameters': sum(p.numel() for p in model.parameters()),
            'model_size_mb': sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024,
            'forward_pass_times': [],
            'memory_usage': {},
            'errors': []
        }
        
        try:
            # Create test data
            examples = TestUtilities.create_dummy_training_examples(num_examples)
            
            # Benchmark forward passes
            for i in range(min(10, num_examples)):
                input_tensor = examples[i].state_tensor.unsqueeze(0)
                
                start_time = time.time()
                with torch.no_grad():
                    policy_logits, value = model(input_tensor)
                forward_time = time.time() - start_time
                
                benchmark_results['forward_pass_times'].append(forward_time)
            
            # Calculate statistics
            times = benchmark_results['forward_pass_times']
            benchmark_results['avg_forward_time'] = np.mean(times)
            benchmark_results['min_forward_time'] = np.min(times)
            benchmark_results['max_forward_time'] = np.max(times)
            benchmark_results['throughput_examples_per_sec'] = 1.0 / np.mean(times) if times else 0
            
        except Exception as e:
            benchmark_results['errors'].append(f"Benchmark failed: {e}")
        
        return benchmark_results


# Utility functions for easy access to centralized functionality
def apply_opening_moves(board: chess.Board, moves: List[str], max_depth: int) -> int:
    """
    Apply opening moves to a chess board with variation.
    
    This function is part of TrainingUtilities but defined here for easy import.
    """
    import random
    
    if not moves:
        return 0
    
    # Sometimes play partial opening for variation
    if max_depth > 4:
        depth = random.randint(max(4, max_depth - 3), max_depth)
    else:
        depth = max_depth
    
    applied_moves = 0
    for i in range(min(depth, len(moves))):
        try:
            move_san = moves[i]
            move = board.parse_san(move_san)
            if move in board.legal_moves:
                board.push(move)
                applied_moves += 1
            else:
                break
        except Exception:
            break
    
    return applied_moves

