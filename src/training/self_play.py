"""
Style-Specific Self-Play for AlphaZero Training - FULLY DEDUPLICATED VERSION

This module provides self-play game generation with complete deduplication.
All shared utilities are in training_utils.py, including statistics, analysis,
and file operations.

Author: Francesco Finucci
"""

import torch
import chess
import logging
import time
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

from ..core.chess_engine import ChessPosition
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
    FileManager,
    apply_opening_moves              # NEW: Centralized file operations
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
        resignation_threshold = config.get('resignation_threshold', -0.9)
        resignation_centipawns = config.get('resignation_centipawns', -700)
        c_puct = config.get('c_puct', 2.0)
        
        self.logger.info(f"Generating {num_games} {style} style games with {mcts_simulations} MCTS sims")
        #self.logger.info(f"Anti-draw measures: resignation_threshold={resignation_threshold}, centipawns={resignation_centipawns}")
        
        # Create MCTS engine
        mcts_engine = self._create_mcts_engine(
            model=model,
            resignation_threshold=resignation_threshold,
            resignation_centipawns=resignation_centipawns,
            c_puct=c_puct
        )
        
        all_training_examples = []
        game_results = []
        
        # Generate games
        for game_idx in range(num_games):
            try:
                self.logger.info("---------------------------------------------------------")
                self.logger.info(f"Starting game {game_idx + 1}/{num_games} ({style} style)")
                
                game_result = self._generate_single_game(
                    mcts_engine=mcts_engine,
                    style=style,
                    game_id=f"{style}_{game_idx}",
                    config=config,
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
        
        # DEDUPLICATED: Use centralized filtering with new anti-draw measures
        filter_draws = config.get('filter_draws', True)
        minimum_decisive_ratio = config.get('minimum_decisive_ratio', 0.75)
        
        if filter_draws:
            # Apply aggressive draw filtering
            filtered_examples = TrainingExampleFilters.filter_decisive_games(all_training_examples)
            decisive_percentage = self.result_analyzer.calculate_decisive_percentage(game_results)
            
            # Check if we meet minimum decisive ratio
            if decisive_percentage < minimum_decisive_ratio * 100:
                self.logger.warning(f"⚠️  Low decisive ratio: {decisive_percentage:.1f}% < {minimum_decisive_ratio*100:.1f}% target")
                self.logger.warning("Consider adjusting resignation thresholds or MCTS parameters")
            
            self.logger.info(f"Draw filtering ENABLED - Filtered to {len(filtered_examples)}/{len(all_training_examples)} examples")
        else:
            # Include all examples (draws allowed)
            filtered_examples = all_training_examples
            decisive_percentage = self.result_analyzer.calculate_decisive_percentage(game_results) if game_results else 0
            self.logger.info(f"Draw filtering DISABLED - Using all {len(filtered_examples)} examples")
        
        final_stats = self.statistics.get_style_summary(style)
        
        self.logger.info(f"Generated {len(game_results)}/{num_games} successful {style} games")
        self.logger.info(f"Decisive games: {decisive_percentage:.1f}%")
        self.logger.info(f"Final training examples: {len(filtered_examples)}")
        
        if decisive_percentage == 0:
            self.logger.warning("⚠️  NO DECISIVE GAMES GENERATED - All games were draws!")
        
        # Use FileManager to save training examples and metadata
        if filtered_examples:
            timestamp = int(time.time())
            filename = f"{style}_selfplay_{timestamp}_{len(filtered_examples)}examples"
            
            # Prepare comprehensive metadata
            metadata = {
                'style': style,
                'num_games_requested': num_games,
                'num_games_generated': len(game_results),
                'num_training_examples': len(filtered_examples),
                'decisive_percentage': decisive_percentage,
                'config': config,
                'final_stats': final_stats,
                'generation_timestamp': timestamp,
                'filter_draws_enabled': filter_draws,
                'minimum_decisive_ratio': minimum_decisive_ratio
            }
            
            # Save using FileManager
            save_success = self.file_manager.save_training_examples(
                examples=filtered_examples,
                filename=filename,
                metadata=metadata
            )
            
            if save_success:
                self.logger.info(f"💾 Saved training examples to disk: {filename}")
            else:
                self.logger.warning("❌ Failed to save training examples to disk")
        
        return filtered_examples

    def _create_mcts_engine(self, model, resignation_threshold: float, c_puct: float = 2.0, resignation_centipawns: int = -500) -> AlphaZeroMCTS:
        """Create MCTS engine with proper configuration."""
        return AlphaZeroMCTS(
            model, 
            c_puct=c_puct,
            device=self.device, 
            resignation_threshold=resignation_threshold,
            resignation_centipawns=resignation_centipawns,
            logger=self.logger
        )
    
    def _generate_single_game(
            self, 
            mcts_engine: AlphaZeroMCTS, 
            style: str,
            game_id: str, 
            config: dict
        ) -> Optional[SelfPlayGameResult]:
        """
        Generate a single self-play game - FULLY DEDUPLICATED.
        
        All game logic is handled by centralized generate_self_play_game().
        """
        # DEDUPLICATED: Use centralized opening selection
        opening = self._select_opening_for_style(style)
        initial_position = self._create_position_from_opening(opening)
        
        try:
            self.logger.info(f"Generating game {game_id} with opening {opening.name if opening else 'None'}...")
            # DEDUPLICATED: Use centralized self-play generation
            training_examples, final_state, move_history, game_resigned, winner = generate_self_play_game(
                chess_engine=mcts_engine,
                initial_state=initial_position,
                num_simulations=config.get('mcts_simulations', 100),
                temperature_schedule=None,  # Let the function create the schedule based on style
                max_moves=config.get('max_moves', 150),
                enable_resignation=config.get('enable_resignation', True),
                filter_draws=False,  # Handle filtering at higher level
                style=style,
                dirichlet_alpha=config.get('dirichlet_alpha', 0.3)
            )
            self.logger.info(f"Game {game_id} generated with {len(training_examples)} examples")
            
            if not training_examples:
                self.logger.warning(f"Game {game_id} generated no training examples")
                return None
            
            # DEDUPLICATED: Use centralized result analysis
            self.logger.info(f"Analyzing game {game_id}...")
            outcome_info = self.result_analyzer.analyze_game_outcome(
                training_examples, game_resigned, winner
            )
            self.logger.info(f"Game {game_id} analysis complete")

            # DEDUPLICATED: Use centralized image saving
            self.logger.info(f"Saving images for game {game_id}...")
            if config.get('save_board_images', True):
                # Create image manager on-demand if needed
                if not hasattr(self, '_temp_image_manager') or self._temp_image_manager is None:
                    board_images_dir = config.get('board_images_dir', 'board_images')
                    self._temp_image_manager = BoardImageManager(board_images_dir, logger=self.logger)
                    self.logger.info(f"Created temporary BoardImageManager for image saving in {board_images_dir}")
                
                # Use either the permanent or temporary image manager
                image_manager = self.image_manager or self._temp_image_manager
                
                image_manager.save_game_images(
                    initial_position=initial_position,
                    final_state=final_state,
                    training_examples=training_examples,
                    game_id=game_id,
                    opening=opening,    
                    outcome_str=outcome_info['outcome_description'],
                    move_history=move_history
                )
                self.logger.info(f"Game {game_id} images saved successfully")
            else:
                self.logger.debug(f"Image saving disabled for game {game_id}")
            self.logger.info(f"Game {game_id} images processing complete")

            # DEDUPLICATED: Use centralized style adherence calculation
            self.logger.info(f"Calculating style adherence for game {game_id}...")
            style_adherence = self._calculate_style_adherence(opening, training_examples)
            self.logger.info(f"Game {game_id} style adherence: {style_adherence:.2f}")

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
                applied_moves = apply_opening_moves(
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

    def load_training_examples(self, filename: str) -> Optional[List[AlphaZeroTrainingExample]]:
        """
        Load previously saved training examples using FileManager.
        
        Args:
            filename: Name of the file to load (without .pt extension)
            
        Returns:
            List of training examples if successful, None if failed
        """
        examples, metadata = self.file_manager.load_training_examples(filename)
        
        if examples is not None and metadata is not None:
            self.logger.info(f"📂 Loaded {len(examples)} training examples from {filename}")
            self.logger.info(f"Original metadata: {metadata.get('style', 'unknown')} style, "
                           f"{metadata.get('decisive_percentage', 'unknown')}% decisive games")
            return examples
        else:
            self.logger.warning(f"❌ Failed to load training examples from {filename}")
            return None
    
    def get_saved_files_info(self) -> Dict[str, List[str]]:
        """
        Get information about saved training files organized by date.
        
        Returns:
            Dictionary with dates as keys and file lists as values
        """
        return self.file_manager.organize_files_by_date()
    
    def cleanup_old_training_data(self, days_to_keep: int = 30) -> int:
        """
        Clean up old training files using FileManager.
        
        Args:
            days_to_keep: Number of days worth of files to keep
            
        Returns:
            Number of files cleaned up
        """
        files_cleaned = self.file_manager.cleanup_old_files(days_to_keep)
        self.logger.info(f"🧹 Cleaned up {files_cleaned} old training files (kept {days_to_keep} days)")
        return files_cleaned
    
    def get_training_data_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of all saved training data.
        
        Returns:
            Dictionary with summary information about saved files
        """
        files_by_date = self.file_manager.organize_files_by_date()
        
        summary = {
            'total_days_with_data': len(files_by_date),
            'files_by_date': files_by_date,
            'recent_files': [],
            'total_files': 0
        }
        
        # Get recent files (last 7 days)
        import datetime
        recent_cutoff = datetime.datetime.now() - datetime.timedelta(days=7)
        recent_cutoff_str = recent_cutoff.strftime("%Y-%m-%d")
        
        for date_str, files in files_by_date.items():
            summary['total_files'] += len(files)
            if date_str >= recent_cutoff_str:
                summary['recent_files'].extend(files)
        
        summary['recent_files_count'] = len(summary['recent_files'])
        
        self.logger.info(f"📊 Training data summary: {summary['total_files']} total files across {summary['total_days_with_data']} days")
        self.logger.info(f"Recent activity: {summary['recent_files_count']} files in last 7 days")
        
        return summary


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