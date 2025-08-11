# src/core/game_utils.py
from abc import ABC, abstractmethod
import chess
import torch
import numpy as np

from typing import List, Optional, Any
from copy import deepcopy

class ChessGameState(ABC):
    """Abstract base class for chess game states compatible with AlphaZero."""
    
    @abstractmethod
    def get_legal_actions(self) -> List[Any]:
        """Return list of legal actions from this state."""
        pass
    
    @abstractmethod
    def apply_action(self, action: Any) -> 'ChessGameState':
        """Return new state after applying action."""
        pass
    
    @abstractmethod
    def is_terminal(self) -> bool:
        """Return True if this is a terminal state."""
        pass
    
    @abstractmethod
    def get_reward(self) -> float:
        """Return reward for current player in terminal state."""
        pass
    
    @abstractmethod
    def get_current_player(self) -> int:
        """Return current player to move (1 for white, -1 for black)."""
        pass
    
    @abstractmethod
    def to_tensor(self) -> torch.Tensor:
        """Convert state to neural network input tensor."""
        pass
    
    @abstractmethod
    def action_to_index(self, action: Any) -> int:
        """Convert action to neural network output index (0-4671)."""
        pass
    
    @abstractmethod
    def index_to_action(self, index: int) -> Any:
        """Convert neural network output index to action."""
        pass
    
    @abstractmethod
    def clone(self) -> 'ChessGameState':
        """Return deep copy of this state."""
        pass

class ChessPosition(ChessGameState):
    """
    Concrete implementation of ChessGameState using python-chess library.
    
    This class bridges the abstract ChessGameState interface with the python-chess
    library, providing all necessary methods for AlphaZero MCTS.
    """
    
    def __init__(self, board: chess.Board = None, history: List[chess.Board] = None):
        """
        Initialize chess position.
        
        Args:
            board: Optional chess.Board instance. If None, creates starting position.
            history: Optional list of previous board states (up to 8 positions).
                    Most recent positions should be at the end of the list.
        """
        self.board = board if board is not None else chess.Board()
        
        # Maintain history of board positions (up to 8 for AlphaZero)
        # history[0] = oldest, history[-1] = most recent (current-1)
        self.history = history if history is not None else []
        
        # Keep only the last 7 positions (current + 7 previous = 8 total)
        if len(self.history) > 7:
            self.history = self.history[-7:]
    
    def get_legal_actions(self) -> List[chess.Move]:
        """Return list of legal moves from current position."""
        return list(self.board.legal_moves)
    
    def apply_action(self, action: chess.Move) -> 'ChessPosition':
        """
        Return new ChessPosition after applying the given move.
        
        Args:
            action: chess.Move to apply
            
        Returns:
            New ChessPosition with move applied and updated history
        """
        # Create new board with move applied
        new_board = self.board.copy()
        new_board.push(action)
        
        # Update history: add current board to history
        new_history = self.history.copy()
        new_history.append(self.board.copy())
        
        # Keep only last 7 positions (current + 7 previous = 8 total)
        if len(new_history) > 7:
            new_history = new_history[-7:]
        
        return ChessPosition(new_board, new_history)
    
    def is_terminal(self) -> bool:
        """Check if current position is terminal (game over)."""
        return self.board.is_game_over()
    
    def get_reward(self) -> float:
        """
        Return reward for current player in terminal state.
        
        Returns:
            1.0 if current player won
            -1.0 if current player lost  
            0.0 if draw
        """
        if not self.is_terminal():
            return 0.0
            
        outcome = self.board.outcome()
        if outcome is None:
            return 0.0  # Should not happen if is_terminal() is True
            
        if outcome.winner is None:
            return 0.0  # Draw
        elif outcome.winner == self.board.turn:
            return 1.0  # Current player wins
        else:
            return -1.0  # Current player loses
    
    def get_current_player(self) -> int:
        """
        Return current player to move.
        
        Returns:
            1 for white, -1 for black
        """
        return 1 if self.board.turn == chess.WHITE else -1
    
    def to_tensor(self) -> torch.Tensor:
        """
        Convert board position to neural network input tensor.
        
        Uses AlphaZero's full 8x8x119 representation:
        - 8 history planes for each piece type (6) and color (2) = 96 planes
        - 8 planes for repetition counts
        - 2 planes for castling rights (white/black)
        - 1 plane for en passant
        - 1 plane for color to move
        - 1 plane for total move count
        - 10 planes for no-progress count (50-move rule)
        
        Returns:
            Tensor of shape (119, 8, 8) representing the board position
        """
        # Initialize tensor: 119 channels, 8x8 board
        tensor = torch.zeros(119, 8, 8, dtype=torch.float32)
        
        # Piece type mapping
        piece_to_channel_offset = {
            chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
            chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
        }
        
        # Channel indices
        channel_idx = 0
        
        # 1. Historical board positions (96 channels: 8 history × 12 piece types)
        # Fill with current position and up to 7 previous positions
        
        # Prepare all positions (current + history)
        all_positions = self.history + [self.board]
        
        # Ensure we have exactly 8 positions (pad with current position if needed)
        while len(all_positions) < 8:
            all_positions.insert(0, self.board)
        
        # Take only the last 8 positions
        all_positions = all_positions[-8:]
        
        # Fill historical channels (96 channels: 8 time steps × 12 piece types)
        for time_step, board_state in enumerate(all_positions):
            base_channel = time_step * 12  # Each time step uses 12 channels
            
            for square in chess.SQUARES:
                piece = board_state.piece_at(square)
                if piece is not None:
                    # Convert square to (row, col) coordinates
                    row = 7 - (square // 8)  # Flip vertically (rank 1 -> row 7)
                    col = square % 8
                    
                    # Determine channel based on piece type and color
                    piece_channel = piece_to_channel_offset[piece.piece_type]
                    if piece.color == chess.WHITE:
                        channel = base_channel + piece_channel  # White pieces: 0-5
                    else:
                        channel = base_channel + piece_channel + 6  # Black pieces: 6-11
                    
                    tensor[channel, row, col] = 1.0
        
        channel_idx += 96  # Skip all 96 historical channels
        
        # 2. Repetition counts (8 channels)
        # Calculate position repetitions for each historical position
        # Pre-compute position FENs for efficiency
        position_fens = [pos.fen().split(' ')[0] for pos in all_positions]
        
        for time_step in range(8):
            channel = 96 + time_step
            
            if time_step < len(all_positions):
                position_fen = position_fens[time_step]
                
                # Count how many times this position appears in history (using pre-computed FENs)
                repetition_count = position_fens.count(position_fen)
                
                # Normalize repetition count (typically 1, 2, or 3)
                normalized_count = min(repetition_count / 3.0, 1.0)
                tensor[channel].fill_(normalized_count)
        channel_idx += 8
        
        # 3. Castling rights (2 channels)
        # White castling rights
        if self.board.has_kingside_castling_rights(chess.WHITE) or self.board.has_queenside_castling_rights(chess.WHITE):
            tensor[channel_idx].fill_(1.0)
        channel_idx += 1
        
        # Black castling rights  
        if self.board.has_kingside_castling_rights(chess.BLACK) or self.board.has_queenside_castling_rights(chess.BLACK):
            tensor[channel_idx].fill_(1.0)
        channel_idx += 1
        
        # 4. En passant square (1 channel)
        if self.board.ep_square is not None:
            ep_row = 7 - (self.board.ep_square // 8)
            ep_col = self.board.ep_square % 8
            tensor[channel_idx, ep_row, ep_col] = 1.0
        channel_idx += 1
        
        # 5. Color to move (1 channel)
        if self.board.turn == chess.WHITE:
            tensor[channel_idx].fill_(1.0)
        channel_idx += 1
        
        # 6. Total move count (1 channel)
        # Normalize move count to [0, 1] range (assuming max ~500 moves)
        move_count = self.board.fullmove_number
        normalized_move_count = min(move_count / 500.0, 1.0)
        tensor[channel_idx].fill_(normalized_move_count)
        channel_idx += 1
        
        # 7. No-progress count for 50-move rule (10 channels)
        # Each channel represents a range of halfmove clock values
        halfmove_clock = self.board.halfmove_clock
        for i in range(10):
            # Each channel represents 10 halfmoves (5 full moves)
            lower_bound = i * 10
            upper_bound = (i + 1) * 10
            if lower_bound <= halfmove_clock < upper_bound:
                tensor[channel_idx + i].fill_(1.0)
                break
        channel_idx += 10
        
        assert channel_idx == 119, f"Expected 119 channels, got {channel_idx}"
        
        return tensor
    
    def action_to_index(self, action: chess.Move) -> int:
        """
        Convert chess move to neural network output index.
        
        AlphaZero uses 4672 possible moves total:
        - 64*64 = 4096 queen moves (from-to squares)
        - 64*8 = 512 knight moves (8 knight directions)  
        - 64 under-promotions (N, B, R for each file)
        
        Args:
            action: chess.Move to convert
            
        Returns:
            Index in range [0, 4671]
        """
        from_square = action.from_square
        to_square = action.to_square
        
        # Handle promotions
        if action.promotion is not None:
            # Under-promotions: Knight, Bishop, Rook (Queen is handled as normal move)
            if action.promotion != chess.QUEEN:
                file_idx = chess.square_file(to_square)
                if action.promotion == chess.KNIGHT:
                    return 4096 + 512 + file_idx
                elif action.promotion == chess.BISHOP:
                    return 4096 + 512 + 8 + file_idx
                elif action.promotion == chess.ROOK:
                    return 4096 + 512 + 16 + file_idx
        
        # Calculate direction and distance
        from_rank, from_file = divmod(from_square, 8)
        to_rank, to_file = divmod(to_square, 8)
        
        rank_diff = to_rank - from_rank
        file_diff = to_file - from_file
        
        # Knight moves
        if abs(rank_diff) == 2 and abs(file_diff) == 1:
            # Knight move
            knight_moves = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
            try:
                direction_idx = knight_moves.index((rank_diff, file_diff))
                return 4096 + from_square * 8 + direction_idx
            except ValueError:
                pass
        elif abs(rank_diff) == 1 and abs(file_diff) == 2:
            # Knight move
            knight_moves = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
            try:
                direction_idx = knight_moves.index((rank_diff, file_diff))
                return 4096 + from_square * 8 + direction_idx
            except ValueError:
                pass
        
        # Queen-like moves (straight lines)
        return from_square * 64 + to_square
    
    def index_to_action(self, index: int) -> chess.Move:
        """
        Convert neural network output index to chess move.
        
        Args:
            index: Index in range [0, 4671]
            
        Returns:
            Corresponding chess.Move
        """
        if index < 4096:
            # Queen-like moves
            from_square = index // 64
            to_square = index % 64
            return chess.Move(from_square, to_square)
        
        elif index < 4096 + 512:
            # Knight moves
            knight_idx = index - 4096
            from_square = knight_idx // 8
            direction_idx = knight_idx % 8
            
            knight_moves = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
            rank_diff, file_diff = knight_moves[direction_idx]
            
            from_rank, from_file = divmod(from_square, 8)
            to_rank = from_rank + rank_diff
            to_file = from_file + file_diff
            
            if 0 <= to_rank < 8 and 0 <= to_file < 8:
                to_square = to_rank * 8 + to_file
                return chess.Move(from_square, to_square)
        
        else:
            # Under-promotions
            promotion_idx = index - 4096 - 512
            file_idx = promotion_idx % 8
            promotion_type_idx = promotion_idx // 8
            
            # Determine promotion piece
            promotion_pieces = [chess.KNIGHT, chess.BISHOP, chess.ROOK]
            if promotion_type_idx < len(promotion_pieces):
                promotion_piece = promotion_pieces[promotion_type_idx]
                
                # Assume promotion from 7th to 8th rank for white
                from_square = chess.square(file_idx, 6)  # 7th rank
                to_square = chess.square(file_idx, 7)    # 8th rank
                
                return chess.Move(from_square, to_square, promotion=promotion_piece)
        
        # Fallback - should not reach here with valid indices
        raise ValueError(f"Invalid move index: {index}")
    
    def clone(self) -> 'ChessPosition':
        """Return deep copy of current chess position with history."""
        return ChessPosition(self.board.copy(), [board.copy() for board in self.history])
    
    def __str__(self) -> str:
        """String representation of the chess position."""
        return str(self.board)
    
    def __repr__(self) -> str:
        """Representation of the chess position."""
        return f"ChessPosition('{self.board.fen()}')"
    
    def fen(self) -> str:
        """Return FEN string of current position."""
        return self.board.fen()
    
    def pgn(self) -> str:
        """Return PGN representation of the game."""
        return str(self.board)
        
    def get_history_length(self) -> int:
        """Get the number of positions in history (not including current)."""
        return len(self.history)
    
    @classmethod
    def from_game_moves(cls, moves: List[chess.Move], starting_board: chess.Board = None) -> 'ChessPosition':
        """
        Create ChessPosition from a sequence of moves with proper history.
        
        Args:
            moves: List of chess moves to apply in sequence
            starting_board: Optional starting position (default: standard starting position)
            
        Returns:
            ChessPosition with proper history built up from the moves
        """
        if starting_board is None:
            starting_board = chess.Board()
        
        position = cls(starting_board.copy())
        
        for move in moves:
            position = position.apply_action(move)
        
        return position


def evaluate_material_balance(board: chess.Board) -> float:
    """
    Evaluate material balance from the perspective of the current player to move.
    
    Args:
        board: Chess board position
        
    Returns:
        Material balance in centipawns (positive = advantage for current player)
    """
    # Standard piece values in centipawns
    PIECE_VALUES = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 0  # King has no material value
    }
    
    white_material = 0
    black_material = 0
    
    # Count material for both sides
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            value = PIECE_VALUES[piece.piece_type]
            if piece.color == chess.WHITE:
                white_material += value
            else:
                black_material += value
    
    # Return material advantage from current player's perspective
    if board.turn == chess.WHITE:
        return white_material - black_material
    else:
        return black_material - white_material


def is_material_advantage_overwhelming(board: chess.Board, threshold_centipawns: int = 500) -> bool:
    """
    Check if current player has an overwhelming material advantage that should trigger resignation.
    
    Args:
        board: Chess board position
        threshold_centipawns: Minimum material advantage to consider overwhelming (default: 500 = 5 pawns)
        
    Returns:
        True if current player has overwhelming material advantage
    """
    return evaluate_material_balance(board) >= threshold_centipawns


def should_resign_material(board: chess.Board, threshold_centipawns: int = 500) -> bool:
    """
    Check if current player should resign due to overwhelming material disadvantage.
    
    Args:
        board: Chess board position  
        threshold_centipawns: Minimum material disadvantage to trigger resignation (default: 500 = 5 pawns)
        
    Returns:
        True if current player should resign due to material disadvantage
    """
    return evaluate_material_balance(board) <= -threshold_centipawns