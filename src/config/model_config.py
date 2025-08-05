"""
Model configuration for AlphaZero chess network.
"""

class ModelConfig:
    """
    Configuration class for AlphaZero chess model.
    
    Attributes:
        input_channels (int): Number of input channels (e.g., 19 for chess).
        board_size (int): Size of the chess board (e.g., 8 for standard chess).
        action_size (int): Number of possible actions (e.g., 4672 for chess).
        num_res_blocks (int): Number of residual blocks in the network.
    """
    
    def __init__(self):
        """
        Chess Input Channels Explanation (119 channels):
        
        The 119 input channels encode the complete chess position and history:
        
        1. Historical piece positions (112 channels):
           - 8 time steps (current position + 7 previous positions)
           - 2 players (White and Black)  
           - 6 piece types (Pawn, Rook, Knight, Bishop, Queen, King)
           - 8 x 2 x 6 = 96 channels for piece positions
           - 2 additional channels per time step for "repetition count"
           - 8 x 2 = 16 additional channels
           - Total: 96 + 16 = 112 channels
        
        2. Game state information (7 channels):
           - Player to move (1 channel): 1 if White to move, 0 if Black
           - Castling rights (4 channels): White/Black kingside/queenside
           - En passant target square (1 channel): Binary mask for en passant
           - Halfmove clock (1 channel): Moves since last pawn move/capture
        
        Total: 112 + 7 = 119 input channels
        
        Each channel is an 8x8 binary plane where 1 indicates the presence
        of that feature at that square. This rich representation allows the
        network to understand:
        - Current board position
        - Recent position history (for detecting repetitions)
        - All legal move constraints (castling, en passant)
        - Game phase information (halfmove clock)
        """
        self.input_channels = 119
        """
        Chess Action Space Explanation (4672 actions - Official AlphaZero):
        
        The original AlphaZero chess paper uses exactly 4672 actions, calculated as:
        8 x 8 x 73 = 4672 possible moves
        
        Where:
        - 8 x 8 = 64 possible starting squares on the chessboard
        - 73 = different move types that can be made from each square
        
        The 73 move types break down as follows:
        
        1. Queen-like moves (56 total):
           - 8 directions (N, S, E, W, NE, NW, SE, SW)
           - Up to 7 squares in each direction
           - 8 x 7 = 56 move types
        
        2. Knight moves (8 total):
           - All possible L-shaped knight moves from any square
        
        3. Pawn underpromotions (9 total):
           - 3 directions (forward, capture-left, capture-right)
           - 3 promotion pieces (knight, bishop, rook) 
           - 3 x 3 = 9 underpromotion moves
           - Note: Queen promotion is handled by queen-like moves
        
        Total: 56 + 8 + 9 = 73 move types per square
                
        The action space is represented as an 8x8x73 tensor where each slice
        corresponds to one of the 73 move types, and each position in that
        slice corresponds to a starting square on the board.
        """
        self.action_size = 4672
        self.board_size = 8
        self.num_res_blocks = 19
        self.num_filters = 256

    def __repr__(self):
        return (f"ModelConfig(input_channels={self.input_channels}, "
                f"board_size={self.board_size}, "
                f"action_size={self.action_size}, "
                f"num_res_blocks={self.num_res_blocks}), "
                f"num_filters={self.num_filters})")

def get_model_config():
    """
    Returns an instance of ModelConfig with default parameters.
    
    Returns:
        ModelConfig: An instance of the model configuration class.
    """
    return ModelConfig()