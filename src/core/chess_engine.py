"""
Interactive Chess Engine with GUI

This module provides a graphical chess interface using python-chess and tkinter
that allows you to play against your trained AlphaZero models.

Features:
- Visual chess board with drag-and-drop moves
- Load and play against trained models
- Multiple difficulty levels (MCTS simulations)
- Game analysis and move history
- Export games to PGN format

Author: Francesco Finucci
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import chess
import chess.svg
import chess.pgn
from pathlib import Path
import torch
import time
from typing import Optional, List, Tuple, Dict
import threading
import queue
import io
from datetime import datetime

# Try to import GUI dependencies
try:
    from PIL import Image, ImageTk
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False
    print("GUI dependencies not available. Install with: pip install pillow")

# Import our modules
import sys
sys.path.append(str(Path(__file__).parent.parent))
from core.alphazero_net import AlphaZeroNet
from core.alphazero_mcts import AlphaZeroMCTS
from core.game_utils import ChessPosition


class ChessEngineGUI:
    """
    Graphical Chess Engine for playing against trained AlphaZero models.
    """
    
    def __init__(self):
        """Initialize the chess engine GUI."""
        if not GUI_AVAILABLE:
            raise ImportError("GUI dependencies not available. Install with: pip install pillow")
        
        self.root = tk.Tk()
        self.root.title("AlphaZero Chess Engine")
        self.root.geometry("1000x800")
        
        # Game state
        self.board = chess.Board()
        self.game_history = []
        self.model = None
        self.mcts_engine = None
        self.device = 'cpu'
        self.use_mcts = True  # Use MCTS by default (as per literature)
        self.mcts_simulations = 100  # For MCTS mode
        self.human_color = chess.WHITE
        self.game_active = False
        
        # GUI elements
        self.selected_square = None
        self.board_squares = {}
        self.move_queue = queue.Queue()
        
        # Initialize GUI
        self.setup_gui()
        self.update_board_display()
        
    def setup_gui(self):
        """Setup the graphical user interface."""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Chess board
        board_frame = ttk.LabelFrame(main_frame, text="Chess Board", padding=10)
        board_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Chess board canvas
        self.board_canvas = tk.Canvas(board_frame, width=480, height=480, bg='white')
        self.board_canvas.pack(pady=10)
        self.board_canvas.bind("<Button-1>", self.on_square_click)
        
        # Right panel - Controls and info
        control_frame = ttk.LabelFrame(main_frame, text="Game Controls", padding=10)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        
        # Model loading
        model_frame = ttk.LabelFrame(control_frame, text="Model Selection", padding=5)
        model_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(model_frame, text="Load Model", command=self.load_model).pack(fill=tk.X, pady=2)
        self.model_label = ttk.Label(model_frame, text="No model loaded", foreground="red")
        self.model_label.pack(pady=2)
        
        # Game settings
        settings_frame = ttk.LabelFrame(control_frame, text="Game Settings", padding=5)
        settings_frame.pack(fill=tk.X, pady=5)
        
        # Color selection
        ttk.Label(settings_frame, text="Play as:").pack(anchor=tk.W)
        self.color_var = tk.StringVar(value="White")
        color_frame = ttk.Frame(settings_frame)
        color_frame.pack(fill=tk.X, pady=2)
        ttk.Radiobutton(color_frame, text="White", variable=self.color_var, 
                       value="White", command=self.on_color_change).pack(side=tk.LEFT)
        ttk.Radiobutton(color_frame, text="Black", variable=self.color_var, 
                       value="Black", command=self.on_color_change).pack(side=tk.LEFT)
        
        # Difficulty setting
        ttk.Label(settings_frame, text="AI Mode:").pack(anchor=tk.W)
        self.mode_var = tk.StringVar(value="Strong (MCTS)")
        mode_frame = ttk.Frame(settings_frame)
        mode_frame.pack(fill=tk.X, pady=2)
        ttk.Radiobutton(mode_frame, text="Fast (Direct)", variable=self.mode_var, 
                       value="Fast (Direct)", command=self.on_mode_change).pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="Strong (MCTS)", variable=self.mode_var, 
                       value="Strong (MCTS)", command=self.on_mode_change).pack(anchor=tk.W)
        
        ttk.Label(settings_frame, text="MCTS Simulations (Strong mode):").pack(anchor=tk.W)
        self.difficulty_var = tk.IntVar(value=100)
        difficulty_scale = ttk.Scale(settings_frame, from_=25, to=400, 
                                   variable=self.difficulty_var, orient=tk.HORIZONTAL)
        difficulty_scale.pack(fill=tk.X, pady=2)
        self.difficulty_label = ttk.Label(settings_frame, text="100 simulations")
        self.difficulty_label.pack()
        difficulty_scale.configure(command=self.on_difficulty_change)
        
        # Game controls
        game_frame = ttk.LabelFrame(control_frame, text="Game Control", padding=5)
        game_frame.pack(fill=tk.X, pady=5)
        
        self.start_button = ttk.Button(game_frame, text="Start Game", command=self.start_game)
        self.start_button.pack(fill=tk.X, pady=2)
        
        ttk.Button(game_frame, text="Reset Game", command=self.reset_game).pack(fill=tk.X, pady=2)
        ttk.Button(game_frame, text="Undo Move", command=self.undo_move).pack(fill=tk.X, pady=2)
        
        # Game info
        info_frame = ttk.LabelFrame(control_frame, text="Game Information", padding=5)
        info_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.status_label = ttk.Label(info_frame, text="Load a model to start playing")
        self.status_label.pack(anchor=tk.W, pady=2)
        
        self.turn_label = ttk.Label(info_frame, text="Turn: White to move")
        self.turn_label.pack(anchor=tk.W, pady=2)
        
        # Move history
        ttk.Label(info_frame, text="Move History:").pack(anchor=tk.W, pady=(10, 2))
        
        history_frame = ttk.Frame(info_frame)
        history_frame.pack(fill=tk.BOTH, expand=True)
        
        self.history_text = tk.Text(history_frame, height=10, width=30, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(history_frame, orient=tk.VERTICAL, command=self.history_text.yview)
        self.history_text.configure(yscrollcommand=scrollbar.set)
        
        self.history_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Export controls
        export_frame = ttk.LabelFrame(control_frame, text="Export", padding=5)
        export_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(export_frame, text="Export PGN", command=self.export_pgn).pack(fill=tk.X, pady=2)
        
        self.create_board_squares()
        
    def create_board_squares(self):
        """Create the visual chess board squares."""
        square_size = 60
        
        for rank in range(8):
            for file in range(8):
                x1 = file * square_size
                y1 = (7 - rank) * square_size  # Flip board for white perspective
                x2 = x1 + square_size
                y2 = y1 + square_size
                
                # Determine square color
                is_light = (rank + file) % 2 == 0
                color = "#F0D9B5" if is_light else "#B58863"
                
                square = chess.square(file, rank)
                rect = self.board_canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="black")
                
                self.board_squares[square] = {
                    'rect': rect,
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'piece_text': None
                }
    
    def update_board_display(self):
        """Update the visual representation of the chess board."""
        # Clear existing pieces
        for square_info in self.board_squares.values():
            if square_info['piece_text']:
                self.board_canvas.delete(square_info['piece_text'])
                square_info['piece_text'] = None
        
        # Draw pieces
        piece_symbols = {
            chess.PAWN: ('♙', '♟'), chess.ROOK: ('♖', '♜'),
            chess.KNIGHT: ('♘', '♞'), chess.BISHOP: ('♗', '♝'),
            chess.QUEEN: ('♕', '♛'), chess.KING: ('♔', '♚')
        }
        
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                square_info = self.board_squares[square]
                symbol = piece_symbols[piece.piece_type][0 if piece.color else 1]
                
                x_center = (square_info['x1'] + square_info['x2']) // 2
                y_center = (square_info['y1'] + square_info['y2']) // 2
                
                text_id = self.board_canvas.create_text(
                    x_center, y_center, text=symbol, font=("Arial", 32), fill="black"
                )
                square_info['piece_text'] = text_id
        
        # Highlight selected square
        if self.selected_square is not None:
            self.highlight_square(self.selected_square, "#FFFF00")
        
        # Update turn indicator
        turn_text = "White to move" if self.board.turn == chess.WHITE else "Black to move"
        self.turn_label.config(text=f"Turn: {turn_text}")
        
        # Check game status
        if self.board.is_checkmate():
            winner = "Black" if self.board.turn == chess.WHITE else "White"
            self.status_label.config(text=f"Checkmate! {winner} wins!", foreground="red")
            self.game_active = False
        elif self.board.is_stalemate():
            self.status_label.config(text="Stalemate! Game is a draw.", foreground="orange")
            self.game_active = False
        elif self.board.is_check():
            self.status_label.config(text="Check!", foreground="red")
    
    def highlight_square(self, square: int, color: str):
        """Highlight a specific square on the board."""
        square_info = self.board_squares[square]
        self.board_canvas.create_rectangle(
            square_info['x1'], square_info['y1'], square_info['x2'], square_info['y2'],
            outline=color, width=3, tags="highlight"
        )
    
    def clear_highlights(self):
        """Clear all square highlights."""
        self.board_canvas.delete("highlight")
    
    def on_square_click(self, event):
        """Handle mouse clicks on the chess board."""
        if not self.game_active or self.board.turn != self.human_color:
            return
        
        # Determine which square was clicked
        x, y = event.x, event.y
        file = x // 60
        rank = 7 - (y // 60)  # Flip for white perspective
        
        if 0 <= file <= 7 and 0 <= rank <= 7:
            clicked_square = chess.square(file, rank)
            
            if self.selected_square is None:
                # Select a piece
                piece = self.board.piece_at(clicked_square)
                if piece and piece.color == self.human_color:
                    self.selected_square = clicked_square
                    self.clear_highlights()
                    self.highlight_square(clicked_square, "#FFFF00")
            else:
                # Try to make a move
                move = chess.Move(self.selected_square, clicked_square)
                
                # Handle pawn promotion
                if (self.board.piece_at(self.selected_square) and 
                    self.board.piece_at(self.selected_square).piece_type == chess.PAWN):
                    if ((self.human_color == chess.WHITE and rank == 7) or 
                        (self.human_color == chess.BLACK and rank == 0)):
                        move.promotion = chess.QUEEN  # Auto-promote to queen
                
                if move in self.board.legal_moves:
                    self.make_move(move)
                
                self.selected_square = None
                self.clear_highlights()
    
    def make_move(self, move: chess.Move):
        """Make a move on the board and update displays."""
        # Record move in history
        move_san = self.board.san(move)
        self.board.push(move)
        
        move_number = len(self.board.move_stack)
        if move_number % 2 == 1:  # White move
            self.history_text.insert(tk.END, f"{(move_number + 1) // 2}. {move_san} ")
        else:  # Black move
            self.history_text.insert(tk.END, f"{move_san}\n")
        
        self.history_text.see(tk.END)
        self.update_board_display()
        
        # If it's now the AI's turn, schedule AI move
        if self.game_active and self.board.turn != self.human_color:
            self.root.after(100, self.ai_move_threaded)
    
    def ai_move_threaded(self):
        """Start AI move calculation in a separate thread."""
        if not self.game_active or not self.model:
            return
        
        mode_text = "MCTS" if self.use_mcts else "Direct"
        self.status_label.config(text=f"AI is thinking ({mode_text})...", foreground="blue")
        
        def calculate_move():
            try:
                if self.use_mcts and self.mcts_engine:
                    # Use MCTS (stronger but slower)
                    ai_move = self.get_mcts_move()
                else:
                    # Use direct inference (faster)
                    ai_move = self.get_direct_move()
                self.move_queue.put(ai_move)
            except Exception as e:
                self.move_queue.put(None)
                print(f"AI move error: {e}")
        
        thread = threading.Thread(target=calculate_move)
        thread.daemon = True
        thread.start()
        
        self.check_ai_move()
    
    def get_mcts_move(self) -> Optional[chess.Move]:
        """Get AI move using MCTS (stronger but slower)."""
        # Convert chess.Board to ChessPosition
        chess_state = ChessPosition(self.board)
        
        # Use MCTS to get best move
        best_action = self.mcts_engine.get_best_action(
            chess_state, 
            num_simulations=self.mcts_simulations
        )
        
        return best_action
    
    def get_direct_move(self) -> Optional[chess.Move]:
        """Get AI move using direct neural network inference (fast)."""
        # Convert chess.Board to ChessPosition to use the same interface
        chess_state = ChessPosition(self.board)
        
        # Get state tensor for neural network
        state_tensor = chess_state.to_tensor().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            policy_logits, value = self.model(state_tensor)
            policy_probs = torch.softmax(policy_logits, dim=1)[0]
        
        # Get legal moves and their probabilities
        legal_moves = chess_state.get_legal_actions()
        if not legal_moves:
            return None
        
        move_probs = []
        for move in legal_moves:
            action_idx = chess_state.action_to_index(move)
            if action_idx is not None and action_idx < len(policy_probs):
                prob = policy_probs[action_idx].item()
                move_probs.append((move, prob))
        
        if not move_probs:
            return None
        
        # Sort by probability and pick the best move
        move_probs.sort(key=lambda x: x[1], reverse=True)
        return move_probs[0][0]
    
    def check_ai_move(self):
        """Check if AI move calculation is complete."""
        try:
            ai_move = self.move_queue.get_nowait()
            if ai_move:
                self.make_move(ai_move)
                self.status_label.config(text="Your turn!", foreground="green")
            else:
                self.status_label.config(text="AI error - your turn", foreground="red")
        except queue.Empty:
            # AI still thinking, check again in 100ms
            self.root.after(100, self.check_ai_move)
    
    def load_model(self):
        """Load a trained AlphaZero model."""
        file_path = filedialog.askopenfilename(
            title="Select Model Checkpoint",
            filetypes=[("PyTorch files", "*.pth"), ("All files", "*.*")],
            initialdir="checkpoints"
        )
        
        if file_path:
            try:
                checkpoint = torch.load(file_path, map_location='cpu')
                
                # Get the actual model configuration from the checkpoint
                job_config = checkpoint.get('job_config', {})
                model_size = job_config.get('model_size', 'micro')
                
                # Load model sizes configuration
                import json
                model_sizes_path = Path(__file__).parent.parent / "experiments" / "configs" / "model_sizes.json"
                with open(model_sizes_path, 'r') as f:
                    model_sizes = json.load(f)
                
                size_config = model_sizes.get(model_size, model_sizes['micro'])
                
                # Create model with correct configuration
                self.model = AlphaZeroNet(
                    board_size=8,                               # Chess board is 8x8
                    action_size=4672,                          # Chess action space
                    num_filters=size_config['filters'],        # From model size config
                    num_res_blocks=size_config['blocks'],      # From model size config  
                    input_channels=119                         # Full AlphaZero chess representation
                )
                
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()
                
                # Create MCTS engine
                self.mcts_engine = AlphaZeroMCTS(
                    neural_network=self.model,
                    c_puct=1.0,
                    device=self.device
                )
                
                model_name = Path(file_path).stem
                self.model_label.config(text=f"Loaded: {model_name} ({model_size})", foreground="green")
                self.start_button.config(state="normal")
                self.status_label.config(text="Model loaded! Click 'Start Game' to play.")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {e}")
                self.model_label.config(text="Failed to load model", foreground="red")
    
    def on_color_change(self):
        """Handle color selection change."""
        self.human_color = chess.WHITE if self.color_var.get() == "White" else chess.BLACK
    
    def on_mode_change(self):
        """Handle AI mode change."""
        self.use_mcts = self.mode_var.get() == "Strong (MCTS)"
    
    def on_difficulty_change(self, value):
        """Handle difficulty slider change."""
        self.mcts_simulations = int(float(value))
        self.difficulty_label.config(text=f"{self.mcts_simulations} simulations")
    
    def start_game(self):
        """Start a new game."""
        if not self.model:
            messagebox.showwarning("Warning", "Please load a model first!")
            return
        
        self.reset_game()
        self.game_active = True
        self.status_label.config(text="Game started! Make your move.", foreground="green")
        
        # If human is black, AI moves first
        if self.human_color == chess.BLACK:
            self.root.after(500, self.ai_move_threaded)
    
    def reset_game(self):
        """Reset the game to initial state."""
        self.board.reset()
        self.selected_square = None
        self.game_active = False
        self.clear_highlights()
        self.update_board_display()
        self.history_text.delete(1.0, tk.END)
        self.status_label.config(text="Game reset. Click 'Start Game' to play.")
    
    def undo_move(self):
        """Undo the last move (or two moves if AI just played)."""
        if len(self.board.move_stack) == 0:
            return
        
        # Undo AI move if it was the last move
        if self.board.turn == self.human_color and len(self.board.move_stack) > 0:
            self.board.pop()
        
        # Undo human move
        if len(self.board.move_stack) > 0:
            self.board.pop()
        
        self.update_board_display()
        self.update_move_history()
    
    def update_move_history(self):
        """Update the move history display."""
        self.history_text.delete(1.0, tk.END)
        
        for i, move in enumerate(self.board.move_stack):
            move_san = self.board.san(move)
            if i % 2 == 0:  # White move
                self.history_text.insert(tk.END, f"{(i // 2) + 1}. {move_san} ")
            else:  # Black move
                self.history_text.insert(tk.END, f"{move_san}\n")
    
    def export_pgn(self):
        """Export the current game to PGN format."""
        if len(self.board.move_stack) == 0:
            messagebox.showinfo("Info", "No moves to export!")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Game as PGN",
            defaultextension=".pgn",
            filetypes=[("PGN files", "*.pgn"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                game = chess.pgn.Game()
                game.headers["Event"] = "AlphaZero vs Human"
                game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
                game.headers["White"] = "Human" if self.human_color == chess.WHITE else "AlphaZero"
                game.headers["Black"] = "AlphaZero" if self.human_color == chess.WHITE else "Human"
                
                node = game
                board_copy = chess.Board()
                for move in self.board.move_stack:
                    node = node.add_variation(move)
                    board_copy.push(move)
                
                with open(file_path, 'w') as f:
                    print(game, file=f)
                
                messagebox.showinfo("Success", f"Game exported to {file_path}")
            
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export game: {e}")
    
    def run(self):
        """Start the GUI application."""
        self.root.mainloop()


def main():
    """Main entry point for the chess engine GUI."""
    if not GUI_AVAILABLE:
        print("GUI dependencies not available.")
        print("Install with: pip install pillow cairosvg")
        return
    
    try:
        app = ChessEngineGUI()
        app.run()
    except Exception as e:
        print(f"Error starting chess engine: {e}")


if __name__ == "__main__":
    main()
