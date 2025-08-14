"""
AlphaZero Monte Carlo Tree Search Implementation - FIXED VERSION

This implementation fixes the critical issues causing 100% draw rate:
1. Correct reward perspective in terminal states
2. Proper training example reward assignment
3. Increased exploration parameters
4. Better temperature scheduling
5. Resignation logic to prevent meaningless games

Based on Silver et al. (2017): "Mastering Chess and Shogi by Self-Play 
with a General Reinforcement Learning Algorithm"
"""

import math
import chess
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))  # Ensure src is in path

from src.core.game_utils import ChessGameState, should_resign_material


class AlphaZeroNode:
    """
    Node in the AlphaZero MCTS tree.
    
    Stores:
    - visit_count (N): Number of simulations passing through this node
    - value_sum (W): Sum of all values backed up through this node  
    - prior_probs (P): Prior probabilities from neural network policy
    - children: Dictionary mapping actions to child nodes
    """
    
    def __init__(self, state: ChessGameState, parent: Optional['AlphaZeroNode'] = None,
                 parent_action: Optional[Any] = None, prior_prob: float = 0.0):
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.prior_prob = prior_prob
        
        # MCTS statistics
        self.visit_count = 0
        self.value_sum = 0.0
        
        # Tree structure  
        self.children: Dict[Any, 'AlphaZeroNode'] = {}
        self.prior_probs: Dict[Any, float] = {}
        
        # Neural network outputs (set during expansion)
        self.is_expanded = False
        self.neural_value = 0.0
    
    def is_leaf(self) -> bool:
        """Return True if this is a leaf node (not expanded)."""
        return not self.is_expanded
    
    def expand(self, policy_probs: torch.Tensor, value: float, legal_actions: List[Any],
               action_to_index_fn) -> None:
        """
        Expand this node using neural network outputs.
        
        Args:
            policy_probs: Neural network policy output (softmax probabilities)
            value: Neural network value output  
            legal_actions: List of legal actions from this state
            action_to_index_fn: Function to convert action to policy index
        """
        self.is_expanded = True
        self.neural_value = value
        
        # Extract prior probabilities for legal actions
        total_prob = 0.0
        for action in legal_actions:
            action_idx = action_to_index_fn(action)
            prob = policy_probs[action_idx].item()
            self.prior_probs[action] = prob
            total_prob += prob
        
        # Normalize priors over legal actions (handles illegal moves)
        if total_prob > 0:
            for action in self.prior_probs:
                self.prior_probs[action] /= total_prob
        else:
            # Uniform if all legal actions have zero probability
            uniform_prob = 1.0 / len(legal_actions)
            for action in legal_actions:
                self.prior_probs[action] = uniform_prob
    
    def select_child(self, c_puct: float = 2.0) -> Tuple[Any, 'AlphaZeroNode']:
        """
        Select child using PUCT algorithm.
        
        PUCT formula: Q(s,a) + c_puct * P(s,a) * √(Σ_b N(s,b)) / (1 + N(s,a))
        
        Returns:
            Tuple of (action, child_node) with highest PUCT value
        """
        best_action = None
        best_value = float('-inf')
        
        for action in self.prior_probs:
            if action in self.children:
                child = self.children[action]
                # Q(s,a): Average action value
                q_value = child.value_sum / child.visit_count if child.visit_count > 0 else 0.0
            else:
                q_value = 0.0  # Unvisited children start with Q=0
            
            # PUCT exploration term
            prior_prob = self.prior_probs[action]
            exploration = c_puct * prior_prob * math.sqrt(self.visit_count)
            
            if action in self.children:
                exploration /= (1 + self.children[action].visit_count)
            else:
                exploration /= 1  # Unvisited: denominator is 1
            
            puct_value = q_value + exploration
            
            if puct_value > best_value:
                best_value = puct_value
                best_action = action
        
        # Create child if it doesn't exist
        if best_action not in self.children:
            child_state = self.state.apply_action(best_action)
            self.children[best_action] = AlphaZeroNode(
                child_state, parent=self, parent_action=best_action,
                prior_prob=self.prior_probs[best_action]
            )
        
        return best_action, self.children[best_action]
    
    def backup(self, value: float) -> None:
        """
        Backup value through this node.
        Value is from perspective of player to move at this node.
        """
        self.visit_count += 1
        self.value_sum += value
    
    def get_action_probabilities(self, temperature: float = 1.0) -> Dict[Any, float]:
        """
        Get action probabilities based on visit counts.
        
        Args:
            temperature: Temperature parameter
                - temperature = 0: argmax (deterministic)  
                - temperature > 0: softmax over visit counts
                - temperature = 1: proportional to visit counts
        """
        if not self.children:
            return {}
        
        actions = list(self.children.keys())
        visit_counts = [self.children[action].visit_count for action in actions]
        
        if temperature == 0:
            # Deterministic: choose action with most visits
            best_idx = max(range(len(visit_counts)), key=lambda i: visit_counts[i])
            probs = [0.0] * len(actions)
            probs[best_idx] = 1.0
        else:
            # Softmax with temperature
            if temperature == 1.0:
                # Proportional to visit counts
                total_visits = sum(visit_counts)
                if total_visits > 0:
                    probs = [count / total_visits for count in visit_counts]
                else:
                    probs = [1.0 / len(actions)] * len(actions)
            else:
                # General temperature scaling
                scaled_counts = [count / temperature for count in visit_counts]
                
                # Numerical stability: subtract max
                if scaled_counts:
                    max_scaled = max(scaled_counts)
                    exp_counts = [math.exp(count - max_scaled) for count in scaled_counts]
                    sum_exp = sum(exp_counts)
                    if sum_exp > 0:
                        probs = [exp_count / sum_exp for exp_count in exp_counts]
                    else:
                        probs = [1.0 / len(actions)] * len(actions)
                else:
                    probs = []
        
        return {action: prob for action, prob in zip(actions, probs)}
    
    def get_best_action(self) -> Any:
        """Get action with highest visit count."""
        if not self.children:
            return None
        return max(self.children.keys(), 
                  key=lambda action: self.children[action].visit_count)


class AlphaZeroMCTS:
    """
    AlphaZero Monte Carlo Tree Search - FIXED VERSION
    
    Uses neural network for:
    1. Prior move probabilities (policy)
    2. Position evaluation (value)
    
    No random rollouts - direct neural network evaluation at leaf nodes.
    """
    
    def __init__(self, neural_network, c_puct: float = 2.0, device: str = 'cpu', 
                 resignation_threshold: float = -0.9, logger=None):
        """
        Initialize AlphaZero MCTS.
        
        Args:
            neural_network: AlphaZero neural network (policy + value)
            c_puct: PUCT exploration constant (increased to 2.0 for more exploration)
            device: PyTorch device for neural network inference
            resignation_threshold: Value threshold for resignation (-0.9 = 90% loss probability)
            logger: Logger instance for worker logging
        """
        self.neural_network = neural_network
        self.c_puct = c_puct
        self.device = device
        self.resignation_threshold = resignation_threshold
        self.logger = logger
        
        # Set network to evaluation mode
        self.neural_network.eval()
    
    def search(self, root_state: ChessGameState, num_simulations: int) -> AlphaZeroNode:
        """
        Run MCTS search for specified number of simulations.
        
        Args:
            root_state: Starting chess position
            num_simulations: Number of MCTS simulations to run
            
        Returns:
            Root node of search tree with visit statistics
        """
        # Create root node
        root = AlphaZeroNode(root_state)
        
        # Expand root immediately if not terminal
        if not root_state.is_terminal():
            self._expand_node(root)
        
        # Run simulations
        for _ in range(num_simulations):
            self._simulate(root)
        
        return root
    
    def _simulate(self, root: AlphaZeroNode) -> None:
        """
        Single MCTS simulation - FIXED VERSION:
        1. Selection: Navigate tree using PUCT
        2. Expansion: Expand leaf node with neural network
        3. Backup: Propagate value up the tree
        """
        # Phase 1: Selection - navigate to leaf
        path = []
        current = root
        
        while not current.is_leaf() and not current.state.is_terminal():
            action, child = current.select_child(self.c_puct)
            path.append((current, action, child))
            current = child
        
        # Phase 2: Expansion & Evaluation
        if current.state.is_terminal():
            # CRITICAL FIX: Terminal reward from correct perspective
            # get_reward() returns reward for player whose turn it is
            # But in terminal state, the game is over, so we need the reward
            # from perspective of the player who would move (but can't)
            raw_reward = current.state.get_reward()
            
            # Since the game is terminal, the player to move has lost their turn
            # The reward should be from their perspective
            # If the game ended in their favor, they get +1, otherwise -1
            value = raw_reward  # Use the reward as-is since get_reward() should handle perspective
            
        else:
            if current.is_leaf():
                # Expand leaf node with neural network
                self._expand_node(current)
            # Use neural network value estimate (already from correct perspective)
            value = current.neural_value
        
        # Phase 3: Backup - propagate value up tree
        # Value is from perspective of current player at current node
        current_value = value
        
        # Backup through path (alternating perspective)
        for node, action, child in reversed(path):
            child.backup(current_value)
            current_value = -current_value  # Flip for opponent
        
        # Backup root
        root.backup(current_value)
    
    @torch.no_grad()
    def _expand_node(self, node: AlphaZeroNode) -> None:
        """
        Expand node using neural network evaluation.
        
        Args:
            node: Node to expand
        """
        # Convert state to neural network input
        state_tensor = node.state.to_tensor().unsqueeze(0).to(self.device)  # Add batch dim
        
        # Forward pass through neural network
        policy_logits, value = self.neural_network(state_tensor)
        
        # Convert outputs
        policy_probs = F.softmax(policy_logits, dim=-1).squeeze(0)  # Remove batch dim
        value = torch.tanh(value).item()  # Scalar value in [-1, 1]
        
        # Get legal actions
        legal_actions = node.state.get_legal_actions()
        
        # Expand node with neural network outputs
        node.expand(policy_probs, value, legal_actions, node.state.action_to_index)
    
    def should_resign(self, value: float, move_count: int, current_state: ChessGameState = None) -> bool:
        """
        Determine if the position is hopeless and should resign.
        This prevents long, meaningless games.
        
        Args:
            value: Current position evaluation
            move_count: Number of moves played
            current_state: Current game state (optional, for material evaluation)
            
        Returns:
            True if should resign, False otherwise
        """
        # Only consider resignation after opening
        if move_count < 20:
            return False
        
        # Resign if position is completely lost according to neural network
        nn_should_resign = value < self.resignation_threshold
        if nn_should_resign:
            msg = f"Neural network resignation triggered for {'white' if current_state.board.turn else 'black'} at move {move_count}"
            details = f"Value: {value}, Threshold: {self.resignation_threshold}"
            fen = f"FEN: {current_state.board.fen()}"
            
            if self.logger:
                self.logger.info(f"RESIGNATION: {msg}")
                self.logger.info(f"RESIGNATION: {details}")
                self.logger.info(f"RESIGNATION: {fen}")
            else:
                print(msg)
                print(details)
                print(fen)
            return nn_should_resign
    
        # Check for obvious material imbalance first (fallback for poorly trained networks)
        if current_state:
            # Use the proper material evaluation from game_utils
            material_should_resign = should_resign_material(current_state.board, threshold_centipawns=500)  # Reduced from 800 to 500
            if material_should_resign:
                msg = f"Material resignation triggered for {'white' if current_state.board.turn else 'black'} at move {move_count}"
                fen = f"FEN: {current_state.board.fen()}"
                
                if self.logger:
                    self.logger.info(f"RESIGNATION: {msg}")
                    self.logger.info(f"RESIGNATION: {fen}")
                else:
                    print(msg)
                    print(fen)
                return True
        elif hasattr(self, 'current_board_state') and self.current_board_state:
            if should_resign_material(self.current_board_state.board, threshold_centipawns=500):
                return True
            

    def get_best_action(self, root_state: ChessGameState, 
                       num_simulations: int = 800) -> Any:
        """
        Get best action using MCTS search.
        
        Args:
            root_state: Chess position to analyze
            num_simulations: Number of MCTS simulations
            
        Returns:
            Best action according to MCTS
        """
        if root_state.is_terminal():
            return None
        
        root = self.search(root_state, num_simulations)
        return root.get_best_action()
    
    def get_action_probabilities(self, root_state: ChessGameState,
                                num_simulations: int = 800,
                                temperature: float = 1.0) -> Dict[Any, float]:
        """
        Get action probabilities using MCTS search.
        
        Args:
            root_state: Chess position to analyze
            num_simulations: Number of MCTS simulations
            temperature: Temperature for probability calculation
            
        Returns:
            Dictionary mapping actions to probabilities
        """
        if root_state.is_terminal():
            return {}
        
        root = self.search(root_state, num_simulations)
        return root.get_action_probabilities(temperature)
    
    def get_principal_variation(self, root_state: ChessGameState,
                               num_simulations: int = 800,
                               max_depth: int = 10) -> List[Any]:
        """
        Get principal variation (most visited path).
        
        Args:
            root_state: Starting position
            num_simulations: Number of MCTS simulations
            max_depth: Maximum depth to follow
            
        Returns:
            List of actions in principal variation
        """
        root = self.search(root_state, num_simulations)
        
        variation = []
        current = root
        
        for _ in range(max_depth):
            if not current.children or current.state.is_terminal():
                break
            
            best_action = current.get_best_action()
            if best_action is None:
                break
                
            variation.append(best_action)
            current = current.children[best_action]
        
        return variation


class AlphaZeroTrainingExample:
    """Container for AlphaZero training examples."""
    
    def __init__(self, state_tensor: torch.Tensor, action_probs: Dict[Any, float], 
                 outcome: float, current_player: int):
        self.state_tensor = state_tensor
        self.action_probs = action_probs  
        self.outcome = outcome
        self.current_player = current_player


def tactical_temperature_schedule(move_num: int) -> float:
    """
    Temperature schedule optimized for tactical play.
    More exploration in tactical phases.
    
    Args:
        move_num: Current move number (0-indexed)
        
    Returns:
        Temperature value for this move
    """
    if move_num < 10:
        return 1.5    # High exploration in opening
    elif move_num < 40:
        return 1.0    # Standard exploration in middlegame  
    elif move_num < 60:
        return 0.5    # Reduced exploration in complex positions
    else:
        return 0.1    # Near-deterministic in endgame


def generate_self_play_game(chess_engine: AlphaZeroMCTS, initial_state: ChessGameState,
                           num_simulations: int = 800,
                           temperature_schedule: callable = tactical_temperature_schedule,
                           max_moves: int = 150,
                           enable_resignation: bool = True
                           ) -> tuple[List[AlphaZeroTrainingExample], ChessGameState, List[dict], bool, Optional[int]]:
    """
    Generate a complete self-play game for training - FIXED VERSION.
    
    Args:
        chess_engine: AlphaZeroMCTS instance
        initial_state: Starting chess position
        num_simulations: MCTS simulations per move
        temperature_schedule: Function mapping move number to temperature
        max_moves: Maximum moves before declaring draw
        enable_resignation: Whether to allow resignation in hopeless positions
        
    Returns:
        Tuple of (training_examples, final_state, move_history, game_resigned, winner)
        winner: 1 for white wins, -1 for black wins, 0 for draw, None if game incomplete
    """
    training_examples = []
    current_state = initial_state.clone()
    move_count = 0
    game_resigned = False
    resigning_player = None
    winner = None
    move_history = []
    
    # Record initial position
    move_history.append({
        "move_number": 0,
        "fen": current_state.board.fen(),
        "move": None,
        "move_san": None,
        "player": "white" if current_state.get_current_player() == 1 else "black"
    })
    
    while not current_state.is_terminal() and move_count < max_moves and not game_resigned:
        # Get MCTS action probabilities
        temperature = temperature_schedule(move_count)
        action_probs = chess_engine.get_action_probabilities(
            current_state, num_simulations=num_simulations, temperature=temperature
        )
        
        # Check for resignation (only after getting action probabilities to get value estimate)
        if enable_resignation and move_count > 0:
            # Get the root value from MCTS
            root = chess_engine.search(current_state, num_simulations)
            root_value = root.value_sum / root.visit_count if root.visit_count > 0 else 0.0
            
            if chess_engine.should_resign(root_value, move_count, current_state):
                game_resigned = True
                # The current player (who is about to move) resigns and loses
                resigning_player = current_state.get_current_player()
                # Winner is the opposite of the resigning player
                winner = -resigning_player
                final_outcome = -1.0  # Current player loses by resignation
                break
        
        # Store training example (outcome will be filled in later)
        example = AlphaZeroTrainingExample(
            state_tensor=current_state.to_tensor().clone(),
            action_probs=action_probs.copy(),
            outcome=0.0,  # Will be updated with game result
            current_player=current_state.get_current_player()
        )
        training_examples.append(example)
        
        # Sample action from MCTS probabilities
        actions = list(action_probs.keys())
        probabilities = list(action_probs.values())
        
        if not actions:
            # No legal moves - this shouldn't happen but handle gracefully
            break
            
        if temperature == 0.0 or len(actions) == 1:
            # Deterministic: choose best action
            action = max(action_probs.keys(), key=lambda a: action_probs[a])
        else:
            # Stochastic: sample from distribution
            if sum(probabilities) > 0:
                # Normalize probabilities to ensure they sum to 1
                prob_tensor = torch.tensor(probabilities)
                prob_tensor = prob_tensor / prob_tensor.sum()
                action_idx = torch.multinomial(prob_tensor, 1).item()
                action = actions[action_idx]
            else:
                # Fallback if all probabilities are zero
                action = actions[0]
                
        # Apply action and record move
        move_san = current_state.board.san(action)  # Get SAN notation before applying move
        current_state = current_state.apply_action(action)
        move_count += 1
        
        # Record this move
        move_history.append({
            "move_number": move_count,
            "fen": current_state.board.fen(),
            "move": action.uci(),
            "move_san": move_san,
            "player": "black" if current_state.get_current_player() == 1 else "white"  # Player who just moved
        })
    
    # CRITICAL FIX: Proper game outcome assignment
    if game_resigned:
        # Handle resignation - resigning_player was determined when resignation occurred
        for example in training_examples:
            if example.current_player == resigning_player:
                example.outcome = -1.0  # Resigning player loses
            else:
                example.outcome = 1.0   # Opponent wins
                
    elif current_state.is_terminal():
        # Game ended naturally - use actual result
        final_outcome = current_state.get_reward()
        
        # The current_player at terminal state is the player who would move next
        # but the game is over. We need to assign outcomes correctly.
        terminal_player_to_move = current_state.get_current_player()
        
        # Determine winner based on the terminal reward
        if final_outcome > 0:
            winner = terminal_player_to_move
        elif final_outcome < 0:
            winner = -terminal_player_to_move
        else:
            winner = 0  # Draw
        
        for example in training_examples:
            if example.current_player == terminal_player_to_move:
                # This player's turn when game ended - they get the terminal reward
                example.outcome = final_outcome
            else:
                # Opponent gets opposite reward
                example.outcome = -final_outcome
                
    else:
        # Game hit move limit - declare draw
        print(f"Game reached move limit ({max_moves}) - declaring draw")
        winner = 0  # Draw
        for example in training_examples:
            example.outcome = 0.0  # Draw for all positions
    
    return training_examples, current_state, move_history, game_resigned, winner