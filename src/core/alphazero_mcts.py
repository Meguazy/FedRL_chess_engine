"""
AlphaZero Monte Carlo Tree Search Implementation

This implementation follows AlphaZero's approach where:
1. Neural network provides move priors and position evaluation
2. PUCT (Polynomial UCT) replaces UCB1 for selection
3. No random rollouts - neural network evaluation at leaf nodes
4. Tree search guided by neural network policy

Based on Silver et al. (2017): "Mastering Chess and Shogi by Self-Play 
with a General Reinforcement Learning Algorithm"
"""

import math
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import logging

from .game_utils import ChessGameState

logger = logging.getLogger(__name__)


class AlphaZeroTrainingExample:
    """Container for AlphaZero training examples."""
    
    def __init__(self, state_tensor: torch.Tensor, action_probs: Dict[Any, float], 
                 outcome: float, current_player: int):
        self.state_tensor = state_tensor
        self.action_probs = action_probs  
        self.outcome = outcome
        self.current_player = current_player
    
    def __repr__(self):
        return f"AlphaZeroTrainingExample(outcome={self.outcome}, current_player={self.current_player})"


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
        Expand this node using neural network outputs with better error handling.
        """
        self.is_expanded = True
        self.neural_value = value
        
        if not legal_actions:
            logger.warning("Expanding node with no legal actions")
            return
        
        # Extract prior probabilities for legal actions
        total_prob = 0.0
        for action in legal_actions:
            try:
                action_idx = action_to_index_fn(action)
                if 0 <= action_idx < len(policy_probs):
                    prob = policy_probs[action_idx].item()
                    self.prior_probs[action] = prob
                    total_prob += prob
                else:
                    logger.warning(f"Action index {action_idx} out of bounds for policy size {len(policy_probs)}")
                    self.prior_probs[action] = 1e-8  # Small probability for invalid actions
            except Exception as e:
                logger.warning(f"Error processing action {action}: {e}")
                self.prior_probs[action] = 1e-8
        
        # Normalize priors over legal actions
        if total_prob > 1e-10:  # Use small epsilon instead of 0
            for action in self.prior_probs:
                self.prior_probs[action] /= total_prob
        else:
            # Uniform distribution if all legal actions have zero probability
            uniform_prob = 1.0 / len(legal_actions)
            for action in legal_actions:
                self.prior_probs[action] = uniform_prob
    
    def select_child(self) -> Tuple[Any, 'AlphaZeroNode']:
        """
        Select child using improved PUCT algorithm.
        """
        if not self.prior_probs:
            logger.warning("Selecting child from node with no prior probabilities")
            return None, None
        
        best_action = None
        best_value = float('-inf')
        
        sqrt_parent_visits = math.sqrt(max(self.visit_count, 1))  # Avoid sqrt(0)
        
        for action in self.prior_probs:
            if action in self.children:
                child = self.children[action]
                # Q(s,a): Average action value
                if child.visit_count > 0:
                    q_value = child.value_sum / child.visit_count
                else:
                    q_value = 0.0
                visit_count = child.visit_count
            else:
                q_value = 0.0  # Unvisited children start with Q=0
                visit_count = 0
            
            # PUCT exploration term
            prior_prob = self.prior_probs[action]
            exploration = self.c_puct * prior_prob * sqrt_parent_visits / (1 + visit_count)
            
            puct_value = q_value + exploration
            
            if puct_value > best_value:
                best_value = puct_value
                best_action = action
        
        if best_action is None:
            logger.warning("No action selected in PUCT")
            return None, None
        
        # Create child if it doesn't exist
        if best_action not in self.children:
            try:
                child_state = self.state.apply_action(best_action)
                self.children[best_action] = AlphaZeroNode(
                    child_state, parent=self, parent_action=best_action,
                    prior_prob=self.prior_probs[best_action]
                )
            except Exception as e:
                logger.error(f"Error creating child for action {best_action}: {e}")
                return None, None
        
        return best_action, self.children[best_action]
    
    def backup(self, value: float) -> None:
        """Backup value through this node."""
        self.visit_count += 1
        self.value_sum += value
    
    def get_q_value(self) -> float:
        """Get the Q-value (average value) of this node."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def get_action_probabilities(self, temperature: float = 1.0) -> Dict[Any, float]:
        """Get action probabilities with improved temperature handling."""
        if not self.children:
            return {}
        
        actions = list(self.children.keys())
        visit_counts = [self.children[action].visit_count for action in actions]
        
        if all(count == 0 for count in visit_counts):
            # All children unvisited - return uniform
            uniform_prob = 1.0 / len(actions)
            return {action: uniform_prob for action in actions}
        
        if temperature == 0:
            # Deterministic: choose action with most visits
            max_visits = max(visit_counts)
            probs = [1.0 if count == max_visits else 0.0 for count in visit_counts]
            # Normalize in case of ties
            prob_sum = sum(probs)
            if prob_sum > 0:
                probs = [p / prob_sum for p in probs]
        else:
            # Temperature scaling
            if temperature == 1.0:
                # Proportional to visit counts
                total_visits = sum(visit_counts)
                probs = [count / total_visits for count in visit_counts]
            else:
                # General temperature scaling with numerical stability
                scaled_counts = [count / temperature for count in visit_counts]
                max_scaled = max(scaled_counts) if scaled_counts else 0
                exp_counts = [math.exp(count - max_scaled) for count in scaled_counts]
                sum_exp = sum(exp_counts)
                
                if sum_exp > 0:
                    probs = [exp_count / sum_exp for exp_count in exp_counts]
                else:
                    # Fallback to uniform
                    probs = [1.0 / len(actions)] * len(actions)
        
        return {action: prob for action, prob in zip(actions, probs)}
    
    def get_best_action(self) -> Any:
        """Get action with highest visit count."""
        if not self.children:
            return None
        return max(self.children.keys(), 
                  key=lambda action: self.children[action].visit_count)


class AlphaZeroMCTS:
    """
    AlphaZero Monte Carlo Tree Search implementation.
    """
    
    def __init__(self, neural_network, c_puct: float = 1.25, device: str = 'cpu', 
                 resignation_threshold: float = -0.9, logger=None):
        """
        Initialize AlphaZero MCTS.
        
        Args:
            neural_network: AlphaZero neural network (policy + value)
            c_puct: PUCT exploration constant (1.25 is standard for chess)
            device: PyTorch device for neural network inference
            resignation_threshold: Value threshold for resignation
            logger: Logger instance for worker logging
        """
        self.neural_network = neural_network
        self.c_puct = c_puct
        self.device = device
        self.resignation_threshold = resignation_threshold
        self.logger = logger or logging.getLogger(__name__)
        
        # Set network to evaluation mode
        self.neural_network.eval()
    
    def search(self, root_state: ChessGameState, num_simulations: int) -> AlphaZeroNode:
        """
        Run MCTS search for specified number of simulations.
        """
        # Create root node
        root = AlphaZeroNode(root_state)
        
        # Expand root immediately if not terminal
        if not root_state.is_terminal():
            try:
                self._expand_node(root)
            except Exception as e:
                logger.error(f"Error expanding root node: {e}")
                return root
        
        # Run simulations
        successful_simulations = 0
        for i in range(num_simulations):
            try:
                self._simulate(root)
                successful_simulations += 1
            except Exception as e:
                logger.warning(f"Simulation {i} failed: {e}")
                # Continue with remaining simulations
                
        if successful_simulations == 0:
            logger.error("All MCTS simulations failed")
        
        return root
    
    def _simulate(self, root: AlphaZeroNode) -> None:
        """Single MCTS simulation with corrected backup logic."""
        # Phase 1: Selection - navigate to leaf
        path = []
        current = root
        
        while not current.is_leaf() and not current.state.is_terminal():
            action, child = current.select_child()
            if action is None or child is None:
                break
            path.append((current, action, child))
            current = child
        
        # Phase 2: Expansion & Evaluation
        if current.state.is_terminal():
            # Terminal node: use game outcome
            value = current.state.get_reward()
        else:
            if current.is_leaf():
                # Expand leaf node with neural network
                self._expand_node(current)
            # Use neural network value estimate
            value = current.neural_value
        
        # Phase 3: Backup - propagate value up tree with correct perspective
        current_value = value
        
        # Backup the leaf node first
        current.backup(current_value)
        
        # Backup through path (alternating perspective)
        for node, action, child in reversed(path):
            # Flip value for the parent (different player's perspective)
            current_value = -current_value
            node.backup(current_value)
    
    @torch.no_grad()
    def _expand_node(self, node: AlphaZeroNode) -> None:
        """Expand node using neural network evaluation with error handling."""
        try:
            # Convert state to neural network input
            state_tensor = node.state.to_tensor().unsqueeze(0).to(self.device)
            
            # Forward pass through neural network
            policy_logits, value = self.neural_network(state_tensor)
            
            # Convert outputs
            policy_probs = F.softmax(policy_logits, dim=-1).squeeze(0)
            value = torch.tanh(value).item()  # Ensure value is in [-1, 1]
            
            # Get legal actions
            legal_actions = node.state.get_legal_actions()
            
            if not legal_actions:
                logger.warning("No legal actions for node expansion")
                return
            
            # Expand node with neural network outputs
            node.expand(policy_probs, value, legal_actions, node.state.action_to_index)
            
        except Exception as e:
            logger.error(f"Error in node expansion: {e}")
            # Fallback: mark as expanded with uniform priors
            legal_actions = node.state.get_legal_actions()
            if legal_actions:
                node.is_expanded = True
                node.neural_value = 0.0
                uniform_prob = 1.0 / len(legal_actions)
                for action in legal_actions:
                    node.prior_probs[action] = uniform_prob
    
    def should_resign(self, value: float, move_count: int, current_state: ChessGameState = None) -> bool:
        """
        Check if should resign based on position evaluation.
        
        Args:
            value: Current position evaluation
            move_count: Number of moves played
            current_state: Current game state (optional)
            
        Returns:
            True if should resign, False otherwise
        """
        # Basic resignation logic
        if move_count < 25:  # Don't resign too early
            return False
        
        if value < self.resignation_threshold:
            if self.logger:
                player = 'white' if current_state and current_state.board.turn else 'black'
                self.logger.info(f"RESIGNATION: {player} resigns at move {move_count} (eval: {value:.3f})")
            return True
        
        return False
    
    def get_best_action(self, root_state: ChessGameState, 
                       num_simulations: int = 800) -> Any:
        """Get best action using MCTS search."""
        if root_state.is_terminal():
            return None
        
        root = self.search(root_state, num_simulations)
        return root.get_best_action()
    
    def get_action_probabilities(self, root_state: ChessGameState,
                                num_simulations: int = 800,
                                temperature: float = 1.0,
                                add_noise: bool = False,
                                dirichlet_alpha: float = 0.3) -> Dict[Any, float]:
        """Get action probabilities using MCTS search."""
        if root_state.is_terminal():
            return {}
        
        root = self.search(root_state, num_simulations)
        action_probs = root.get_action_probabilities(temperature)
        
        # Add Dirichlet noise for exploration if requested
        if add_noise and action_probs:
            action_probs = self._add_dirichlet_noise(action_probs, alpha=dirichlet_alpha)
        
        return action_probs
    
    def _add_dirichlet_noise(self, action_probs: Dict[Any, float], alpha: float = 0.3) -> Dict[Any, float]:
        """Add Dirichlet noise to action probabilities for exploration."""
        if not action_probs:
            return action_probs
        
        try:
            import numpy as np
            
            actions = list(action_probs.keys())
            probs = list(action_probs.values())
            
            # Generate Dirichlet noise
            noise = np.random.dirichlet([alpha] * len(actions))
            
            # Mix original probabilities with noise (75% original, 25% noise)
            mixed_probs = [0.75 * p + 0.25 * n for p, n in zip(probs, noise)]
            
            # Normalize
            total = sum(mixed_probs)
            if total > 0:
                mixed_probs = [p / total for p in mixed_probs]
            
            return {action: prob for action, prob in zip(actions, mixed_probs)}
            
        except Exception:
            # Return original probabilities if noise addition fails
            return action_probs
    
    def get_principal_variation(self, root_state: ChessGameState,
                               num_simulations: int = 800,
                               max_depth: int = 10) -> List[Any]:
        """Get principal variation (most visited path)."""
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


def generate_self_play_game(chess_engine: AlphaZeroMCTS, initial_state: ChessGameState,
                           num_simulations: int = 800,
                           temperature_schedule: callable = None,
                           max_moves: int = 150,
                           enable_resignation: bool = True,
                           filter_draws: bool = True,
                           style: str = 'standard',
                           dirichlet_alpha: float = 0.3
                           ) -> tuple[List[AlphaZeroTrainingExample], ChessGameState, List[dict], bool, Optional[int]]:
    """
    Generate a complete self-play game for training.
    
    Args:
        chess_engine: AlphaZeroMCTS instance
        initial_state: Starting chess position
        num_simulations: MCTS simulations per move
        temperature_schedule: Function mapping move number to temperature
        max_moves: Maximum moves before declaring draw
        enable_resignation: Whether to allow resignation
        filter_draws: Whether to filter out drawn games
        style: Playing style for temperature schedule
        dirichlet_alpha: Dirichlet concentration parameter
        
    Returns:
        Tuple of (training_examples, final_state, move_history, game_resigned, winner)
    """
    # Default temperature schedule
    if temperature_schedule is None:
        if style == 'tactical':
            temperature_schedule = lambda move: 1.2 if move < 20 else (0.8 if move < 40 else 0.0)
        elif style == 'positional':
            temperature_schedule = lambda move: 1.0 if move < 15 else (0.5 if move < 25 else 0.0)
        elif style == 'dynamic':
            temperature_schedule = lambda move: 1.1 if move < 25 else (0.7 if move < 50 else 0.0)
        else:  # standard
            temperature_schedule = lambda move: 1.0 if move < 30 else 0.0
    
    training_examples = []
    current_state = initial_state.clone()
    move_count = 0
    game_resigned = False
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
        
        # Add noise during training for better exploration
        add_noise = move_count < 30  # Add noise for first 30 moves
        action_probs = chess_engine.get_action_probabilities(
            current_state, num_simulations=num_simulations, 
            temperature=temperature, add_noise=add_noise,
            dirichlet_alpha=dirichlet_alpha
        )
        
        # Check for resignation
        if enable_resignation and move_count > 0:
            root = chess_engine.search(current_state, num_simulations)
            root_value = root.value_sum / root.visit_count if root.visit_count > 0 else 0.0
            
            if chess_engine.should_resign(root_value, move_count, current_state):
                game_resigned = True
                winner = -current_state.get_current_player()  # Opponent wins
                break
        
        # Store training example
        example = AlphaZeroTrainingExample(
            state_tensor=current_state.to_tensor().clone(),
            action_probs=action_probs.copy(),
            outcome=0.0,  # Will be updated with game result
            current_player=current_state.get_current_player()
        )
        training_examples.append(example)
        
        # Sample action from probabilities
        if temperature == 0.0:
            # Deterministic: choose best action
            action = max(action_probs.keys(), key=lambda a: action_probs[a])
        else:
            # Stochastic: sample from distribution
            actions = list(action_probs.keys())
            probabilities = list(action_probs.values())
            action_idx = torch.multinomial(torch.tensor(probabilities), 1).item()
            action = actions[action_idx]
        
        if action is None:
            break
                
        # Apply action and record move
        move_san = current_state.board.san(action)
        current_state = current_state.apply_action(action)
        move_count += 1
        
        # Record this move
        move_history.append({
            "move_number": move_count,
            "fen": current_state.board.fen(),
            "move": action.uci(),
            "move_san": move_san,
            "player": "black" if current_state.get_current_player() == 1 else "white"
        })
    
    # Determine game outcome
    if game_resigned:
        final_outcome = -1.0 if winner == current_state.get_current_player() else 1.0
    elif current_state.is_terminal():
        final_outcome = current_state.get_reward()
        if current_state.board.outcome():
            if current_state.board.outcome().winner is None:
                winner = 0  # Draw
            elif current_state.board.outcome().winner:
                winner = 1  # White wins
            else:
                winner = -1  # Black wins
        else:
            winner = 0  # Draw
    else:
        # Game hit move limit - declare draw
        final_outcome = 0.0
        winner = 0
    
    # Update all training examples with game outcome
    for example in training_examples:
        if example.current_player == current_state.get_current_player():
            example.outcome = final_outcome
        else:
            example.outcome = -final_outcome
    
    # Filter draws if requested
    if filter_draws and abs(final_outcome) < 0.5:
        training_examples = []  # Remove drawn games
    
    return training_examples, current_state, move_history, game_resigned, winner


# Make sure all classes are available for import
__all__ = [
    'AlphaZeroMCTS',
    'AlphaZeroNode', 
    'AlphaZeroTrainingExample',
    'generate_self_play_game'
]