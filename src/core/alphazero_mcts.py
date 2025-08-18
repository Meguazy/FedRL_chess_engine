"""
AlphaZero Monte Carlo Tree Search Implementation - DEDUPLICATED VERSION

This version removes all duplicated logic by using centralized training utilities.
All shared functionality has been moved to training_utils.py.

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
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.game_utils import ChessGameState

# Use lazy imports to avoid circular dependency
def _get_training_utilities():
    """Lazy import of training utilities to avoid circular imports."""
    from src.training.training_utils import TrainingUtilities
    return TrainingUtilities

def _get_temperature_schedules():
    """Lazy import of temperature schedules to avoid circular imports."""
    from src.training.training_utils import TemperatureSchedules
    return TemperatureSchedules

def _get_resignation_logic():
    """Lazy import of resignation logic to avoid circular imports."""
    from src.training.training_utils import ResignationLogic
    return ResignationLogic

def _get_game_outcome_analyzer():
    """Lazy import of game outcome analyzer to avoid circular imports."""
    from src.training.training_utils import GameOutcomeAnalyzer
    return GameOutcomeAnalyzer


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
    AlphaZero Monte Carlo Tree Search - DEDUPLICATED VERSION
    
    Uses centralized utilities from training_utils.py to eliminate duplication.
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
        Single MCTS simulation:
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
            # Terminal reward from correct perspective
            raw_reward = current.state.get_reward()
            value = raw_reward  # Use the reward as-is since get_reward() should handle perspective
        else:
            if current.is_leaf():
                # Expand leaf node with neural network
                self._expand_node(current)
            # Use neural network value estimate (already from correct perspective)
            value = current.neural_value
        
        # Phase 3: Backup - propagate value up tree
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
        
        # Apply tanh since it was removed from neural network forward()
        value = torch.tanh(value).item()  # Scalar value in [-1, 1]
        
        # Get legal actions
        legal_actions = node.state.get_legal_actions()
        
        # Expand node with neural network outputs
        node.expand(policy_probs, value, legal_actions, node.state.action_to_index)
    
    def should_resign(self, value: float, move_count: int, current_state: ChessGameState = None) -> bool:
        """
        DEDUPLICATED: Use centralized resignation logic.
        
        Args:
            value: Current position evaluation
            move_count: Number of moves played
            current_state: Current game state (optional, for material evaluation)
            
        Returns:
            True if should resign, False otherwise
        """
        resignation_logic = _get_resignation_logic()
        should_resign, reason = resignation_logic.should_resign_combined(
            value=value,
            board=current_state.board if current_state else None,
            move_count=move_count,
            eval_threshold=self.resignation_threshold,
            material_threshold=300,
            min_moves=25
        )
        
        if should_resign and self.logger:
            player = 'white' if current_state and current_state.board.turn else 'black'
            self.logger.info(f"RESIGNATION: {player} resigns at move {move_count} (reason: {reason})")
            if current_state:
                self.logger.info(f"RESIGNATION: FEN: {current_state.board.fen()}")
        
        return should_resign
    
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
                                temperature: float = 1.0,
                                add_noise: bool = False,
                                dirichlet_alpha: float = 0.3) -> Dict[Any, float]:
        """
        Get action probabilities using MCTS search.
        
        Args:
            root_state: Chess position to analyze
            num_simulations: Number of MCTS simulations
            temperature: Temperature for probability calculation
            add_noise: Whether to add Dirichlet noise for exploration (training only)
            dirichlet_alpha: Dirichlet concentration parameter for noise
            
        Returns:
            Dictionary mapping actions to probabilities
        """
        if root_state.is_terminal():
            return {}
        
        root = self.search(root_state, num_simulations)
        action_probs = root.get_action_probabilities(temperature)
        
        # DEDUPLICATED: Use centralized Dirichlet noise with configurable alpha
        if add_noise and action_probs:
            training_utils = _get_training_utilities()
            action_probs = training_utils.add_dirichlet_noise(action_probs, alpha=dirichlet_alpha)
        
        return action_probs
    
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
    Generate a complete self-play game for training - DEDUPLICATED VERSION.
    
    Args:
        chess_engine: AlphaZeroMCTS instance
        initial_state: Starting chess position
        num_simulations: MCTS simulations per move
        temperature_schedule: Function mapping move number to temperature (if None, uses style-based schedule)
        max_moves: Maximum moves before declaring draw
        enable_resignation: Whether to allow resignation in hopeless positions
        filter_draws: Whether to filter out drawn games from training data
        style: Playing style for temperature schedule ('tactical', 'positional', 'dynamic', 'standard')
        dirichlet_alpha: Dirichlet concentration parameter for exploration noise
        
    Returns:
        Tuple of (training_examples, final_state, move_history, game_resigned, winner)
    """
    # DEDUPLICATED: Use centralized temperature schedule
    if temperature_schedule is None:
        temp_schedules = _get_temperature_schedules()
        temperature_schedule = temp_schedules.get_schedule_for_style(style)
    
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
        
        # Add noise during training for better exploration
        add_noise = move_count < 30  # Add noise for first 30 moves
        action_probs = chess_engine.get_action_probabilities(
            current_state, num_simulations=num_simulations, 
            temperature=temperature, add_noise=add_noise,
            dirichlet_alpha=dirichlet_alpha  # Pass through the alpha parameter
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
        
        # DEDUPLICATED: Use centralized action sampling
        training_utils = _get_training_utilities()
        action = training_utils.sample_action_from_probabilities(action_probs, temperature)
        
        if action is None:
            # No legal moves - this shouldn't happen but handle gracefully
            break
                
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
    
    # DEDUPLICATED: Use centralized game outcome analysis
    game_analyzer = _get_game_outcome_analyzer()
    if game_resigned:
        game_result = game_analyzer.create_resignation_result(
            resigning_player, move_count, current_state.board.fen()
        )
    elif current_state.is_terminal():
        game_result = game_analyzer.analyze_terminal_position(current_state.board)
    else:
        # Game hit move limit - declare draw
        from src.training.training_utils import GameResult
        game_result = GameResult(
            winner=0,
            outcome_type="move_limit",
            move_count=move_count,
            final_fen=current_state.board.fen()
        )
    
    # DEDUPLICATED: Use centralized outcome assignment
    game_analyzer = _get_game_outcome_analyzer()
    game_analyzer.assign_training_outcomes(training_examples, game_result)
    winner = game_result.winner
    
    # DEDUPLICATED: Use centralized filtering
    if filter_draws:
        from src.training.training_utils import TrainingExampleFilters
        training_examples = TrainingExampleFilters.filter_decisive_games(training_examples)
    
    return training_examples, current_state, move_history, game_resigned, winner