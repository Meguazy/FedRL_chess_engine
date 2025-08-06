"""
Monte Carlo Tree Search (MCTS) is a heuristic search algorithm used for decision-making processes, particularly in game playing. It combines the precision of tree search with the generality of random sampling, making it effective for large search spaces.

REFERENCES:
Tier 1 (Must Cite):
PAPER: Kocsis & Szepesvári (2006) - Original MCTS
PAPER: Silver et al. (2018) - AlphaZero (your direct source)
PAPER: Silver et al. (2017) - AlphaGo Zero (foundation)

Tier 2 (Should Cite):
PAPER: Silver et al. (2016) - AlphaGo (historical context)
PAPER: Browne et al. (2012) - MCTS Survey (comprehensive overview)
PAPER: Auer et al. (2002) - UCB1 (mathematical foundation)

Tier 3 (Nice to Have):
PAPER: Chess-specific MCTS papers (domain relevance)
PAPER: MCTS convergence theory (theoretical rigor)
"""

import math
import random
import time
import logging

from typing import List, Optional, Tuple, Dict, Any
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GameState(ABC):
    """
    Abstract base class for game states.
    
    Subclasses must implement methods to get legal actions, apply actions, and check terminal states.
    """
    
    @abstractmethod
    def get_legal_actions(self) -> List[Any]:
        """Return a list of legal actions for the current state."""
        pass

    @abstractmethod
    def apply_action(self, action: Any) -> 'GameState':
        """Return a new GameState after applying the given action."""
        pass

    @abstractmethod
    def is_terminal(self) -> bool:
        """Check if the current state is terminal (game over)."""
        pass

    @abstractmethod
    def get_reward(self) -> float:
        """Return the reward for the current state."""
        pass

    @abstractmethod
    def get_current_player(self) -> int:
        """Return the index of the current player."""
        pass

    @abstractmethod
    def clone(self) -> 'GameState':
        """Return a deep copy of the current game state."""
        pass


class MCTSNode:
    """
    Node in the MCTS tree.
    
    Each node stores:
    - visit_count (n): Number of times this node has been visited
    - value_sum (w): Sum of all rewards backpropagated through this node
    - parent: Parent node in the tree
    - children: Dictionary mapping actions to child nodes
    - untried_actions: Actions not yet expanded from this node
    """

    def __init__(
        self,
        state: GameState,
        parent: Optional['MCTSNode'] = None,
        parent_action: Optional[Any] = None
    ):
        self.state = state
        self.parent = parent
        self.parent_action = parent_action

        # Initialize visit count and value sum
        self.visit_count = 0
        self.value_sum = 0.0

        # Initialize children and untried actions
        self.children: Dict[Any, 'MCTSNode'] = {}
        self.untried_actions: List[Any] = state.get_legal_actions()

    def is_fully_expanded(self) -> bool:
        """Return True if all actions have been tried, False otherwise."""
        return len(self.untried_actions) == 0
    
    def is_terminal(self) -> bool:
        """Check if the node's state is terminal."""
        return self.state.is_terminal()

    def ucb1_value(self, exploration_constant: float = math.sqrt(2)) -> float:
        """
        Calculate UCB1 value for this node.
        
        UCB1 formula from Auer et al. (2002):
        UCB1(i) = X̄ᵢ + c√(ln(n)/nᵢ)
        
        Where:
        - X̄ᵢ is the average reward for action i
        - c is the exploration constant (√2 theoretically optimal)
        - n is the total number of plays
        - nᵢ is the number of times action i has been played
        """
        # If the node has never been visited, return infinity to prioritize exploration
        if self.visit_count == 0:
            return float('inf')
        
        if self.parent is None:
            # If this is the root node, we don't have a parent visit count
            return self.value_sum / self.visit_count
        
        exploitation = self.value_sum / self.visit_count
        exploration = exploration_constant * math.sqrt(
            math.log(self.parent.visit_count) / self.visit_count
        )

        return exploitation + exploration
    
    def add_child(self, action: Any, child_state: GameState) -> 'MCTSNode':
        """Add a child node for the given action."""
        child_node = MCTSNode(child_state, parent=self, parent_action=action)
        self.children[action] = child_node
        self.untried_actions.remove(action)
        return child_node
    
    def update(self, reward: float):
        """Update the node with the given reward."""
        self.visit_count += 1
        self.value_sum += reward

    def select_child(self, exploration_constant: float = math.sqrt(2)) -> 'MCTSNode':
        """
        Select the child node with the highest UCB1 value.
        
        Args:
            exploration_constant: Exploration constant for UCB1 (default: √2)
        
        Returns:
            The child node with the highest UCB1 value.
        """
        return max(self.children.values(), 
                key=lambda child: child.ucb1_value(exploration_constant))
    
    def select_best_action(self) -> Any:
        """
        Select the action leading to the child with the highest visit count.
        
        Returns:
            The action corresponding to the child node with the highest visit count.
        """
        return max(self.children.keys(), 
                   key=lambda action: self.children[action].visit_count)
    
    def get_action_probabilities(self, temperature: float = 1.0) -> Dict[Any, float]:
        """
        Get action probabilities based on visit counts.
        
        With temperature = 0: Returns deterministic policy (max visits)
        With temperature > 0: Returns softmax over visit counts
        """
        if not self.children:
            return {}
        
        if temperature == 0:
            # Deterministic policy: choose action with max visits
            best_action = self.select_best_action()
            return {action: 1.0 if action == best_action else 0.0 
                    for action in self.children.keys()}
        else:
            # Softmax probabilities with temperature
            visit_counts = [child.visit_count for child in self.children.values()]
            scaled_counts = [count / temperature for count in visit_counts]

            # Numerical stability: subtract max for softmax
            max_scaled = max(scaled_counts)
            exp_counts = [math.exp(count - max_scaled) for count in scaled_counts]
            sum_exp = sum(exp_counts)

            probabilities = [exp_count / sum_exp for exp_count in exp_counts]

            return {action: prob for action, prob in zip(self.children.keys(), probabilities)}


class MCTS:
    """
    Monte Carlo Tree Search implementation.
    
    Follows the UCT (Upper Confidence bounds applied to Trees) algorithm
    from Kocsis & Szepesvári (2006).
    """
    
    def __init__(
        self,
        exploration_constant: float = math.sqrt(2),
        max_rollout_depth: int = 100
    ):
        """
        Run MCTS for given number of iterations or time limit.
        
        Args:
            root_state: Starting game state
            iterations: Number of MCTS iterations to run
            time_limit: Optional time limit in seconds
            
        Returns:
            Root node of the search tree
        """
        self.exploration_constant = exploration_constant
        self.max_rollout_depth = max_rollout_depth
    
    def search(
        self,
        root_state: GameState,
        iterations: int,
        time_limit: Optional[float] = None
    ) -> MCTSNode:
        """
        Run MCTS for given number of iterations or time limit.
        
        Args:
            root_state: Starting game state
            iterations: Number of MCTS iterations to run
            time_limit: Optional time limit in seconds
            
        Returns:
            Root node of the search tree
        """
        root = MCTSNode(root_state)
        start_time = time.time()

        for i in range(iterations):
            # Check time limit
            if time_limit is not None and (time.time() - start_time) >= time_limit:
                break

            # Single MCTS iteration
            self._mcts_iteration(root)

        return root
    
    def _mcts_iteration(self, root: MCTSNode):
        """
        Single MCTS iteration following the four phases:
        1. Selection: Navigate tree using UCB1
        2. Expansion: Add new node to tree
        3. Simulation: Random rollout from new node
        4. Backpropagation: Update statistics
        """
        # Phase 1: Selection
        node = self._select(root)

        # Phase 2: Expansion
        if not node.is_terminal() and node.visit_count > 0:
            node = self._expand(node)

        # Phase 3: Simulation
        reward = self._simulate(node)

        # Phase 4: Backpropagation
        self._backpropagate(node, reward)

    def _select(self, node: MCTSNode) -> MCTSNode:
        """
        Select a node using UCB1 until reaching a leaf node.
        
        Args:
            node: Current MCTSNode
        
        Returns:
            Leaf node selected for expansion
        """
        while not node.is_terminal() and node.is_fully_expanded():
            node = node.select_child(self.exploration_constant)
        return node
    
    def _expand(self, node: MCTSNode) -> MCTSNode:
        """
        Expand the node by adding a new child for one of the untried actions.
        
        Args:
            node: Current MCTSNode
        
        Returns:
            Newly created child node
        """
        if node.untried_actions:
            action = random.choice(node.untried_actions)
            child_state = node.state.apply_action(action)
            return node.add_child(action, child_state)
        return node
    
    def _simulate(self, node: MCTSNode) -> float:
        """
        Simulation phase: Random rollout from the given node until
        reaching a terminal state.
        
        Returns the reward from the perspective of the player who was
        to move at the root of the search.
        """
        current_state = node.state.clone()
        depth = 0

        # Random rollout until terminal state or max depth
        while not current_state.is_terminal() and depth < self.max_rollout_depth:
            legal_actions = current_state.get_legal_actions()
            if not legal_actions:
                break

            action = random.choice(legal_actions)
            current_state = current_state.apply_action(action)
            depth += 1

        # Return the reward for the player who was to move at the root
        if current_state.is_terminal():
            return current_state.get_reward(node.state.get_current_player())
        else:
            # If we hit max depth without terminal state, return 0 reward
            return 0.0
        
    def _backpropagate(self, node: MCTSNode, reward: float):
        """
        Backpropagation phase: Update visit counts and value sums
        for all nodes along the path from the given node to the root.
        
        The reward is negated at each level since players alternate.
        
        Args:
            node: Leaf node where the simulation ended
            reward: Reward to backpropagate
        """
        while node is not None:
            node.update(reward)
            reward = -reward
            node = node.parent

    def get_best_action(
            self, 
            root_state: GameState, 
            iterations: int = 1000,
            time_limit: Optional[float] = None
        ) -> Any:
        """
        Get the best action from the given state using MCTS.
        
        Args:
            root_state: Game state to analyze
            iterations: Number of MCTS iterations
            time_limit: Optional time limit in seconds
            
        Returns:
            Best action according to MCTS
        """
        if root_state.is_terminal():
            return None
        
        root = self.search(root_state, iterations, time_limit)

        if not root.children:
            # No expansions happened, return random legal action
            legal_actions = root_state.get_legal_actions()
            return random.choice(legal_actions) if legal_actions else None
        
        return root.select_best_action()
    
    def get_action_probabilities(
            self, root_state: GameState, 
            iterations: int = 1000,
            temperature: float = 1.0,
            time_limit: Optional[float] = None
        ) -> Dict[Any, float]:
        """
        Get action probabilities from the given state using MCTS.
        
        Args:
            root_state: Game state to analyze
            iterations: Number of MCTS iterations
            temperature: Temperature for probability calculation
            time_limit: Optional time limit in seconds
            
        Returns:
            Dictionary mapping actions to probabilities
        """
        if root_state.is_terminal():
            return {}
        
        root = self.search(root_state, iterations, time_limit)
        return root.get_action_probabilities(temperature)
    
    def get_principal_variation(
            self, 
            root_state: GameState, 
            iterations: int = 1000,
            depth: int = 10
        ) -> List[Any]:
        """
        Get the principal variation (most likely sequence of moves) up to given depth.
        
        Args:
            root_state: Starting game state
            iterations: Number of MCTS iterations
            depth: Maximum depth of variation to return
            
        Returns:
            List of actions representing the principal variation
        """
        root = self.search(root_state, iterations)
        variation = []

        current_node = root
        for _ in range(depth):
            if not current_node.children or current_node.is_terminal():
                break

            best_action = current_node.select_best_action()
            variation.append(best_action)
            current_node = current_node.children[best_action]

        return variation
    

class MCTSStats:
    """Utility class for collecting and analyzing MCTS statistics."""
    
    @staticmethod
    def print_tree_stats(root: MCTSNode, max_depth: int = 3):
        """Print statistics about the MCTS tree."""
        logger.info(f"Root visits: {root.visit_count}")
        logger.info(f"Root value: {root.value_sum / root.visit_count if root.visit_count > 0 else 0:.3f}")
        logger.info(f"Children: {len(root.children)}")

        if root.children and max_depth > 0:
            logger.info("\nTop children:")
            sorted_children = sorted(root.children.items(), 
                                   key=lambda x: x[1].visit_count, reverse=True)
            
            for i, (action, child) in enumerate(sorted_children[:5]):
                avg_value = child.value_sum / child.visit_count if child.visit_count > 0 else 0
                ucb1_val = child.ucb1_value()
                logger.info(f"  {i+1}. Action: {action}, Visits: {child.visit_count}, "
                             f"Value: {avg_value:.3f}, UCB1: {ucb1_val:.3f}")

    @staticmethod
    def get_tree_depth(root: MCTSNode) -> int:
        """Get maximum depth of the MCTS tree."""
        if not root.children:
            return 0
        return 1 + max(MCTSStats.get_tree_depth(child) for child in root.children.values())
    
    @staticmethod
    def get_tree_size(root: MCTSNode) -> int:
        """Get total number of nodes in the MCTS tree."""
        size = 1
        for child in root.children.values():
            size += MCTSStats.get_tree_size(child)
        return size