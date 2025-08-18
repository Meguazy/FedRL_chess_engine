"""
Optimized AlphaZero Monte Carlo Tree Search Implementation for RTX 4090

Key optimizations:
1. Batched neural network inference
2. Mixed precision for faster GPU computation
3. Memory-efficient operations
4. Parallel simulation support
5. Optimized tensor operations
"""

import math
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import logging
import numpy as np
from collections import deque

from .game_utils import ChessGameState, should_resign_material

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
    Optimized AlphaZero MCTS node with virtual loss support.
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
        self.virtual_loss = 0  # For parallel search
        
        # Tree structure  
        self.children: Dict[Any, 'AlphaZeroNode'] = {}
        self.prior_probs: Dict[Any, float] = {}
        
        # Neural network outputs
        self.is_expanded = False
        self.neural_value = 0.0
    
    def is_leaf(self) -> bool:
        return not self.is_expanded
    
    def expand(self, policy_probs: torch.Tensor, value: float, legal_actions: List[Any],
               action_to_index_fn) -> None:
        """Optimized expansion with vectorized operations."""
        self.is_expanded = True
        self.neural_value = value
        
        if not legal_actions:
            return
        
        # Vectorized prior extraction
        action_indices = []
        valid_actions = []
        
        for action in legal_actions:
            try:
                idx = action_to_index_fn(action)
                if 0 <= idx < len(policy_probs):
                    action_indices.append(idx)
                    valid_actions.append(action)
            except:
                continue
        
        if not valid_actions:
            return
        
        # Batch extract probabilities
        if isinstance(policy_probs, torch.Tensor):
            indices_tensor = torch.tensor(action_indices, device=policy_probs.device)
            probs = policy_probs[indices_tensor]
            
            # Normalize
            prob_sum = probs.sum().item()
            if prob_sum > 1e-10:
                probs = probs / prob_sum
            else:
                probs = torch.ones_like(probs) / len(probs)
            
            # Store probabilities
            for action, prob in zip(valid_actions, probs):
                self.prior_probs[action] = prob.item()
        else:
            # Fallback for non-tensor input
            total_prob = 0.0
            for action in valid_actions:
                try:
                    idx = action_to_index_fn(action)
                    prob = policy_probs[idx].item() if hasattr(policy_probs[idx], 'item') else policy_probs[idx]
                    self.prior_probs[action] = prob
                    total_prob += prob
                except:
                    self.prior_probs[action] = 1e-8
            
            # Normalize
            if total_prob > 1e-10:
                for action in self.prior_probs:
                    self.prior_probs[action] /= total_prob
            else:
                uniform_prob = 1.0 / len(valid_actions)
                for action in valid_actions:
                    self.prior_probs[action] = uniform_prob
    
    def select_child(self, c_puct: float) -> Tuple[Any, 'AlphaZeroNode']:
        """Optimized child selection with virtual loss support."""
        if not self.prior_probs:
            return None, None
        
        best_action = None
        best_value = float('-inf')
        
        # Pre-compute square root term
        sqrt_parent_visits = math.sqrt(max(self.visit_count + self.virtual_loss, 1))
        
        for action in self.prior_probs:
            if action in self.children:
                child = self.children[action]
                effective_visits = child.visit_count + child.virtual_loss
                
                if effective_visits > 0:
                    q_value = (child.value_sum - child.virtual_loss) / effective_visits
                else:
                    q_value = 0.0
                visit_count = effective_visits
            else:
                q_value = 0.0
                visit_count = 0
            
            # PUCT formula
            prior_prob = self.prior_probs[action]
            exploration = c_puct * prior_prob * sqrt_parent_visits / (1 + visit_count)
            puct_value = q_value + exploration
            
            if puct_value > best_value:
                best_value = puct_value
                best_action = action
        
        if best_action is None:
            return None, None
        
        # Create child if needed
        if best_action not in self.children:
            try:
                child_state = self.state.apply_action(best_action)
                self.children[best_action] = AlphaZeroNode(
                    child_state, parent=self, parent_action=best_action,
                    prior_prob=self.prior_probs[best_action]
                )
            except:
                return None, None
        
        return best_action, self.children[best_action]
    
    def add_virtual_loss(self):
        """Add virtual loss for parallel search."""
        self.virtual_loss += 1
    
    def remove_virtual_loss(self):
        """Remove virtual loss after backup."""
        self.virtual_loss = max(0, self.virtual_loss - 1)
    
    def backup(self, value: float):
        """Optimized backup with virtual loss handling."""
        self.visit_count += 1
        self.value_sum += value
        self.remove_virtual_loss()
    
    def get_q_value(self) -> float:
        """Get the Q-value (average value) of this node."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def get_action_probabilities(self, temperature: float = 1.0) -> Dict[Any, float]:
        """Optimized probability calculation with numpy acceleration."""
        if not self.children:
            return {}
        
        actions = list(self.children.keys())
        visit_counts = np.array([self.children[action].visit_count for action in actions])
        
        if np.all(visit_counts == 0):
            uniform_prob = 1.0 / len(actions)
            return {action: uniform_prob for action in actions}
        
        if temperature == 0:
            # Deterministic selection
            max_visits = visit_counts.max()
            probs = (visit_counts == max_visits).astype(float)
            probs = probs / probs.sum()
        else:
            # Temperature scaling
            if temperature == 1.0:
                probs = visit_counts / visit_counts.sum()
            else:
                scaled_counts = visit_counts / temperature
                max_scaled = scaled_counts.max()
                exp_counts = np.exp(scaled_counts - max_scaled)
                probs = exp_counts / exp_counts.sum()
        
        return {action: prob for action, prob in zip(actions, probs)}
    
    def get_best_action(self) -> Any:
        """Get action with highest visit count."""
        if not self.children:
            return None
        return max(self.children.keys(), 
                  key=lambda action: self.children[action].visit_count)


class AlphaZeroMCTS:
    """
    GPU-optimized AlphaZero MCTS with batched inference.
    """
    
    def __init__(self, neural_network, c_puct: float = 1.25, device: str = 'cuda', 
                 resignation_threshold: float = -0.9, resignation_centipawns: int = -500, 
                 logger=None, batch_size: int = 128, use_mixed_precision: bool = True):  # Increased default batch_size
        
        self.neural_network = neural_network
        self.c_puct = c_puct
        self.device = device
        self.resignation_threshold = resignation_threshold
        self.resignation_centipawns = resignation_centipawns
        self.logger = logger or logging.getLogger(__name__)
        self.batch_size = batch_size  # Much larger batches for RTX 4090
        self.use_mixed_precision = use_mixed_precision
        
        # Apply AGGRESSIVE GPU optimizations for maximum utilization
        self._apply_aggressive_optimizations()
        
        # Set network to evaluation mode
        self.neural_network.eval()
    
    def _apply_aggressive_optimizations(self):
        """Apply AGGRESSIVE RTX 4090 optimizations for maximum GPU utilization."""
        if torch.cuda.is_available():
            # AGGRESSIVE memory settings - use almost all VRAM
            torch.cuda.set_per_process_memory_fraction(0.95)
            torch.cuda.empty_cache()
            
            # MAXIMUM performance settings
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # Enable TF32 for RTX 4090 (new API)
            torch.backends.cudnn.conv.fp32_precision = 'tf32'
            torch.backends.cuda.matmul.fp32_precision = 'tf32'
            
            # AGGRESSIVE compilation for maximum speed
            try:
                self.neural_network = torch.compile(
                    self.neural_network, 
                    mode="max-autotune",  # Most aggressive optimization
                    fullgraph=True       # Compile entire graph
                )
                print("🚀 Model compiled with max-autotune for RTX 4090")
            except Exception as e:
                print(f"⚠️ Advanced compile failed: {e}, using reduce-overhead")
                try:
                    self.neural_network = torch.compile(
                        self.neural_network,
                        mode="reduce-overhead"
                    )
                    print("✅ Model compiled with reduce-overhead")
                except:
                    print("❌ Compilation failed, using standard model")
            
            # Pre-allocate GPU memory to avoid fragmentation
            try:
                dummy_input = torch.randn(self.batch_size, 119, 8, 8).to(self.device)
                with torch.no_grad():
                    _ = self.neural_network(dummy_input)
                del dummy_input
                torch.cuda.empty_cache()
                print(f"✅ Pre-allocated GPU memory for batch size {self.batch_size}")
            except Exception as e:
                print(f"⚠️ Memory pre-allocation failed: {e}")
        
        print(f"🔥 AGGRESSIVE optimizations applied for RTX 4090")
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"📊 Using RTX 4090 with {total_memory:.1f}GB VRAM")
    
    def search(self, root_state: ChessGameState, num_simulations: int) -> AlphaZeroNode:
        """Optimized search with aggressive batching for high GPU utilization."""
        root = AlphaZeroNode(root_state)
        
        # Pre-expand root
        if not root_state.is_terminal():
            self._expand_nodes_batched([root])
        
        # AGGRESSIVE BATCHING: Process simulations in large batches
        # This is key to achieving 80-90% GPU utilization
        large_batch_size = min(64, num_simulations)  # Much larger batches
        num_batches = (num_simulations + large_batch_size - 1) // large_batch_size
        
        for batch_idx in range(num_batches):
            current_batch_size = min(large_batch_size, num_simulations - batch_idx * large_batch_size)
            
            # Collect many leaf nodes at once
            all_paths_and_leaves = []
            for _ in range(current_batch_size):
                path, leaf = self._select_path(root)
                if path is not None and leaf is not None:
                    all_paths_and_leaves.append((path, leaf))
            
            # Batch expand ALL leaves at once (this hits GPU hard)
            leaves_to_expand = []
            for path, leaf in all_paths_and_leaves:
                if not leaf.state.is_terminal() and leaf.is_leaf():
                    leaves_to_expand.append(leaf)
            
            if leaves_to_expand:
                self._expand_nodes_batched(leaves_to_expand)
            
            # Backup all paths
            for path, leaf in all_paths_and_leaves:
                if leaf.state.is_terminal():
                    value = leaf.state.get_reward()
                else:
                    value = leaf.neural_value
                self._backup_path(path, leaf, value)
        
        return root
    
    def _run_simulation_batch(self, root: AlphaZeroNode, batch_size: int):
        """Run multiple simulations with batched neural network calls."""
        
        # Phase 1: Selection - collect paths to leaves
        paths_and_leaves = []
        for _ in range(batch_size):
            path, leaf = self._select_path(root)
            if path is not None and leaf is not None:
                paths_and_leaves.append((path, leaf))
        
        # Phase 2: Batch expansion
        leaves_to_expand = []
        for path, leaf in paths_and_leaves:
            if not leaf.state.is_terminal() and leaf.is_leaf():
                leaves_to_expand.append(leaf)
        
        if leaves_to_expand:
            self._expand_nodes_batched(leaves_to_expand)
        
        # Phase 3: Backup
        for path, leaf in paths_and_leaves:
            if leaf.state.is_terminal():
                value = leaf.state.get_reward()
            else:
                value = leaf.neural_value
            self._backup_path(path, leaf, value)
    
    def _select_path(self, root: AlphaZeroNode) -> Tuple[List, AlphaZeroNode]:
        """Select path to leaf using PUCT."""
        path = []
        current = root
        
        while not current.is_leaf() and not current.state.is_terminal():
            action, child = current.select_child(self.c_puct)
            if action is None or child is None:
                break
            path.append((current, action, child))
            current = child
        
        return path, current
    
    @torch.no_grad()
    def _expand_nodes_batched(self, nodes: List[AlphaZeroNode]):
        """AGGRESSIVE batching to maximize GPU utilization - this is the key fix."""
        if not nodes:
            return
        
        # Prepare larger batches - RTX 4090 can handle much more
        state_tensors = []
        valid_nodes = []
        
        for node in nodes:
            try:
                state_tensor = node.state.to_tensor()
                state_tensors.append(state_tensor)
                valid_nodes.append(node)
            except Exception as e:
                logger.warning(f"Error preparing state tensor: {e}")
        
        if not state_tensors:
            return
        
        # LARGER BATCH PROCESSING - This maximizes GPU usage
        max_gpu_batch = 128  # Much larger for RTX 4090
        
        for i in range(0, len(state_tensors), max_gpu_batch):
            batch_end = min(i + max_gpu_batch, len(state_tensors))
            current_batch_tensors = state_tensors[i:batch_end]
            current_batch_nodes = valid_nodes[i:batch_end]
            
            # Stack tensors for massive batch inference
            batch_tensor = torch.stack(current_batch_tensors).to(self.device)
            
            # AGGRESSIVE OPTIMIZATION: Use mixed precision + optimized forward pass
            if self.use_mixed_precision:
                with torch.amp.autocast('cuda'):
                    policy_logits, values = self.neural_network(batch_tensor)
            else:
                policy_logits, values = self.neural_network(batch_tensor)
            
            # Convert outputs efficiently
            policy_probs = F.softmax(policy_logits, dim=-1)
            values = torch.tanh(values).squeeze(-1)
            
            # Expand nodes efficiently
            for j, node in enumerate(current_batch_nodes):
                try:
                    legal_actions = node.state.get_legal_actions()
                    if legal_actions:
                        node.expand(
                            policy_probs[j],
                            values[j].item(),
                            legal_actions,
                            node.state.action_to_index
                        )
                except Exception as e:
                    logger.warning(f"Error expanding node: {e}")
        
        # Force GPU synchronization to ensure all work is submitted
        torch.cuda.synchronize()
    
    def _backup_path(self, path: List, leaf: AlphaZeroNode, value: float):
        """Backup value through the path."""
        current_value = value
        
        # Backup leaf
        leaf.backup(current_value)
        
        # Backup through path
        for node, action, child in reversed(path):
            current_value = -current_value
            node.backup(current_value)
    
    def _simulate(self, root: AlphaZeroNode) -> None:
        """Single simulation (kept for compatibility)."""
        path = []
        current = root
        
        while not current.is_leaf() and not current.state.is_terminal():
            action, child = current.select_child(self.c_puct)
            if action is None or child is None:
                break
            path.append((current, action, child))
            current = child
        
        # Expansion & Evaluation
        if current.state.is_terminal():
            value = current.state.get_reward()
        else:
            if current.is_leaf():
                self._expand_node(current)
            value = current.neural_value
        
        # Backup
        current_value = value
        current.backup(current_value)
        
        for node, action, child in reversed(path):
            current_value = -current_value
            node.backup(current_value)
    
    @torch.no_grad()
    def _expand_node(self, node: AlphaZeroNode) -> None:
        """Single node expansion (kept for compatibility)."""
        try:
            state_tensor = node.state.to_tensor().unsqueeze(0).to(self.device)
            
            if self.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    policy_logits, value = self.neural_network(state_tensor)
            else:
                policy_logits, value = self.neural_network(state_tensor)
            
            policy_probs = F.softmax(policy_logits, dim=-1).squeeze(0)
            value = torch.tanh(value).item()
            
            legal_actions = node.state.get_legal_actions()
            if legal_actions:
                node.expand(policy_probs, value, legal_actions, node.state.action_to_index)
                
        except Exception as e:
            self.logger.error(f"Error in node expansion: {e}")
            # Fallback
            legal_actions = node.state.get_legal_actions()
            if legal_actions:
                node.is_expanded = True
                node.neural_value = 0.0
                uniform_prob = 1.0 / len(legal_actions)
                for action in legal_actions:
                    node.prior_probs[action] = uniform_prob
    
    def should_resign(self, value: float, move_count: int, current_state: ChessGameState = None) -> bool:
        """Optimized resignation check."""
        if move_count < 25:
            return False
        
        if value < self.resignation_threshold:
            if self.logger:
                player = 'white' if current_state and current_state.board.turn else 'black'
                self.logger.info(f"RESIGNATION: {player} resigns due to evaluation threshold at move {move_count} (eval: {value:.3f})")
            return True

        if current_state and should_resign_material(current_state.board, self.resignation_centipawns):
            if self.logger:
                player = 'white' if current_state and current_state.board.turn else 'black'
                self.logger.info(f"Final FEN: {current_state.board.fen()}")
                self.logger.info(f"RESIGNATION: {player} resigns due to material disadvantage at move {move_count} (eval: {value:.3f})")
            return True

        return False
    
    def get_best_action(self, root_state: ChessGameState, num_simulations: int = 800) -> Any:
        """Get best action with optimized search."""
        if root_state.is_terminal():
            return None
        
        root = self.search(root_state, num_simulations)
        return root.get_best_action()
    
    def get_action_probabilities(self, root_state: ChessGameState,
                                num_simulations: int = 800,
                                temperature: float = 1.0,
                                add_noise: bool = False,
                                dirichlet_alpha: float = 0.3) -> Dict[Any, float]:
        """Get action probabilities with optimized search."""
        if root_state.is_terminal():
            return {}
        
        root = self.search(root_state, num_simulations)
        action_probs = root.get_action_probabilities(temperature)
        
        if add_noise and action_probs:
            action_probs = self._add_dirichlet_noise(action_probs, alpha=dirichlet_alpha)
        
        return action_probs
    
    def _add_dirichlet_noise(self, action_probs: Dict[Any, float], alpha: float = 0.3) -> Dict[Any, float]:
        """Add Dirichlet noise for exploration."""
        if not action_probs:
            return action_probs
        
        try:
            actions = list(action_probs.keys())
            probs = np.array(list(action_probs.values()))
            
            noise = np.random.dirichlet([alpha] * len(actions))
            mixed_probs = 0.75 * probs + 0.25 * noise
            mixed_probs = mixed_probs / mixed_probs.sum()
            
            return {action: prob for action, prob in zip(actions, mixed_probs)}
        except:
            return action_probs
    
    def get_principal_variation(self, root_state: ChessGameState,
                               num_simulations: int = 800,
                               max_depth: int = 10) -> List[Any]:
        """Get principal variation."""
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
                           num_simulations: int = 150,  # Reduced but with better batching
                           temperature_schedule: callable = None,
                           max_moves: int = 150,
                           enable_resignation: bool = True,
                           filter_draws: bool = True,
                           style: str = 'tactical',
                           dirichlet_alpha: float = 0.3
                           ) -> tuple[List[AlphaZeroTrainingExample], ChessGameState, List[dict], bool, Optional[int]]:
    """
    Generate optimized self-play game.
    """
    logger = logging.getLogger(__name__ + ".generate_self_play_game")
    
    # Optimized temperature schedule
    if temperature_schedule is None:
        if style == 'tactical':
            temperature_schedule = lambda move: 1.2 if move < 15 else (0.8 if move < 30 else 0.0)
        elif style == 'positional':
            temperature_schedule = lambda move: 1.0 if move < 12 else (0.5 if move < 25 else 0.0)
        elif style == 'dynamic':
            temperature_schedule = lambda move: 1.1 if move < 20 else (0.7 if move < 35 else 0.0)
        else:
            temperature_schedule = lambda move: 1.0 if move < 25 else 0.0
    
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
        temperature = temperature_schedule(move_count)
        
        # Add noise for exploration
        add_noise = move_count < 25
        action_probs = chess_engine.get_action_probabilities(
            current_state, num_simulations=num_simulations, 
            temperature=temperature, add_noise=add_noise,
            dirichlet_alpha=dirichlet_alpha
        )

        # Log every 10 moves
        if move_count > 0 and move_count % 2 == 0:
            logger.info(
                f"Move {move_count}: FEN={current_state.board.fen()}, "
                f"Player={'white' if current_state.get_current_player() == 1 else 'black'}"
            )
        
        # Check resignation
        if enable_resignation and move_count > 20:
            root = chess_engine.search(current_state, num_simulations)
            root_value = root.value_sum / root.visit_count if root.visit_count > 0 else 0.0
            
            if chess_engine.should_resign(root_value, move_count, current_state):
                game_resigned = True
                winner = -current_state.get_current_player()
                break
        
        # Store training example
        example = AlphaZeroTrainingExample(
            state_tensor=current_state.to_tensor().clone(),
            action_probs=action_probs.copy(),
            outcome=0.0,
            current_player=current_state.get_current_player()
        )
        training_examples.append(example)
        
        # Sample action
        if temperature == 0.0:
            action = max(action_probs.keys(), key=lambda a: action_probs[a])
        else:
            actions = list(action_probs.keys())
            probabilities = list(action_probs.values())
            action_idx = torch.multinomial(torch.tensor(probabilities), 1).item()
            action = actions[action_idx]
        
        if action is None:
            break
                
        # Apply move
        move_san = current_state.board.san(action)
        current_state = current_state.apply_action(action)
        move_count += 1
        
        move_history.append({
            "move_number": move_count,
            "fen": current_state.board.fen(),
            "move": action.uci(),
            "move_san": move_san,
            "player": "black" if current_state.get_current_player() == 1 else "white"
        })
    
    # Determine outcome
    if game_resigned:
        final_outcome = -1.0 if winner == current_state.get_current_player() else 1.0
    elif current_state.is_terminal():
        final_outcome = current_state.get_reward()
        if current_state.board.outcome():
            if current_state.board.outcome().winner is None:
                winner = 0
            elif current_state.board.outcome().winner:
                winner = 1
            else:
                winner = -1
        else:
            winner = 0
    else:
        final_outcome = 0.0
        winner = 0
    
    # Update training examples
    for example in training_examples:
        if example.current_player == current_state.get_current_player():
            example.outcome = final_outcome
        else:
            example.outcome = -final_outcome
    
    # Filter draws if requested
    if filter_draws and abs(final_outcome) < 0.5:
        training_examples = []
    
    return training_examples, current_state, move_history, game_resigned, winner


# Export classes
__all__ = [
    'AlphaZeroMCTS',
    'AlphaZeroNode', 
    'AlphaZeroTrainingExample',
    'generate_self_play_game'
]