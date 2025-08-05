"""
AlphaZero Neural Network Architecture

This module defines the neural network architecture used in AlphaZero.

Referenced papers:
PAPER: Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm
PAPER: Mastering the game of Go without human knowledge
PAPER: A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config.model_config import get_model_config

class ResidualBlock(nn.Module):
    """A single residual block for the AlphaZero network with two 3x3 convolutions."""

    def __init__(self, num_filters: int):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x):
        """Forward pass through the residual block."""
        residual = x

        # First convolution layer with ReLU activation
        out = F.relu(self.bn1(self.conv1(x)))

        # Second convolution layer
        out = self.bn2(self.conv2(out))

        # Add the residual connection
        out += residual
        # Final ReLU activation
        out = F.relu(out)

        return out

class AlphaZeroNet(nn.Module):
    """
    AlphaZero neural network with ResNet backbone and dual heads for policy and value.
    
    Args:
        board_size: Size of the game board (e.g., 19 for Go, 8 for Chess)
        action_size: Number of possible actions/moves
        num_filters: Number of convolutional filters (default: 256)
        num_res_blocks: Number of residual blocks (default: 19)
        input_channels: Number of input channels (default: 17 for Go)
    """
    
    def __init__(
        self,
        board_size: int = 8,
        action_size: int = 4672,
        num_filters: int = 256,
        num_res_blocks: int = 19,
        input_channels: int = 119
    ):
        super(AlphaZeroNet, self).__init__()

        self.board_size = board_size
        self.action_size = action_size
        self.num_filters = num_filters

        # Initial convolution block
        self.initial_conv = nn.Conv2d(input_channels, num_filters, kernel_size=3, padding=1, bias=False)
        self.initial_bn = nn.BatchNorm2d(num_filters)

        # Residual blocks (19 as per AlphaZero architecture)
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(num_filters) for _ in range(num_res_blocks)]
        )

        # Policy head (move probabilities)
        self.policy_conv = nn.Conv2d(num_filters, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, action_size)

        # Value head (position evaluation)
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, num_filters)
        self.value_fc2 = nn.Linear(num_filters, 1)

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_channels, board_size, board_size)
            
        Returns:
            policy_logits: Raw policy logits of shape (batch_size, action_size)
            value: Position value of shape (batch_size, 1)
        """
        batch_size = x.size(0)

        # Initial convolution block
        x = F.relu(self.initial_bn(self.initial_conv(x)))

        # Shared ResNet processing through residual blocks
        for res_block in self.res_blocks:
            x = res_block(x)

        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(batch_size, -1)  # Flatten
        policy_logits = self.policy_fc(policy)

        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(batch_size, -1)  # Flatten
        value = F.relu(self.value_fc1(value))
        value = self.value_fc2(value)
        # Apply tanh activation to value output that ranges from -1 to 1
        # This is common in AlphaZero architectures to normalize the value output
        # -1 means a loss, 1 means a win and 0 means a draw
        value = torch.tanh(value)

        return policy_logits, value
    
    def predict(self, x):
        """
        Prediction method that returns probabilities and value.
        
        Args:
            x: Input tensor
            
        Returns:
            policy_probs: Action probabilities (softmax of logits)
            value: Position value
        """
        self.eval()  # Set to evaluation mode

        with torch.no_grad():
            policy_logits, value = self.forward(x)
            policy_probs = F.softmax(policy_logits, dim=-1)  # Convert logits to probabilities
        return policy_probs, value
    
def create_alphazero_chess_net() -> AlphaZeroNet:
    """
    Factory function to create AlphaZero networks for chess.
        
    Returns:
        AlphaZeroNet instance configured for chess
    """
    # Use the model configuration for chess
    model_config = get_model_config()

    # Create the AlphaZero network with chess-specific parameters
    net = AlphaZeroNet(
        board_size=model_config.board_size,
        action_size=model_config.action_size,
        input_channels=model_config.input_channels
    )
    return net