# model/dqn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DQN")

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        """
        Initialize the DQN model
        
        Args:
            input_shape: Shape of the input state (frames, height, width)
            n_actions: Number of possible actions
        """
        super(DQN, self).__init__()
        
        self.input_shape = input_shape
        self.n_actions = n_actions
        
        # CNN layers - Using fixed kernel sizes and strides for TensorFlow Lite compatibility
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate size after convolutions
        def conv2d_size_out(size, kernel_size, stride):
            return (size - kernel_size) // stride + 1
        
        # Calculate flattened size after convolutions
        conv_width = conv2d_size_out(conv2d_size_out(conv2d_size_out(input_shape[2], 8, 4), 4, 2), 3, 1)
        conv_height = conv2d_size_out(conv2d_size_out(conv2d_size_out(input_shape[1], 8, 4), 4, 2), 3, 1)
        linear_input_size = conv_width * conv_height * 64
        
        # Fully connected layers
        self.fc1 = nn.Linear(linear_input_size, 512)
        self.fc2 = nn.Linear(512, n_actions)
        
        # Initialize weights
        self._initialize_weights()
        
        # For tracking performance
        self.forward_times = []
        
        logger.info(f"DQN initialized with input shape {input_shape} and {n_actions} actions")
        logger.info(f"Convolutional feature shape: {conv_height}x{conv_width}x64")
    
    def _initialize_weights(self):
        """Initialize model weights using He initialization for ReLU networks"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, frames, height, width)
            
        Returns:
            Q-values for each action
        """
        start_time = time.time()
        
        # Ensure input tensor has the right shape and dtype
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        
        # Ensure batch dimension
        if x.dim() == 3:
            x = x.unsqueeze(0)  # Add batch dimension if missing
        
        # Ensure correct shape (batch_size, frames, height, width)
        if x.shape[1:] != self.input_shape:
            logger.warning(f"Input shape mismatch: expected {self.input_shape}, got {x.shape[1:]}. Reshaping...")
            # Try to reshape the input while preserving batch size
            batch_size = x.shape[0]
            try:
                x = x.reshape(batch_size, *self.input_shape)
            except RuntimeError:
                logger.error(f"Could not reshape input from {x.shape} to {(batch_size, *self.input_shape)}")
                raise
        
        # Convolutional layers with ReLU activations
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        # Track forward pass time
        forward_time = time.time() - start_time
        self.forward_times.append(forward_time)
        
        return x
    
    def get_conv_output_size(self):
        """Calculate the output size of the convolutional layers"""
        # Create a dummy input tensor
        dummy_input = torch.zeros(1, *self.input_shape)
        
        # Pass it through conv layers
        x = F.relu(self.conv1(dummy_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Return the shape
        return x.shape
    
    def get_forward_time_stats(self):
        """
        Get statistics about forward pass time
        
        Returns:
            Dictionary of forward pass time statistics
        """
        if not self.forward_times:
            return {
                'mean': 0,
                'std': 0,
                'min': 0,
                'max': 0,
                'count': 0
            }
        
        return {
            'mean': np.mean(self.forward_times),
            'std': np.std(self.forward_times),
            'min': np.min(self.forward_times),
            'max': np.max(self.forward_times),
            'count': len(self.forward_times)
        }
    
    def log_performance_stats(self):
        """Log performance statistics to the log file"""
        stats = self.get_forward_time_stats()
        
        logger.info("DQN Forward Pass Performance Statistics:")
        logger.info(f"  Mean time: {stats['mean']*1000:.2f}ms")
        logger.info(f"  Std Dev: {stats['std']*1000:.2f}ms")
        logger.info(f"  Min: {stats['min']*1000:.2f}ms")
        logger.info(f"  Max: {stats['max']*1000:.2f}ms")
        logger.info(f"  Count: {stats['count']}")
    
    def save(self, file_path):
        """Save the model to a file"""
        torch.save(self.state_dict(), file_path)
        logger.info(f"Model saved to {file_path}")
    
    def load(self, file_path, device='cpu'):
        """Load the model from a file"""
        self.load_state_dict(torch.load(file_path, map_location=device))
        self.to(device)
        logger.info(f"Model loaded from {file_path} to {device}")


class DuelingDQN(nn.Module):
    """
    Dueling DQN architecture with separate value and advantage streams
    Helps the model learn which states are valuable without having to learn
    the effect of each action for each state
    """
    def __init__(self, input_shape, n_actions):
        """
        Initialize the Dueling DQN model
        
        Args:
            input_shape: Shape of the input state (frames, height, width)
            n_actions: Number of possible actions
        """
        super(DuelingDQN, self).__init__()
        
        self.input_shape = input_shape
        self.n_actions = n_actions
        
        # Shared convolutional layers (feature extractor)
        # Using static kernel sizes and strides for TensorFlow Lite compatibility
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate size after convolutions
        def conv2d_size_out(size, kernel_size, stride):
            return (size - kernel_size) // stride + 1
        
        # Calculate flattened size after convolutions
        conv_width = conv2d_size_out(conv2d_size_out(conv2d_size_out(input_shape[2], 8, 4), 4, 2), 3, 1)
        conv_height = conv2d_size_out(conv2d_size_out(conv2d_size_out(input_shape[1], 8, 4), 4, 2), 3, 1)
        self.conv_output_size = conv_width * conv_height * 64
        
        # Value stream (estimates V(s))
        self.value_stream = nn.Sequential(
            nn.Linear(self.conv_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)  # Single value output
        )
        
        # Advantage stream (estimates A(s,a))
        self.advantage_stream = nn.Sequential(
            nn.Linear(self.conv_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)  # One output per action
        )
        
        # Initialize weights
        self._initialize_weights()
        
        # For tracking performance
        self.forward_times = []
        
        logger.info(f"DuelingDQN initialized with input shape {input_shape} and {n_actions} actions")
        logger.info(f"Convolutional feature shape: {conv_height}x{conv_width}x64")
    
    def _initialize_weights(self):
        """Initialize model weights using He initialization for ReLU networks"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, frames, height, width)
            
        Returns:
            Q-values for each action
        """
        start_time = time.time()
        
        # Ensure input tensor has the right shape and dtype
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        
        # Ensure batch dimension
        if x.dim() == 3:
            x = x.unsqueeze(0)  # Add batch dimension if missing
        
        # Ensure correct shape (batch_size, frames, height, width)
        if x.shape[1:] != self.input_shape:
            logger.warning(f"Input shape mismatch: expected {self.input_shape}, got {x.shape[1:]}. Reshaping...")
            # Try to reshape the input while preserving batch size
            batch_size = x.shape[0]
            try:
                x = x.reshape(batch_size, *self.input_shape)
            except RuntimeError:
                logger.error(f"Could not reshape input from {x.shape} to {(batch_size, *self.input_shape)}")
                raise
            
        # Shared convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Split into value and advantage streams
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        
        # Combine value and advantage to get Q-values
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        # This formula ensures identifiability by forcing the advantage
        # function to have zero mean across actions
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        # Track forward pass time
        forward_time = time.time() - start_time
        self.forward_times.append(forward_time)
        
        return q_values
    
    def get_conv_output_size(self):
        """Calculate the output size of the convolutional layers"""
        # Create a dummy input tensor
        dummy_input = torch.zeros(1, *self.input_shape)
        
        # Pass it through conv layers
        x = F.relu(self.conv1(dummy_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Return the shape
        return x.shape
    
    def get_forward_time_stats(self):
        """
        Get statistics about forward pass time
        
        Returns:
            Dictionary of forward pass time statistics
        """
        if not self.forward_times:
            return {
                'mean': 0,
                'std': 0,
                'min': 0,
                'max': 0,
                'count': 0
            }
        
        return {
            'mean': np.mean(self.forward_times),
            'std': np.std(self.forward_times),
            'min': np.min(self.forward_times),
            'max': np.max(self.forward_times),
            'count': len(self.forward_times)
        }
    
    def log_performance_stats(self):
        """Log performance statistics to the log file"""
        stats = self.get_forward_time_stats()
        
        logger.info("DuelingDQN Forward Pass Performance Statistics:")
        logger.info(f"  Mean time: {stats['mean']*1000:.2f}ms")
        logger.info(f"  Std Dev: {stats['std']*1000:.2f}ms")
        logger.info(f"  Min: {stats['min']*1000:.2f}ms")
        logger.info(f"  Max: {stats['max']*1000:.2f}ms")
        logger.info(f"  Count: {stats['count']}")
    
    def save(self, file_path):
        """Save the model to a file"""
        torch.save(self.state_dict(), file_path)
        logger.info(f"Model saved to {file_path}")
    
    def load(self, file_path, device='cpu'):
        """Load the model from a file"""
        self.load_state_dict(torch.load(file_path, map_location=device))
        self.to(device)
        logger.info(f"Model loaded from {file_path} to {device}")