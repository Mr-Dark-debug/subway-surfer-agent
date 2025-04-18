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
        
        # CNN layers with memory efficiency in mind
        # Using smaller number of filters in early layers
        self.conv1 = nn.Conv2d(input_shape[0], 16, kernel_size=8, stride=4)  # Reduced from 32
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)  # Reduced from 64
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)  # Reduced from 64
        
        # Calculate size after convolutions
        def conv2d_size_out(size, kernel_size, stride):
            return (size - kernel_size) // stride + 1
        
        # Calculate flattened size after convolutions
        conv_width = conv2d_size_out(conv2d_size_out(conv2d_size_out(input_shape[2], 8, 4), 4, 2), 3, 1)
        conv_height = conv2d_size_out(conv2d_size_out(conv2d_size_out(input_shape[1], 8, 4), 4, 2), 3, 1)
        linear_input_size = conv_width * conv_height * 32
        
        # Fully connected layers (also reduced size)
        self.fc1 = nn.Linear(linear_input_size, 256)  # Reduced from 512
        self.fc2 = nn.Linear(256, n_actions)
        
        # Initialize weights
        self._initialize_weights()
        
        # For tracking performance
        self.forward_times = []
        
        logger.info(f"Memory-efficient DQN initialized with input shape {input_shape} and {n_actions} actions")
        logger.info(f"Convolutional feature shape: {conv_height}x{conv_width}x32")
        
        # Calculate and log model parameter count
        param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Model has {param_count:,} trainable parameters")
    
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
        
        # Shared convolutional layers (feature extractor) with reduced size
        self.conv1 = nn.Conv2d(input_shape[0], 16, kernel_size=8, stride=4)  # Reduced from 32
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)  # Reduced from 64
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)  # Reduced from 64
        
        # Calculate size after convolutions
        def conv2d_size_out(size, kernel_size, stride):
            return (size - kernel_size) // stride + 1
        
        # Calculate flattened size after convolutions
        conv_width = conv2d_size_out(conv2d_size_out(conv2d_size_out(input_shape[2], 8, 4), 4, 2), 3, 1)
        conv_height = conv2d_size_out(conv2d_size_out(conv2d_size_out(input_shape[1], 8, 4), 4, 2), 3, 1)
        self.conv_output_size = conv_width * conv_height * 32
        
        # Value stream (estimates V(s)) with reduced size
        self.value_stream = nn.Sequential(
            nn.Linear(self.conv_output_size, 256),  # Reduced from 512
            nn.ReLU(),
            nn.Linear(256, 1)  # Single value output
        )
        
        # Advantage stream (estimates A(s,a)) with reduced size
        self.advantage_stream = nn.Sequential(
            nn.Linear(self.conv_output_size, 256),  # Reduced from 512
            nn.ReLU(),
            nn.Linear(256, n_actions)  # One output per action
        )
        
        # Initialize weights
        self._initialize_weights()
        
        # For tracking performance
        self.forward_times = []
        
        logger.info(f"Memory-efficient DuelingDQN initialized with input shape {input_shape} and {n_actions} actions")
        logger.info(f"Convolutional feature shape: {conv_height}x{conv_width}x32")
        
        # Calculate and log model parameter count
        param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Model has {param_count:,} trainable parameters")
    
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

# Memory-efficient Dueling DQN with reduced parameter count
class EfficientDuelingDQN(nn.Module):
    """
    Highly optimized Dueling DQN for memory-constrained environments (4GB VRAM)
    Uses depth-wise separable convolutions and smaller layers
    """
    def __init__(self, input_shape, n_actions):
        super(EfficientDuelingDQN, self).__init__()
        
        self.input_shape = input_shape
        self.n_actions = n_actions
        
        # First conv layer: Regular convolution for initial feature extraction
        self.conv1 = nn.Conv2d(input_shape[0], 16, kernel_size=8, stride=4)
        
        # Second layer: Depthwise separable convolution
        # First part: Depthwise convolution
        self.conv2_depthwise = nn.Conv2d(16, 16, kernel_size=4, stride=2, groups=16)
        # Second part: Pointwise convolution
        self.conv2_pointwise = nn.Conv2d(16, 32, kernel_size=1)
        
        # Third layer: Depthwise separable convolution
        # First part: Depthwise convolution
        self.conv3_depthwise = nn.Conv2d(32, 32, kernel_size=3, stride=1, groups=32)
        # Second part: Pointwise convolution
        self.conv3_pointwise = nn.Conv2d(32, 32, kernel_size=1)
        
        # Calculate output size
        def conv2d_size_out(size, kernel_size, stride):
            return (size - kernel_size) // stride + 1
        
        # Calculate size of flattened features
        conv_width = conv2d_size_out(conv2d_size_out(conv2d_size_out(input_shape[2], 8, 4), 4, 2), 3, 1)
        conv_height = conv2d_size_out(conv2d_size_out(conv2d_size_out(input_shape[1], 8, 4), 4, 2), 3, 1)
        self.conv_output_size = conv_width * conv_height * 32
        
        # Value stream (smaller)
        self.value_stream = nn.Sequential(
            nn.Linear(self.conv_output_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Advantage stream (smaller)
        self.advantage_stream = nn.Sequential(
            nn.Linear(self.conv_output_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )
        
        # Initialize weights
        self._initialize_weights()
        
        # Tracking
        self.forward_times = []
        
        # Log model info
        logger.info(f"EfficientDuelingDQN initialized with input shape {input_shape} and {n_actions} actions")
        
        # Calculate and log model parameter count
        param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Memory-efficient model has {param_count:,} trainable parameters")
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        start_time = time.time()
        
        # Handle input tensor
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        
        if x.dim() == 3:
            x = x.unsqueeze(0)
            
        if x.shape[1:] != self.input_shape:
            batch_size = x.shape[0]
            try:
                x = x.reshape(batch_size, *self.input_shape)
            except RuntimeError:
                logger.error(f"Could not reshape input from {x.shape} to {(batch_size, *self.input_shape)}")
                raise
        
        # Feature extraction with depthwise separable convolutions
        x = F.relu(self.conv1(x))
        
        # Depthwise separable conv 2
        x = F.relu(self.conv2_depthwise(x))
        x = F.relu(self.conv2_pointwise(x))
        
        # Depthwise separable conv 3
        x = F.relu(self.conv3_depthwise(x))
        x = F.relu(self.conv3_pointwise(x))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Value and advantage streams
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        
        # Combine for Q-values
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        # Track time
        self.forward_times.append(time.time() - start_time)
        
        return q_values
    
    def get_forward_time_stats(self):
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
        stats = self.get_forward_time_stats()
        
        logger.info("EfficientDuelingDQN Forward Pass Performance Statistics:")
        logger.info(f"  Mean time: {stats['mean']*1000:.2f}ms")
        logger.info(f"  Std Dev: {stats['std']*1000:.2f}ms")
        logger.info(f"  Min: {stats['min']*1000:.2f}ms")
        logger.info(f"  Max: {stats['max']*1000:.2f}ms")
        logger.info(f"  Count: {stats['count']}")