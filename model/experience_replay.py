# model/experience_replay.py
import numpy as np
import random
from collections import deque
import torch
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ReplayBuffer")

class ReplayBuffer:
    def __init__(self, capacity, state_shape, device="cpu"):
        """
        Initialize a Replay Buffer for DQN training
        
        Args:
            capacity: Maximum number of transitions to store
            state_shape: Shape of a state (frames, height, width)
            device: Device to store tensor data on (cpu/cuda)
        """
        self.capacity = capacity
        self.device = device
        self.memory = deque(maxlen=capacity)
        self.state_shape = state_shape
        
        logger.info(f"ReplayBuffer initialized with capacity {capacity} on device {device}")
    
    def push(self, state, action, reward, next_state, done):
        """
        Store a transition in the buffer
        
        Args:
            state: Current state (numpy array)
            action: Action taken (integer)
            reward: Reward received (float)
            next_state: Next state (numpy array)
            done: Whether the episode ended (bool)
        """
        # Ensure state and next_state are numpy arrays
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.cpu().numpy()
            
        # Store the transition
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        Sample a batch of transitions from the buffer
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as tensors
        """
        if len(self.memory) < batch_size:
            logger.warning(f"Not enough transitions in buffer ({len(self.memory)}) to sample batch of size {batch_size}")
            return None
        
        # Sample random batch
        transitions = random.sample(self.memory, batch_size)
        
        # Convert to arrays then to tensors (more efficient than converting one by one)
        # Batched conversion to tensors is much faster
        batch = tuple(zip(*transitions))
        
        # Extract components and convert to tensors
        states = torch.tensor(np.array(batch[0]), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array(batch[1]), dtype=torch.long, device=self.device)
        rewards = torch.tensor(np.array(batch[2]), dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array(batch[3]), dtype=torch.float32, device=self.device)
        dones = torch.tensor(np.array(batch[4]), dtype=torch.bool, device=self.device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """Return the current size of the buffer"""
        return len(self.memory)
    
    def is_ready(self, batch_size):
        """
        Check if the buffer has enough samples for a batch
        
        Args:
            batch_size: Size of batch to check for
            
        Returns:
            Boolean indicating if buffer has enough samples
        """
        return len(self) >= batch_size
    
    def clear(self):
        """Clear the replay buffer"""
        self.memory.clear()
        logger.info("Replay buffer cleared")
    
    def save(self, path):
        """
        Save the replay buffer to a file
        
        Args:
            path: Path to save the buffer to
            
        Note: This can be useful for continuing training from a saved state
        """
        # Convert deque to list for saving
        memory_list = list(self.memory)
        
        # Save using numpy as it's more efficient for large arrays
        np.save(path, memory_list)
        logger.info(f"Replay buffer saved to {path}")
    
    def load(self, path):
        """
        Load the replay buffer from a file
        
        Args:
            path: Path to load the buffer from
        """
        try:
            # Load the data
            memory_list = np.load(path, allow_pickle=True)
            
            # Convert back to deque
            self.memory = deque(memory_list, maxlen=self.capacity)
            
            logger.info(f"Replay buffer loaded from {path} with {len(self.memory)} transitions")
        except Exception as e:
            logger.error(f"Error loading replay buffer: {str(e)}")


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay (PER) buffer
    
    Stores transitions with priorities based on TD error,
    which helps focus learning on the most informative transitions.
    """
    def __init__(self, capacity, state_shape, device="cpu", alpha=0.6, beta_start=0.4, beta_frames=100000):
        """
        Initialize a Prioritized Replay Buffer
        
        Args:
            capacity: Maximum number of transitions to store
            state_shape: Shape of a state
            device: Device to store tensor data on (cpu/cuda)
            alpha: How much prioritization to use (0 = no prioritization, 1 = full prioritization)
            beta_start: Initial importance-sampling weight
            beta_frames: Number of frames over which to anneal beta to 1.0
        """
        self.capacity = capacity
        self.device = device
        self.state_shape = state_shape
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1  # For beta annealing
        
        # Initialize buffer storage
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        
        logger.info(f"PrioritizedReplayBuffer initialized with capacity {capacity}, alpha={alpha}")
    
    def push(self, state, action, reward, next_state, done):
        """
        Store a transition with maximum priority
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode ended
        """
        # Ensure state and next_state are numpy arrays
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.cpu().numpy()
        
        # Find the maximum priority in buffer
        max_priority = self.priorities.max() if self.buffer else 1.0
        
        # If buffer not full, add to it
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            # Replace old transition
            self.buffer[self.position] = (state, action, reward, next_state, done)
        
        # Update priority
        self.priorities[self.position] = max_priority
        
        # Update position (circular buffer)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """
        Sample a batch of transitions based on priorities
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones, weights, indices)
        """
        if len(self.buffer) < batch_size:
            logger.warning(f"Not enough transitions in buffer ({len(self.buffer)}) to sample batch of size {batch_size}")
            return None
        
        # Calculate current beta value
        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        self.frame += 1
        
        # Get current size of buffer
        buffer_size = len(self.buffer)
        
        # Calculate sampling probabilities
        probs = self.priorities[:buffer_size] ** self.alpha
        probs /= probs.sum()
        
        # Sample indices based on priorities
        indices = np.random.choice(buffer_size, batch_size, replace=False, p=probs)
        
        # Get samples from indices
        samples = [self.buffer[idx] for idx in indices]
        
        # Calculate importance-sampling weights
        weights = (buffer_size * probs[indices]) ** (-beta)
        weights /= weights.max()  # Normalize to maximum weight
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
        
        # Extract components and convert to tensors
        batch = tuple(zip(*samples))
        states = torch.tensor(np.array(batch[0]), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array(batch[1]), dtype=torch.long, device=self.device)
        rewards = torch.tensor(np.array(batch[2]), dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array(batch[3]), dtype=torch.float32, device=self.device)
        dones = torch.tensor(np.array(batch[4]), dtype=torch.bool, device=self.device)
        
        return states, actions, rewards, next_states, dones, weights, indices
    
    def update_priorities(self, indices, priorities):
        """
        Update priorities for sampled transitions
        
        Args:
            indices: Indices of transitions to update
            priorities: New priority values
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
    
    def __len__(self):
        """Return the current size of the buffer"""
        return len(self.buffer)
    
    def is_ready(self, batch_size):
        """Check if buffer has enough samples for a batch"""
        return len(self) >= batch_size