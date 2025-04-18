# model/experience_replay.py
import numpy as np
import random
from collections import deque
import torch
import logging
import time
import gc

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
        
        # For performance tracking
        self.push_times = []
        self.sample_times = []
        self.total_bytes = 0
        
        # Calculate approximate memory usage per transition
        bytes_per_float = 4  # 32-bit float
        state_size = np.prod(state_shape) * bytes_per_float * 2  # state and next_state
        action_reward_done_size = 8 + 4 + 1  # action (int64), reward (float32), done (bool)
        transition_size = state_size + action_reward_done_size
        self.estimated_memory_usage = capacity * transition_size / (1024 * 1024)  # MB
        
        logger.info(f"ReplayBuffer initialized with capacity {capacity} on device {device}")
        logger.info(f"Estimated memory usage: {self.estimated_memory_usage:.2f} MB")
    
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
        start_time = time.time()
        
        # Ensure state and next_state are numpy arrays
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.cpu().numpy()
            
        # Store the transition
        self.memory.append((state, action, reward, next_state, done))
        
        # Track memory usage
        transition_bytes = (state.nbytes + next_state.nbytes + 8 + 4 + 1)
        self.total_bytes += transition_bytes
        
        # Track push time
        self.push_times.append(time.time() - start_time)
    
    def sample(self, batch_size):
        """
        Sample a batch of transitions from the buffer
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as tensors
        """
        start_time = time.time()
        
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
        
        # Track sample time
        self.sample_times.append(time.time() - start_time)
        
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
        self.total_bytes = 0
        
        # Force garbage collection to free memory
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info("Replay buffer cleared and memory freed")
    
    def get_performance_stats(self):
        """
        Get performance statistics about the replay buffer
        
        Returns:
            Dictionary of performance statistics
        """
        stats = {
            'push_time': {
                'mean': np.mean(self.push_times) if self.push_times else 0,
                'std': np.std(self.push_times) if self.push_times else 0,
                'count': len(self.push_times)
            },
            'sample_time': {
                'mean': np.mean(self.sample_times) if self.sample_times else 0,
                'std': np.std(self.sample_times) if self.sample_times else 0,
                'count': len(self.sample_times)
            },
            'memory': {
                'size': len(self.memory),
                'capacity': self.capacity,
                'utilization': len(self.memory) / self.capacity if self.capacity > 0 else 0,
                'estimated_mb': self.estimated_memory_usage,
                'actual_mb': self.total_bytes / (1024 * 1024)
            }
        }
        
        return stats
    
    def log_performance_stats(self):
        """Log performance statistics to the log file"""
        stats = self.get_performance_stats()
        
        logger.info("ReplayBuffer Performance Statistics:")
        logger.info(f"  Push time: {stats['push_time']['mean']*1000:.2f}ms (±{stats['push_time']['std']*1000:.2f}ms)")
        logger.info(f"  Sample time: {stats['sample_time']['mean']*1000:.2f}ms (±{stats['sample_time']['std']*1000:.2f}ms)")
        logger.info(f"  Buffer utilization: {stats['memory']['utilization']*100:.1f}% ({stats['memory']['size']}/{stats['memory']['capacity']})")
        logger.info(f"  Memory usage: {stats['memory']['actual_mb']:.2f} MB (estimated: {stats['memory']['estimated_mb']:.2f} MB)")


class EfficientReplayBuffer:
    """
    Memory-efficient implementation of replay buffer
    - Uses NumPy arrays instead of deque for more compact storage
    - Uses memory mapping for very large buffers
    - Implements circular buffer without recreating arrays
    """
    def __init__(self, capacity, state_shape, device="cpu", use_mmap=False, mmap_dir="./mmap"):
        """
        Initialize an efficient replay buffer
        
        Args:
            capacity: Maximum number of transitions to store
            state_shape: Shape of a state (frames, height, width)
            device: Device to store tensor data on (cpu/cuda)
            use_mmap: Whether to use memory mapping for large buffers
            mmap_dir: Directory to store memory-mapped files
        """
        self.capacity = capacity
        self.device = device
        self.state_shape = state_shape
        self.position = 0
        self.size = 0
        self.use_mmap = use_mmap
        
        # Create storage using numpy arrays
        if use_mmap:
            import os
            os.makedirs(mmap_dir, exist_ok=True)
            
            # Create memory-mapped arrays for large buffers
            self.states = np.memmap(f"{mmap_dir}/states.dat", dtype=np.float32, mode='w+',
                                   shape=(capacity, *state_shape))
            self.next_states = np.memmap(f"{mmap_dir}/next_states.dat", dtype=np.float32, mode='w+',
                                        shape=(capacity, *state_shape))
            self.actions = np.memmap(f"{mmap_dir}/actions.dat", dtype=np.int64, mode='w+',
                                    shape=(capacity,))
            self.rewards = np.memmap(f"{mmap_dir}/rewards.dat", dtype=np.float32, mode='w+',
                                    shape=(capacity,))
            self.dones = np.memmap(f"{mmap_dir}/dones.dat", dtype=np.bool_, mode='w+',
                                  shape=(capacity,))
            
            logger.info(f"Using memory-mapped arrays in {mmap_dir}")
        else:
            # Use regular numpy arrays for smaller buffers
            self.states = np.zeros((capacity, *state_shape), dtype=np.float32)
            self.next_states = np.zeros((capacity, *state_shape), dtype=np.float32)
            self.actions = np.zeros(capacity, dtype=np.int64)
            self.rewards = np.zeros(capacity, dtype=np.float32)
            self.dones = np.zeros(capacity, dtype=np.bool_)
        
        # For performance tracking
        self.push_times = []
        self.sample_times = []
        
        # Calculate approximate memory usage per transition
        bytes_per_float = 4  # 32-bit float
        state_size = np.prod(state_shape) * bytes_per_float * 2  # state and next_state
        action_reward_done_size = 8 + 4 + 1  # action (int64), reward (float32), done (bool)
        transition_size = state_size + action_reward_done_size
        self.estimated_memory_usage = capacity * transition_size / (1024 * 1024)  # MB
        
        logger.info(f"EfficientReplayBuffer initialized with capacity {capacity} on device {device}")
        logger.info(f"Estimated memory usage: {self.estimated_memory_usage:.2f} MB")
    
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
        start_time = time.time()
        
        # Ensure state and next_state are numpy arrays
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.cpu().numpy()
        
        # Store transition in arrays
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done
        
        # Update position and size
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
        # Flush to disk if using memory mapping
        if self.use_mmap and self.position % 1000 == 0:
            self.states.flush()
            self.next_states.flush()
            self.actions.flush()
            self.rewards.flush()
            self.dones.flush()
        
        # Track push time
        self.push_times.append(time.time() - start_time)
    
    def sample(self, batch_size):
        """
        Sample a batch of transitions from the buffer
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as tensors
        """
        start_time = time.time()
        
        if self.size < batch_size:
            logger.warning(f"Not enough transitions in buffer ({self.size}) to sample batch of size {batch_size}")
            return None
        
        # Sample random indices
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        # Get batch from arrays
        states = torch.tensor(self.states[indices], dtype=torch.float32, device=self.device)
        actions = torch.tensor(self.actions[indices], dtype=torch.long, device=self.device)
        rewards = torch.tensor(self.rewards[indices], dtype=torch.float32, device=self.device)
        next_states = torch.tensor(self.next_states[indices], dtype=torch.float32, device=self.device)
        dones = torch.tensor(self.dones[indices], dtype=torch.bool, device=self.device)
        
        # Track sample time
        self.sample_times.append(time.time() - start_time)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """Return the current size of the buffer"""
        return self.size
    
    def is_ready(self, batch_size):
        """Check if buffer has enough samples for a batch"""
        return self.size >= batch_size
    
    def clear(self):
        """Clear the replay buffer"""
        self.position = 0
        self.size = 0
        
        # Zero out arrays
        if not self.use_mmap:
            self.states.fill(0)
            self.next_states.fill(0)
            self.actions.fill(0)
            self.rewards.fill(0)
            self.dones.fill(0)
        
        # Force garbage collection to free memory
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info("Replay buffer cleared and memory freed")
    
    def get_performance_stats(self):
        """
        Get performance statistics about the replay buffer
        
        Returns:
            Dictionary of performance statistics
        """
        stats = {
            'push_time': {
                'mean': np.mean(self.push_times) if self.push_times else 0,
                'std': np.std(self.push_times) if self.push_times else 0,
                'count': len(self.push_times)
            },
            'sample_time': {
                'mean': np.mean(self.sample_times) if self.sample_times else 0,
                'std': np.std(self.sample_times) if self.sample_times else 0,
                'count': len(self.sample_times)
            },
            'memory': {
                'size': self.size,
                'capacity': self.capacity,
                'utilization': self.size / self.capacity if self.capacity > 0 else 0,
                'estimated_mb': self.estimated_memory_usage
            }
        }
        
        return stats
    
    def log_performance_stats(self):
        """Log performance statistics to the log file"""
        stats = self.get_performance_stats()
        
        logger.info("EfficientReplayBuffer Performance Statistics:")
        logger.info(f"  Push time: {stats['push_time']['mean']*1000:.2f}ms (±{stats['push_time']['std']*1000:.2f}ms)")
        logger.info(f"  Sample time: {stats['sample_time']['mean']*1000:.2f}ms (±{stats['sample_time']['std']*1000:.2f}ms)")
        logger.info(f"  Buffer utilization: {stats['memory']['utilization']*100:.1f}% ({stats['memory']['size']}/{stats['memory']['capacity']})")
        logger.info(f"  Estimated memory usage: {stats['memory']['estimated_mb']:.2f} MB")
        if self.use_mmap:
            logger.info("  Using memory-mapped storage (most data on disk)")


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
        
        # Performance tracking
        self.push_times = []
        self.sample_times = []
        self.update_times = []
        self.total_bytes = 0
        
        # Calculate approximate memory usage
        bytes_per_float = 4  # 32-bit float
        state_size = np.prod(state_shape) * bytes_per_float * 2  # state and next_state
        action_reward_done_size = 8 + 4 + 1  # action (int64), reward (float32), done (bool)
        transition_size = state_size + action_reward_done_size
        priorities_size = capacity * 4  # float32 priorities
        self.estimated_memory_usage = (capacity * transition_size + priorities_size) / (1024 * 1024)  # MB
        
        logger.info(f"PrioritizedReplayBuffer initialized with capacity {capacity}, alpha={alpha}")
        logger.info(f"Estimated memory usage: {self.estimated_memory_usage:.2f} MB")
    
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
        start_time = time.time()
        
        # Ensure state and next_state are numpy arrays
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.cpu().numpy()
        
        # Compress state and next_state to save memory if possible
        # Only compress if they have the same values (when reset happens)
        same_state = np.array_equal(state, next_state)
        
        # Track memory usage
        transition_size = (state.nbytes + (0 if same_state else next_state.nbytes) + 8 + 4 + 1)
        self.total_bytes += transition_size
        
        # Find the maximum priority in buffer
        max_priority = self.priorities.max() if self.buffer else 1.0
        
        # If buffer not full, add to it
        if len(self.buffer) < self.capacity:
            if same_state:
                # Store the transition with a reference to the same state
                self.buffer.append((state, action, reward, 'same_as_state', done))
            else:
                # Store the complete transition
                self.buffer.append((state, action, reward, next_state, done))
        else:
            # Replace old transition
            if same_state:
                self.buffer[self.position] = (state, action, reward, 'same_as_state', done)
            else:
                self.buffer[self.position] = (state, action, reward, next_state, done)
        
        # Update priority
        self.priorities[self.position] = max_priority
        
        # Update position (circular buffer)
        self.position = (self.position + 1) % self.capacity
        
        # Track push time
        self.push_times.append(time.time() - start_time)
    
    def sample(self, batch_size):
        """
        Sample a batch of transitions based on priorities
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones, weights, indices)
        """
        start_time = time.time()
        
        buffer_size = len(self.buffer)
        if buffer_size < batch_size:
            logger.warning(f"Not enough transitions in buffer ({buffer_size}) to sample batch of size {batch_size}")
            return None
        
        # Calculate current beta value
        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        self.frame += 1
        
        # Calculate sampling probabilities
        priorities = self.priorities[:buffer_size]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices based on priorities
        indices = np.random.choice(buffer_size, batch_size, replace=False, p=probs)
        
        # Get samples from indices
        samples = [self.buffer[idx] for idx in indices]
        
        # Process samples and handle 'same_as_state' references
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for state, action, reward, next_state, done in samples:
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            # Handle 'same_as_state' reference
            if next_state == 'same_as_state':
                next_states.append(state)  # Use the state as next_state
            else:
                next_states.append(next_state)
            dones.append(done)
        
        # Calculate importance-sampling weights
        weights = (buffer_size * probs[indices]) ** (-beta)
        weights /= weights.max()  # Normalize to maximum weight
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
        
        # Convert to arrays then to tensors
        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.long, device=self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones = torch.tensor(np.array(dones), dtype=torch.bool, device=self.device)
        
        # Track sample time
        self.sample_times.append(time.time() - start_time)
        
        return states, actions, rewards, next_states, dones, weights, indices
    
    def update_priorities(self, indices, priorities):
        """
        Update priorities for sampled transitions
        
        Args:
            indices: Indices of transitions to update
            priorities: New priority values
        """
        start_time = time.time()
        
        for idx, priority in zip(indices, priorities):
            if idx < len(self.buffer):
                self.priorities[idx] = priority
        
        # Track update time
        self.update_times.append(time.time() - start_time)
    
    def __len__(self):
        """Return the current size of the buffer"""
        return len(self.buffer)
    
    def is_ready(self, batch_size):
        """Check if buffer has enough samples for a batch"""
        return len(self) >= batch_size
    
    def clear(self):
        """Clear the prioritized replay buffer"""
        self.buffer = []
        self.priorities = np.zeros((self.capacity,), dtype=np.float32)
        self.position = 0
        self.total_bytes = 0
        
        # Force garbage collection
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info("Prioritized replay buffer cleared and memory freed")
    
    def get_performance_stats(self):
        """
        Get performance statistics about the prioritized replay buffer
        
        Returns:
            Dictionary of performance statistics
        """
        stats = {
            'push_time': {
                'mean': np.mean(self.push_times) if self.push_times else 0,
                'std': np.std(self.push_times) if self.push_times else 0,
                'count': len(self.push_times)
            },
            'sample_time': {
                'mean': np.mean(self.sample_times) if self.sample_times else 0,
                'std': np.std(self.sample_times) if self.sample_times else 0,
                'count': len(self.sample_times)
            },
            'update_time': {
                'mean': np.mean(self.update_times) if self.update_times else 0,
                'std': np.std(self.update_times) if self.update_times else 0,
                'count': len(self.update_times)
            },
            'memory': {
                'size': len(self.buffer),
                'capacity': self.capacity,
                'utilization': len(self.buffer) / self.capacity if self.capacity > 0 else 0,
                'estimated_mb': self.estimated_memory_usage,
                'actual_mb': self.total_bytes / (1024 * 1024)
            },
            'beta': min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        }
        
        return stats
    
    def log_performance_stats(self):
        """Log performance statistics to the log file"""
        stats = self.get_performance_stats()
        
        logger.info("PrioritizedReplayBuffer Performance Statistics:")
        logger.info(f"  Push time: {stats['push_time']['mean']*1000:.2f}ms (±{stats['push_time']['std']*1000:.2f}ms)")
        logger.info(f"  Sample time: {stats['sample_time']['mean']*1000:.2f}ms (±{stats['sample_time']['std']*1000:.2f}ms)")
        logger.info(f"  Update time: {stats['update_time']['mean']*1000:.2f}ms (±{stats['update_time']['std']*1000:.2f}ms)")
        logger.info(f"  Buffer utilization: {stats['memory']['utilization']*100:.1f}% ({stats['memory']['size']}/{stats['memory']['capacity']})")
        logger.info(f"  Memory usage: {stats['memory']['actual_mb']:.2f} MB (estimated: {stats['memory']['estimated_mb']:.2f} MB)")
        logger.info(f"  Current beta: {stats['beta']:.4f}")


class EfficientPrioritizedReplayBuffer:
    """
    Memory-efficient implementation of Prioritized Experience Replay
    - Uses NumPy arrays for efficient storage
    - Implements Sum Tree for fast priority-based sampling
    - Optimized for limited VRAM environments
    """
    def __init__(self, capacity, state_shape, device="cpu", alpha=0.6, beta_start=0.4, beta_frames=100000):
        """
        Initialize an efficient prioritized replay buffer
        
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
        
        # Initialize buffer with numpy arrays
        self.states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        
        # Tracking
        self.position = 0
        self.size = 0
        
        # For efficient priority-based sampling, use a sum tree
        self.initialize_sum_tree(capacity)
        
        # Constants
        self.max_priority = 1.0
        self.epsilon = 1e-6  # Small constant to ensure non-zero probabilities
        
        # Performance tracking
        self.push_times = []
        self.sample_times = []
        self.update_times = []
        
        # Calculate and log memory usage
        bytes_per_float = 4  # 32-bit float
        state_size = 2 * np.prod(state_shape) * bytes_per_float  # state and next_state
        action_reward_done_size = 8 + 4 + 1  # action (int64), reward (float32), done (bool)
        sum_tree_size = 2 * capacity * 4  # float32 nodes in sum tree
        
        self.estimated_memory_usage = (capacity * (state_size + action_reward_done_size) + sum_tree_size) / (1024 * 1024)  # MB
        
        logger.info(f"EfficientPrioritizedReplayBuffer initialized with capacity {capacity}, alpha={alpha}")
        logger.info(f"Estimated memory usage: {self.estimated_memory_usage:.2f} MB")
    
    def initialize_sum_tree(self, capacity):
        """Initialize a sum tree data structure for efficient priority-based sampling"""
        # For a sum tree, we need 2*capacity-1 nodes
        # The first capacity-1 nodes are internal nodes, the last capacity nodes are leaf nodes
        tree_capacity = 2 * capacity - 1
        self.sum_tree = np.zeros(tree_capacity, dtype=np.float32)
    
    def _propagate(self, idx, change):
        """Propagate a priority change up the tree"""
        parent = (idx - 1) // 2
        self.sum_tree[parent] += change
        
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx, s):
        """Retrieve sample index from a value s"""
        left = 2 * idx + 1
        right = left + 1
        
        # If we're at a leaf node, return the index
        if left >= len(self.sum_tree):
            return idx
        
        # Otherwise, recursively search for the correct sample
        if s <= self.sum_tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.sum_tree[left])
    
    def total_priority(self):
        """Get the total priority in the tree"""
        return self.sum_tree[0]
    
    def add(self, priority):
        """Add a priority value to the tree"""
        # Convert leaf index to tree index
        tree_idx = self.position + self.capacity - 1
        
        # Update the tree
        self.sum_tree[tree_idx] = priority
        self._propagate(tree_idx, priority)
    
    def update(self, tree_idx, priority):
        """Update a priority value in the tree"""
        # Calculate the change in priority
        change = priority - self.sum_tree[tree_idx]
        
        # Update the tree
        self.sum_tree[tree_idx] = priority
        self._propagate(tree_idx, change)
    
    def get_leaf(self, s):
        """Get a leaf node index based on a value s"""
        idx = self._retrieve(0, s)
        data_idx = idx - (self.capacity - 1)
        return idx, data_idx, self.sum_tree[idx]
    
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
        start_time = time.time()
        
        # Ensure state and next_state are numpy arrays
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.cpu().numpy()
        
        # Store transition
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done
        
        # Add with maximum priority
        priority = self.max_priority ** self.alpha
        self.add(priority)
        
        # Update position and size
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
        # Track push time
        self.push_times.append(time.time() - start_time)
    
    def sample(self, batch_size):
        """
        Sample a batch of transitions based on priorities
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones, weights, indices)
        """
        start_time = time.time()
        
        if self.size < batch_size:
            logger.warning(f"Not enough transitions in buffer ({self.size}) to sample batch of size {batch_size}")
            return None
        
        # Current beta value for importance sampling weights
        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        self.frame += 1
        
        # Calculate the segment size
        total_priority = self.total_priority()
        segment = total_priority / batch_size
        
        # Lists to store batch data
        tree_indices = np.zeros(batch_size, dtype=np.int32)
        data_indices = np.zeros(batch_size, dtype=np.int32)
        priorities = np.zeros(batch_size, dtype=np.float32)
        
        # Sample from each segment
        for i in range(batch_size):
            # Get a value within the segment
            a, b = segment * i, segment * (i + 1)
            s = np.random.uniform(a, b)
            
            # Get the leaf node
            tree_idx, data_idx, priority = self.get_leaf(s)
            
            # Store indices and priorities
            tree_indices[i] = tree_idx
            data_indices[i] = data_idx
            priorities[i] = priority
        
        # Calculate importance sampling weights
        p_min = np.min(priorities) / total_priority
        max_weight = (p_min * self.size) ** (-beta)
        
        weights = np.zeros(batch_size, dtype=np.float32)
        for i in range(batch_size):
            p_sample = priorities[i] / total_priority
            weight = (p_sample * self.size) ** (-beta)
            weights[i] = weight / max_weight
        
        # Get batch data from arrays
        states = torch.tensor(self.states[data_indices], dtype=torch.float32, device=self.device)
        actions = torch.tensor(self.actions[data_indices], dtype=torch.long, device=self.device)
        rewards = torch.tensor(self.rewards[data_indices], dtype=torch.float32, device=self.device)
        next_states = torch.tensor(self.next_states[data_indices], dtype=torch.float32, device=self.device)
        dones = torch.tensor(self.dones[data_indices], dtype=torch.bool, device=self.device)
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
        
        # Track sample time
        self.sample_times.append(time.time() - start_time)
        
        return states, actions, rewards, next_states, dones, weights, tree_indices
    
    def update_priorities(self, tree_indices, priorities):
        """
        Update priorities for sampled transitions
        
        Args:
            tree_indices: Tree indices of transitions to update
            priorities: New priority values
        """
        start_time = time.time()
        
        for idx, priority in zip(tree_indices, priorities):
            # Add epsilon to ensure all transitions can be sampled
            priority = (priority + self.epsilon) ** self.alpha
            
            # Update max priority
            self.max_priority = max(self.max_priority, priority)
            
            # Update tree
            self.update(idx, priority)
        
        # Track update time
        self.update_times.append(time.time() - start_time)
    
    def __len__(self):
        """Return the current size of the buffer"""
        return self.size
    
    def is_ready(self, batch_size):
        """Check if buffer has enough samples for a batch"""
        return self.size >= batch_size
    
    def clear(self):
        """Clear the buffer"""
        self.position = 0
        self.size = 0
        
        # Reset arrays and sum tree
        self.states.fill(0)
        self.next_states.fill(0)
        self.actions.fill(0)
        self.rewards.fill(0)
        self.dones.fill(0)
        self.sum_tree.fill(0)
        
        # Reset max priority
        self.max_priority = 1.0
        
        # Force garbage collection
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Efficient prioritized replay buffer cleared and memory freed")
    
    def get_performance_stats(self):
        """
        Get performance statistics about the buffer
        
        Returns:
            Dictionary of performance statistics
        """
        stats = {
            'push_time': {
                'mean': np.mean(self.push_times) if self.push_times else 0,
                'std': np.std(self.push_times) if self.push_times else 0,
                'count': len(self.push_times)
            },
            'sample_time': {
                'mean': np.mean(self.sample_times) if self.sample_times else 0,
                'std': np.std(self.sample_times) if self.sample_times else 0,
                'count': len(self.sample_times)
            },
            'update_time': {
                'mean': np.mean(self.update_times) if self.update_times else 0,
                'std': np.std(self.update_times) if self.update_times else 0,
                'count': len(self.update_times)
            },
            'memory': {
                'size': self.size,
                'capacity': self.capacity,
                'utilization': self.size / self.capacity if self.capacity > 0 else 0,
                'estimated_mb': self.estimated_memory_usage
            },
            'beta': min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames),
            'max_priority': self.max_priority
        }
        
        return stats
    
    def log_performance_stats(self):
        """Log performance statistics to the log file"""
        stats = self.get_performance_stats()
        
        logger.info("EfficientPrioritizedReplayBuffer Performance Statistics:")
        logger.info(f"  Push time: {stats['push_time']['mean']*1000:.2f}ms (±{stats['push_time']['std']*1000:.2f}ms)")
        logger.info(f"  Sample time: {stats['sample_time']['mean']*1000:.2f}ms (±{stats['sample_time']['std']*1000:.2f}ms)")
        logger.info(f"  Update time: {stats['update_time']['mean']*1000:.2f}ms (±{stats['update_time']['std']*1000:.2f}ms)")
        logger.info(f"  Buffer utilization: {stats['memory']['utilization']*100:.1f}% ({stats['memory']['size']}/{stats['memory']['capacity']})")
        logger.info(f"  Estimated memory usage: {stats['memory']['estimated_mb']:.2f} MB")
        logger.info(f"  Current beta: {stats['beta']:.4f}, Max priority: {stats['max_priority']:.4f}")