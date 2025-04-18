# model/agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import logging
import os
import time
import gc
from datetime import datetime
import matplotlib.pyplot as plt

# Import custom modules - use improved versions
from model.dqn import DQN, DuelingDQN, EfficientDuelingDQN
from model.experience_replay import ReplayBuffer, EfficientReplayBuffer, PrioritizedReplayBuffer, EfficientPrioritizedReplayBuffer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DQNAgent")

class DQNAgent:
    def __init__(self, state_shape, n_actions, checkpoint_dir="./logs/checkpoints/", 
                 use_dueling=True, use_double=True, use_per=True, device=None,
                 memory_efficient=True):
        """
        Initialize the DQN Agent
        
        Args:
            state_shape: Shape of the input state (frames, height, width)
            n_actions: Number of possible actions
            checkpoint_dir: Directory to save model checkpoints
            use_dueling: Whether to use Dueling DQN architecture
            use_double: Whether to use Double DQN algorithm
            use_per: Whether to use Prioritized Experience Replay
            device: Device to run the model on (None for auto-detection)
            memory_efficient: Whether to use memory-efficient implementations
        """
        # Set device (GPU or CPU)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.checkpoint_dir = checkpoint_dir
        self.use_dueling = use_dueling
        self.use_double = use_double
        self.use_per = use_per
        self.memory_efficient = memory_efficient
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # DQN Networks: policy network and target network (for stability)
        if memory_efficient:
            if use_dueling:
                logger.info("Using memory-efficient Dueling DQN architecture")
                self.policy_net = EfficientDuelingDQN(state_shape, n_actions).to(self.device)
                self.target_net = EfficientDuelingDQN(state_shape, n_actions).to(self.device)
            else:
                logger.info("Using memory-efficient standard DQN architecture")
                self.policy_net = DQN(state_shape, n_actions).to(self.device)
                self.target_net = DQN(state_shape, n_actions).to(self.device)
        else:
            if use_dueling:
                logger.info("Using standard Dueling DQN architecture")
                self.policy_net = DuelingDQN(state_shape, n_actions).to(self.device)
                self.target_net = DuelingDQN(state_shape, n_actions).to(self.device)
            else:
                logger.info("Using standard DQN architecture")
                self.policy_net = DQN(state_shape, n_actions).to(self.device)
                self.target_net = DQN(state_shape, n_actions).to(self.device)
            
        # Initialize target network with policy network's weights
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Set target network to evaluation mode
        
        # Print model size
        self._log_model_size()
        
        # Hyperparameters
        self.gamma = 0.99  # Discount factor
        self.batch_size = 32
        self.learning_rate = 0.0005
        self.target_update = 10  # Update target network every 10 episodes
        self.epsilon_start = 1.0
        self.epsilon_end = 0.05
        self.epsilon_decay = 0.995
        self.epsilon = self.epsilon_start
        
        # Experience replay
        self.memory_capacity = 10000
        
        # Create appropriate replay buffer based on settings
        if memory_efficient:
            if use_per:
                logger.info("Using EfficientPrioritizedReplayBuffer")
                self.memory = EfficientPrioritizedReplayBuffer(
                    self.memory_capacity, state_shape, device=self.device
                )
            else:
                logger.info("Using EfficientReplayBuffer")
                self.memory = EfficientReplayBuffer(
                    self.memory_capacity, state_shape, device=self.device
                )
        else:
            if use_per:
                logger.info("Using PrioritizedReplayBuffer")
                self.memory = PrioritizedReplayBuffer(
                    self.memory_capacity, state_shape, device=self.device
                )
            else:
                logger.info("Using standard ReplayBuffer")
                self.memory = ReplayBuffer(
                    self.memory_capacity, state_shape, device=self.device
                )
        
        # Optimizer with gradient clipping
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
        # Loss function
        self.criterion = nn.SmoothL1Loss(reduction='none')  # Huber loss for stability
        
        # Training metrics
        self.episode_count = 0
        self.total_steps = 0
        self.learn_steps = 0
        self.episode_rewards = []
        self.loss_history = []
        self.q_value_history = []
        self.epsilon_history = []
        
        # Performance tracking
        self.selection_times = []
        self.learn_times = []
        
        # Success rate tracking
        self.success_history = []  # Track successful episodes
        self.score_history = []    # Track final scores
        
        logger.info(f"Using {'Double' if use_double else 'Standard'} Q-learning")
        logger.info(f"Running on device: {self.device}")
        
        # Check available memory and adjust batch size if needed
        if self.device.type == 'cuda':
            self._check_and_adjust_memory()
    
    def _log_model_size(self):
        """Log model size information"""
        policy_params = sum(p.numel() for p in self.policy_net.parameters())
        
        logger.info(f"Model has {policy_params:,} parameters")
        
        # Estimate model size in memory
        bytes_per_param = 4  # Each parameter is a float32 (4 bytes)
        model_size_bytes = policy_params * bytes_per_param
        model_size_mb = model_size_bytes / (1024 * 1024)
        
        # Double for both policy and target networks
        total_model_size_mb = model_size_mb * 2
        
        logger.info(f"Estimated model memory usage: {total_model_size_mb:.2f} MB (both networks)")
    
    def _check_and_adjust_memory(self):
        """Check available GPU memory and adjust batch size if needed"""
        if not torch.cuda.is_available():
            return
            
        try:
            # Get total and free memory
            total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**2)  # MB
            reserved_mem = torch.cuda.memory_reserved(0) / (1024**2)  # MB
            allocated_mem = torch.cuda.memory_allocated(0) / (1024**2)  # MB
            free_mem = total_mem - reserved_mem
            
            logger.info(f"GPU memory: {total_mem:.0f}MB total, {free_mem:.0f}MB free, {allocated_mem:.0f}MB allocated")
            
            # If we have less than 1GB free, reduce batch size
            if free_mem < 1024:
                old_batch_size = self.batch_size
                self.batch_size = max(8, self.batch_size // 2)
                logger.warning(f"Low GPU memory detected ({free_mem:.0f}MB free). Reducing batch size from {old_batch_size} to {self.batch_size}")
                
                # Also reduce replay buffer size if extremely low memory
                if free_mem < 512 and self.memory_capacity > 5000:
                    old_capacity = self.memory_capacity
                    self.memory_capacity = 5000
                    logger.warning(f"Very low GPU memory. Reducing replay buffer capacity from {old_capacity} to {self.memory_capacity}")
                    # Recreate memory with new capacity
                    self._recreate_memory()
        except Exception as e:
            logger.warning(f"Error checking GPU memory: {e}")
    
    def _recreate_memory(self):
        """Recreate memory buffer with current settings (used when capacity changes)"""
        # Store old memory contents
        old_memory = self.memory
        
        # Create new memory with current settings
        if self.memory_efficient:
            if self.use_per:
                self.memory = EfficientPrioritizedReplayBuffer(
                    self.memory_capacity, self.state_shape, device=self.device
                )
            else:
                self.memory = EfficientReplayBuffer(
                    self.memory_capacity, self.state_shape, device=self.device
                )
        else:
            if self.use_per:
                self.memory = PrioritizedReplayBuffer(
                    self.memory_capacity, self.state_shape, device=self.device
                )
            else:
                self.memory = ReplayBuffer(
                    self.memory_capacity, self.state_shape, device=self.device
                )
                
        # Copy over some transitions if possible
        try:
            if hasattr(old_memory, 'buffer'):
                # For PER
                buffer_size = min(len(old_memory.buffer), self.memory_capacity)
                for i in range(buffer_size):
                    transition = old_memory.buffer[i]
                    self.memory.push(*transition)
            elif hasattr(old_memory, 'memory'):
                # For standard buffer
                buffer_size = min(len(old_memory.memory), self.memory_capacity)
                for i in range(buffer_size):
                    transition = old_memory.memory[i]
                    self.memory.push(*transition)
        except Exception as e:
            logger.warning(f"Error copying old memory contents: {e}")
            
        # Force garbage collection to free old memory
        del old_memory
        gc.collect()
        torch.cuda.empty_cache()
    
    def select_action(self, state, training=True):
        """
        Select an action using epsilon-greedy policy
        
        Args:
            state: Current state (numpy array or tensor)
            training: Whether the agent is training (if False, always choose best action)
            
        Returns:
            Selected action (integer)
        """
        start_time = time.time()
        
        # Convert state to tensor if it's a numpy array
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # Epsilon-greedy action selection
        if training and random.random() < self.epsilon:
            # Random action
            action = random.randint(0, self.n_actions - 1)
        else:
            # Best action according to policy network
            with torch.no_grad():
                q_values = self.policy_net(state)
                
                # Log Q-values periodically
                if self.total_steps % 100 == 0:
                    self.q_value_history.append(q_values.cpu().numpy().mean())
                
                action = q_values.max(1)[1].item()
        
        # Track selection time
        self.selection_times.append(time.time() - start_time)
        
        return action
    
    def decay_epsilon(self):
        """
        Decay epsilon based on episode rewards
        Using adaptive decay based on performance
        """
        # Check if we have enough episodes to evaluate performance
        if len(self.episode_rewards) >= 10:
            avg_reward = np.mean(self.episode_rewards[-10:])
            
            # Adaptive decay - decay faster if rewards are higher
            if avg_reward > 50:
                decay_rate = 0.98  # Faster decay for good performance
            elif avg_reward > 20:
                decay_rate = 0.985  # Medium decay for decent performance
            else:
                decay_rate = self.epsilon_decay  # Default decay rate for poor performance
            
            self.epsilon = max(self.epsilon_end, self.epsilon * decay_rate)
        else:
            # Use default decay rate if not enough episodes
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # Log epsilon value
        self.epsilon_history.append(self.epsilon)
    
    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in replay memory
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode ended
        """
        self.memory.push(state, action, reward, next_state, done)
        self.total_steps += 1
    
    def learn(self):
        """
        Train the policy network using experiences from the replay buffer
        
        Returns:
            Loss value if learning occurred, None otherwise
        """
        start_time = time.time()
        
        # Check if we have enough samples for a batch
        if not self.memory.is_ready(self.batch_size):
            return None
        
        # Sample from replay buffer
        if self.use_per:
            states, actions, rewards, next_states, dones, weights, indices = self.memory.sample(self.batch_size)
        else:
            states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
            weights = torch.ones_like(rewards)  # Uniform weights when not using PER
        
        # Calculate Q-values for current states and actions
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Calculate target Q-values
        if self.use_double:
            # Double DQN: use policy network to select actions, target network to evaluate them
            with torch.no_grad():
                # Get actions from policy network
                policy_next_q_values = self.policy_net(next_states)
                next_actions = policy_next_q_values.max(1)[1].unsqueeze(1)
                
                # Evaluate actions using target network
                next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)
                
                # Set Q-value of terminal states to 0
                next_q_values[dones] = 0.0
                
                # Calculate target Q-values
                target_q_values = rewards + self.gamma * next_q_values
        else:
            # Standard DQN: use target network for both selection and evaluation
            with torch.no_grad():
                next_q_values = self.target_net(next_states).max(1)[0]
                next_q_values[dones] = 0.0
                target_q_values = rewards + self.gamma * next_q_values
        
        # Calculate loss
        losses = self.criterion(q_values, target_q_values)
        
        # Apply importance sampling weights if using PER
        weighted_losses = losses * weights
        loss = weighted_losses.mean()
        
        # Update priorities in PER buffer if using it
        if self.use_per:
            td_errors = torch.abs(target_q_values - q_values).detach().cpu().numpy()
            self.memory.update_priorities(indices, td_errors + 1e-6)  # Small constant for stability
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients to prevent exploding gradients
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
            
        self.optimizer.step()
        
        self.learn_steps += 1
        
        # Track loss
        loss_value = loss.item()
        self.loss_history.append(loss_value)
        
        # Track learn time
        self.learn_times.append(time.time() - start_time)
        
        return loss_value
    
    def update_target_network(self):
        """Update the target network with the policy network's weights"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
        logger.debug("Target network updated")
    
    def add_episode_reward(self, reward, score=None):
        """
        Add episode reward to history
        
        Args:
            reward: Total reward for the episode
            score: Final game score (optional)
        """
        self.episode_rewards.append(reward)
        self.episode_count += 1
        
        # Track score if provided
        if score is not None:
            self.score_history.append(score)
        
        # Track success (defined as reward > 0)
        success = reward > 0
        self.success_history.append(success)
    
    def save_checkpoint(self, episode, rewards=None, avg_reward=None, suffix=""):
        """
        Save a checkpoint of the agent
        
        Args:
            episode: Current episode number
            rewards: List of rewards (optional)
            avg_reward: Average reward (optional)
            suffix: Additional suffix for filename (optional)
            
        Returns:
            Path to saved checkpoint
        """
        # Create timestamp for the checkpoint
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create checkpoint filename
        if suffix:
            filename = f"dqn_episode_{episode}_{suffix}_{timestamp}.pt"
        else:
            filename = f"dqn_episode_{episode}_{timestamp}.pt"
        
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        # Save model state
        checkpoint = {
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'episode': episode,
            'epsilon': self.epsilon,
            'total_steps': self.total_steps,
            'learn_steps': self.learn_steps,
            'loss_history': self.loss_history,
            'q_value_history': self.q_value_history,
            'epsilon_history': self.epsilon_history,
            'episode_rewards': self.episode_rewards,
            'success_history': self.success_history,
            'score_history': self.score_history,
            'avg_reward': avg_reward,
            'state_shape': self.state_shape,
            'n_actions': self.n_actions,
            'use_dueling': self.use_dueling,
            'use_double': self.use_double,
            'use_per': self.use_per,
            'memory_efficient': self.memory_efficient,
            'hyperparameters': {
                'gamma': self.gamma,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'target_update': self.target_update,
                'epsilon_start': self.epsilon_start,
                'epsilon_end': self.epsilon_end,
                'epsilon_decay': self.epsilon_decay
            }
        }
        
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved: {filepath}")
        
        return filepath
    
    def load_checkpoint(self, filepath):
        """
        Load a checkpoint
        
        Args:
            filepath: Path to the checkpoint file
            
        Returns:
            Episode number and epsilon value from the checkpoint
        """
        if not os.path.exists(filepath):
            logger.error(f"Checkpoint file not found: {filepath}")
            return 0, self.epsilon_start
        
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            
            # Load model state
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            
            # Load training state
            episode = checkpoint.get('episode', 0)
            self.epsilon = checkpoint.get('epsilon', self.epsilon_start)
            self.total_steps = checkpoint.get('total_steps', 0)
            self.learn_steps = checkpoint.get('learn_steps', 0)
            
            # Load history if available
            if 'loss_history' in checkpoint:
                self.loss_history = checkpoint['loss_history']
            if 'q_value_history' in checkpoint:
                self.q_value_history = checkpoint['q_value_history']
            if 'epsilon_history' in checkpoint:
                self.epsilon_history = checkpoint['epsilon_history']
            if 'episode_rewards' in checkpoint:
                self.episode_rewards = checkpoint['episode_rewards']
            if 'success_history' in checkpoint:
                self.success_history = checkpoint['success_history']
            if 'score_history' in checkpoint:
                self.score_history = checkpoint['score_history']
                
            # Load hyperparameters if available
            if 'hyperparameters' in checkpoint:
                hyperparameters = checkpoint['hyperparameters']
                self.gamma = hyperparameters.get('gamma', self.gamma)
                self.batch_size = hyperparameters.get('batch_size', self.batch_size)
                self.learning_rate = hyperparameters.get('learning_rate', self.learning_rate)
                self.target_update = hyperparameters.get('target_update', self.target_update)
                self.epsilon_start = hyperparameters.get('epsilon_start', self.epsilon_start)
                self.epsilon_end = hyperparameters.get('epsilon_end', self.epsilon_end)
                self.epsilon_decay = hyperparameters.get('epsilon_decay', self.epsilon_decay)
                
                # Update optimizer learning rate
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.learning_rate
            
            logger.info(f"Checkpoint loaded: {filepath}")
            logger.info(f"Resuming from episode {episode} with epsilon {self.epsilon:.4f}")
            
            return episode, self.epsilon
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")
            return 0, self.epsilon_start
    
    def plot_metrics(self, save_dir='./logs/plots'):
        """
        Plot training metrics and save to disk
        
        Args:
            save_dir: Directory to save plots
        """
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Plot loss history
        if self.loss_history:
            plt.figure(figsize=(10, 5))
            plt.plot(self.loss_history)
            plt.title('Training Loss')
            plt.xlabel('Learning Steps')
            plt.ylabel('Loss')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(save_dir, f'loss_history_{timestamp}.png'))
            plt.close()
        
        # Plot Q-value history
        if self.q_value_history:
            plt.figure(figsize=(10, 5))
            plt.plot(self.q_value_history)
            plt.title('Average Q-Value')
            plt.xlabel('Steps (x100)')
            plt.ylabel('Q-Value')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(save_dir, f'q_value_history_{timestamp}.png'))
            plt.close()
        
        # Plot epsilon history
        if self.epsilon_history:
            plt.figure(figsize=(10, 5))
            plt.plot(self.epsilon_history)
            plt.title('Epsilon Decay')
            plt.xlabel('Episodes')
            plt.ylabel('Epsilon')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(save_dir, f'epsilon_history_{timestamp}.png'))
            plt.close()
        
        # Plot episode rewards
        if self.episode_rewards:
            plt.figure(figsize=(10, 5))
            plt.plot(self.episode_rewards, alpha=0.6)
            
            # Plot moving average if enough episodes
            if len(self.episode_rewards) >= 10:
                moving_avg = np.convolve(self.episode_rewards, np.ones(10)/10, mode='valid')
                plt.plot(range(9, len(self.episode_rewards)), moving_avg, 'r-', label='10-Episode Moving Average')
                
            plt.title('Episode Rewards')
            plt.xlabel('Episodes')
            plt.ylabel('Reward')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.savefig(os.path.join(save_dir, f'episode_rewards_{timestamp}.png'))
            plt.close()
        
        # Plot success rate
        if self.success_history:
            plt.figure(figsize=(10, 5))
            
            # Calculate success rate using a moving window
            window_size = 10
            if len(self.success_history) >= window_size:
                success_rates = []
                for i in range(len(self.success_history) - window_size + 1):
                    window = self.success_history[i:i+window_size]
                    success_rates.append(sum(window) / window_size)
                
                plt.plot(range(window_size-1, len(self.success_history)), success_rates)
                plt.title('Success Rate (10-Episode Moving Window)')
                plt.xlabel('Episodes')
                plt.ylabel('Success Rate')
                plt.grid(True, alpha=0.3)
                plt.ylim(0, 1)
                plt.savefig(os.path.join(save_dir, f'success_rate_{timestamp}.png'))
                plt.close()
        
        # Plot combined rewards and epsilon
        if self.episode_rewards and self.epsilon_history:
            plt.figure(figsize=(10, 6))
            
            # Plot rewards on primary y-axis
            ax1 = plt.gca()
            ax1.set_xlabel('Episodes')
            ax1.set_ylabel('Reward', color='tab:blue')
            ax1.plot(self.episode_rewards, alpha=0.5, color='tab:blue', label='Reward')
            
            if len(self.episode_rewards) >= 10:
                moving_avg = np.convolve(self.episode_rewards, np.ones(10)/10, mode='valid')
                ax1.plot(range(9, len(self.episode_rewards)), moving_avg, color='tab:blue', 
                        linewidth=2, label='10-Episode Moving Average')
            
            ax1.tick_params(axis='y', labelcolor='tab:blue')
            
            # Add secondary y-axis for epsilon
            ax2 = ax1.twinx()
            ax2.set_ylabel('Epsilon', color='tab:red')
            ax2.plot(self.epsilon_history, color='tab:red', label='Epsilon')
            ax2.tick_params(axis='y', labelcolor='tab:red')
            
            # Add legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center')
            
            plt.title('Rewards and Exploration Rate')
            plt.grid(True, alpha=0.2)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'rewards_epsilon_{timestamp}.png'))
            plt.close()
    
    def get_performance_stats(self):
        """
        Get statistics about agent performance
        
        Returns:
            Dictionary of performance statistics
        """
        stats = {
            'action_selection_time': {
                'mean': np.mean(self.selection_times) if self.selection_times else 0,
                'std': np.std(self.selection_times) if self.selection_times else 0,
                'min': np.min(self.selection_times) if self.selection_times else 0,
                'max': np.max(self.selection_times) if self.selection_times else 0,
                'count': len(self.selection_times)
            },
            'learning_time': {
                'mean': np.mean(self.learn_times) if self.learn_times else 0,
                'std': np.std(self.learn_times) if self.learn_times else 0,
                'min': np.min(self.learn_times) if self.learn_times else 0,
                'max': np.max(self.learn_times) if self.learn_times else 0,
                'count': len(self.learn_times)
            },
            'rewards': {
                'mean': np.mean(self.episode_rewards) if self.episode_rewards else 0,
                'std': np.std(self.episode_rewards) if self.episode_rewards else 0,
                'min': np.min(self.episode_rewards) if self.episode_rewards else 0,
                'max': np.max(self.episode_rewards) if self.episode_rewards else 0,
                'last_10_avg': np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else (np.mean(self.episode_rewards) if self.episode_rewards else 0)
            },
            'loss': {
                'mean': np.mean(self.loss_history) if self.loss_history else 0,
                'std': np.std(self.loss_history) if self.loss_history else 0,
                'min': np.min(self.loss_history) if self.loss_history else 0,
                'max': np.max(self.loss_history) if self.loss_history else 0,
                'current': self.loss_history[-1] if self.loss_history else 0
            },
            'steps': self.total_steps,
            'episodes': self.episode_count,
            'learn_steps': self.learn_steps,
            'current_epsilon': self.epsilon,
            'success_rate': {
                'overall': np.mean(self.success_history) if self.success_history else 0,
                'last_10': np.mean(self.success_history[-10:]) if len(self.success_history) >= 10 else (np.mean(self.success_history) if self.success_history else 0)
            }
        }
        
        # Add GPU memory stats if available
        if self.device.type == 'cuda' and torch.cuda.is_available():
            try:
                total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**2)  # MB
                reserved_mem = torch.cuda.memory_reserved(0) / (1024**2)  # MB
                allocated_mem = torch.cuda.memory_allocated(0) / (1024**2)  # MB
                free_mem = total_mem - reserved_mem
                
                stats['gpu_memory'] = {
                    'total_mb': total_mem,
                    'reserved_mb': reserved_mem,
                    'allocated_mb': allocated_mem,
                    'free_mb': free_mem,
                    'utilization': allocated_mem / total_mem
                }
            except Exception as e:
                logger.warning(f"Error getting GPU memory stats: {e}")
        
        return stats
    
    def log_performance_stats(self):
        """Log performance statistics to the log file"""
        stats = self.get_performance_stats()
        
        logger.info("DQNAgent Performance Statistics:")
        logger.info(f"  Episodes: {stats['episodes']}, Total Steps: {stats['steps']}, Learning Steps: {stats['learn_steps']}")
        logger.info(f"  Current Epsilon: {stats['current_epsilon']:.4f}")
        logger.info(f"  Action Selection Time: {stats['action_selection_time']['mean']*1000:.2f}ms (±{stats['action_selection_time']['std']*1000:.2f}ms)")
        logger.info(f"  Learning Time: {stats['learning_time']['mean']*1000:.2f}ms (±{stats['learning_time']['std']*1000:.2f}ms)")
        logger.info(f"  Average Reward: {stats['rewards']['mean']:.2f}, Last 10 Average: {stats['rewards']['last_10_avg']:.2f}")
        logger.info(f"  Current Loss: {stats['loss']['current']:.6f}, Average Loss: {stats['loss']['mean']:.6f}")
        logger.info(f"  Success Rate: {stats['success_rate']['overall']*100:.1f}%, Last 10: {stats['success_rate']['last_10']*100:.1f}%")
        
        # Log memory if available
        if 'gpu_memory' in stats:
            logger.info(f"  GPU Memory: {stats['gpu_memory']['allocated_mb']:.0f}MB used / {stats['gpu_memory']['total_mb']:.0f}MB total ({stats['gpu_memory']['utilization']*100:.1f}%)")
            
        # Log memory info from buffer if available
        if hasattr(self.memory, 'get_performance_stats'):
            try:
                memory_stats = self.memory.get_performance_stats()
                if 'memory' in memory_stats:
                    logger.info(f"  Replay Buffer: {memory_stats['memory']['size']}/{memory_stats['memory']['capacity']} transitions ({memory_stats['memory']['utilization']*100:.1f}%)")
                    if 'estimated_mb' in memory_stats['memory']:
                        logger.info(f"  Buffer Memory Usage: ~{memory_stats['memory']['estimated_mb']:.2f}MB")
            except Exception as e:
                logger.warning(f"Error logging memory stats: {e}")

class AdaptiveDQNAgent(DQNAgent):
    """
    Extension of DQNAgent with adaptive hyperparameters
    that change based on performance and memory constraints
    """
    def __init__(self, state_shape, n_actions, checkpoint_dir="./logs/checkpoints/", 
                 use_dueling=True, use_double=True, use_per=True, device=None,
                 memory_efficient=True):
        """Initialize with same parameters as DQNAgent"""
        super().__init__(state_shape, n_actions, checkpoint_dir, use_dueling, 
                        use_double, use_per, device, memory_efficient)
        
        # Additional metrics for adaptive tuning
        self.adaptation_interval = 10  # Check every 10 episodes
        self.low_memory_threshold = 256  # MB of free GPU memory
        self.high_reward_threshold = 30  # Reward threshold for good performance
        self.low_reward_threshold = 10  # Reward threshold for poor performance
        
        # Adaptation history
        self.adaptation_history = []
        
        logger.info("Using AdaptiveDQNAgent with dynamic hyperparameter tuning")
    
    def adapt_hyperparameters(self):
        """
        Adapt hyperparameters based on performance and memory constraints
        Called periodically during training
        """
        if self.episode_count % self.adaptation_interval != 0:
            return
            
        # Get current performance stats
        stats = self.get_performance_stats()
        
        # Skip if not enough episodes
        if len(self.episode_rewards) < 10:
            return
            
        # Calculate recent performance
        recent_rewards = self.episode_rewards[-10:]
        avg_reward = np.mean(recent_rewards)
        reward_trend = np.mean(recent_rewards[-5:]) - np.mean(recent_rewards[:5])
        
        # Check GPU memory if available
        low_memory = False
        if 'gpu_memory' in stats and self.device.type == 'cuda':
            free_mem = stats['gpu_memory']['free_mb']
            low_memory = free_mem < self.low_memory_threshold
        
        # Initialize changes dictionary
        changes = {}
        
        # Adjust batch size based on memory and performance
        if low_memory and self.batch_size > 16:
            old_batch_size = self.batch_size
            self.batch_size = max(8, self.batch_size // 2)
            changes['batch_size'] = (old_batch_size, self.batch_size)
            
        elif not low_memory and avg_reward > self.high_reward_threshold and self.batch_size < 64:
            old_batch_size = self.batch_size
            self.batch_size = min(64, self.batch_size * 2)
            changes['batch_size'] = (old_batch_size, self.batch_size)
        
        # Adjust learning rate based on performance trend
        if reward_trend < -5 and self.learning_rate > 0.0001:
            # Performance degrading, reduce learning rate
            old_lr = self.learning_rate
            self.learning_rate *= 0.5
            changes['learning_rate'] = (old_lr, self.learning_rate)
            
            # Update optimizer
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.learning_rate
                
        elif reward_trend > 5 and avg_reward < self.high_reward_threshold and self.learning_rate < 0.001:
            # Performance improving but still not great, increase learning rate
            old_lr = self.learning_rate
            self.learning_rate *= 1.5
            changes['learning_rate'] = (old_lr, self.learning_rate)
            
            # Update optimizer
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.learning_rate
        
        # Adjust target network update frequency
        if avg_reward > self.high_reward_threshold and self.target_update > 5:
            old_target_update = self.target_update
            self.target_update = max(5, self.target_update - 2)
            changes['target_update'] = (old_target_update, self.target_update)
            
        elif avg_reward < self.low_reward_threshold and self.target_update < 20:
            old_target_update = self.target_update
            self.target_update = min(20, self.target_update + 2)
            changes['target_update'] = (old_target_update, self.target_update)
        
        # Adjust epsilon decay rate based on performance
        if avg_reward > self.high_reward_threshold and self.epsilon_decay > 0.98:
            old_epsilon_decay = self.epsilon_decay
            self.epsilon_decay = max(0.97, self.epsilon_decay - 0.005)
            changes['epsilon_decay'] = (old_epsilon_decay, self.epsilon_decay)
            
        elif avg_reward < self.low_reward_threshold and self.epsilon_decay < 0.995:
            old_epsilon_decay = self.epsilon_decay
            self.epsilon_decay = min(0.995, self.epsilon_decay + 0.005)
            changes['epsilon_decay'] = (old_epsilon_decay, self.epsilon_decay)
        
        # Log changes if any were made
        if changes:
            adaptation = {
                'episode': self.episode_count,
                'avg_reward': avg_reward,
                'reward_trend': reward_trend,
                'low_memory': low_memory,
                'changes': changes
            }
            self.adaptation_history.append(adaptation)
            
            # Log the changes
            change_str = ", ".join([f"{k}: {v[0]:.6f}->{v[1]:.6f}" for k, v in changes.items()])
            logger.info(f"Adapted hyperparameters at episode {self.episode_count}: {change_str}")
            logger.info(f"Adaptation reason: avg_reward={avg_reward:.2f}, trend={reward_trend:.2f}, low_memory={low_memory}")
    
    def plot_adaptations(self, save_dir='./logs/plots'):
        """
        Plot hyperparameter adaptations over time
        
        Args:
            save_dir: Directory to save plots
        """
        if not self.adaptation_history:
            return
            
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Extract data
        episodes = [a['episode'] for a in self.adaptation_history]
        
        # Group changes by parameter
        params = {}
        for adaptation in self.adaptation_history:
            for param, (old_val, new_val) in adaptation['changes'].items():
                if param not in params:
                    params[param] = {'episodes': [], 'values': []}
                
                # Add data point for the change
                params[param]['episodes'].append(adaptation['episode'])
                params[param]['values'].append(new_val)
        
        # Create plot with subplots for each parameter
        if params:
            n_params = len(params)
            fig, axs = plt.subplots(n_params, 1, figsize=(10, 4 * n_params), sharex=True)
            
            # If only one parameter, axs is not a list
            if n_params == 1:
                axs = [axs]
            
            for i, (param, data) in enumerate(params.items()):
                axs[i].plot(data['episodes'], data['values'], 'o-')
                axs[i].set_ylabel(param)
                axs[i].set_title(f"{param} Adaptation")
                axs[i].grid(True, alpha=0.3)
            
            axs[-1].set_xlabel('Episode')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'hyperparameter_adaptation_{timestamp}.png'))
            plt.close()