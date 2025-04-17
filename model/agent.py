# model/agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import logging
import os
import time
from datetime import datetime
import matplotlib.pyplot as plt

# Import custom modules
from model.dqn import DQN, DuelingDQN
from model.experience_replay import ReplayBuffer, PrioritizedReplayBuffer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DQNAgent")

class DQNAgent:
    def __init__(self, state_shape, n_actions, checkpoint_dir="./logs/checkpoints/", use_dueling=True, 
                 use_double=True, use_per=True, device=None):
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
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # DQN Networks: policy network and target network (for stability)
        if use_dueling:
            self.policy_net = DuelingDQN(state_shape, n_actions).to(self.device)
            self.target_net = DuelingDQN(state_shape, n_actions).to(self.device)
            logger.info(f"DQNAgent initialized with Dueling DQN")
        else:
            self.policy_net = DQN(state_shape, n_actions).to(self.device)
            self.target_net = DQN(state_shape, n_actions).to(self.device)
            logger.info(f"DQNAgent initialized with Standard DQN")
            
        # Initialize target network with policy network's weights
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Set target network to evaluation mode
        
        # Hyperparameters
        self.gamma = 0.99  # Discount factor
        self.batch_size = 32
        self.learning_rate = 0.0005
        self.target_update = 10  # Update target network every 10 episodes
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.995
        self.epsilon = self.epsilon_start
        
        # Experience replay
        self.memory_capacity = 10000
        if use_per:
            self.memory = PrioritizedReplayBuffer(self.memory_capacity, state_shape, device=self.device)
            logger.info(f"Using Prioritized Experience Replay")
        else:
            self.memory = ReplayBuffer(self.memory_capacity, state_shape, device=self.device)
            logger.info(f"Using Standard Experience Replay")
        
        # Optimizer
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
        
        logger.info(f"Using {'Double' if use_double else 'Standard'} Q-learning")
        logger.info(f"Running on device: {self.device}")
    
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
                decay_rate = 0.98  # Faster decay
            elif avg_reward > 20:
                decay_rate = 0.985
            else:
                decay_rate = self.epsilon_decay  # Default decay rate
            
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
    
    def add_episode_reward(self, reward):
        """Add episode reward to history"""
        self.episode_rewards.append(reward)
        self.episode_count += 1
    
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
            'avg_reward': avg_reward,
            'state_shape': self.state_shape,
            'n_actions': self.n_actions,
            'use_dueling': self.use_dueling,
            'use_double': self.use_double,
            'use_per': self.use_per,
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
    
    def plot_metrics(self, save_dir='./logs/'):
        """
        Plot training metrics and save to disk
        
        Args:
            save_dir: Directory to save plots
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Plot loss history
        if self.loss_history:
            plt.figure(figsize=(10, 5))
            plt.plot(self.loss_history)
            plt.title('Training Loss')
            plt.xlabel('Learning Steps')
            plt.ylabel('Loss')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(save_dir, f'loss_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'))
            plt.close()
        
        # Plot Q-value history
        if self.q_value_history:
            plt.figure(figsize=(10, 5))
            plt.plot(self.q_value_history)
            plt.title('Average Q-Value')
            plt.xlabel('Steps (x100)')
            plt.ylabel('Q-Value')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(save_dir, f'q_value_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'))
            plt.close()
        
        # Plot epsilon history
        if self.epsilon_history:
            plt.figure(figsize=(10, 5))
            plt.plot(self.epsilon_history)
            plt.title('Epsilon Decay')
            plt.xlabel('Episodes')
            plt.ylabel('Epsilon')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(save_dir, f'epsilon_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'))
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
            plt.savefig(os.path.join(save_dir, f'episode_rewards_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'))
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
            'current_epsilon': self.epsilon
        }
        
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