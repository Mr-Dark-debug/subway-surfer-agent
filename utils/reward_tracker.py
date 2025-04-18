# utils/reward_tracker.py
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import time
from datetime import datetime
import logging
import sys
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RewardTracker")

class RewardTracker:
    """
    Tracks and analyzes reward components during training
    """
    def __init__(self, log_dir="./logs", window_size=100):
        """
        Initialize the reward tracker
        
        Args:
            log_dir: Directory to save logs and plots
            window_size: Window size for moving average
        """
        self.log_dir = log_dir
        self.window_size = window_size
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize reward tracking
        self.total_rewards = []
        self.survival_rewards = []
        self.score_rewards = []
        self.coin_rewards = []
        self.penalty_rewards = []  # For negative rewards (game over, etc.)
        
        # Episode lengths
        self.episode_lengths = []
        
        # Timestamps for plotting
        self.timestamps = []
        self.start_time = time.time()
        
        # CSV logger
        self.csv_path = os.path.join(log_dir, f"reward_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        self._init_csv_logger()
        
        logger.info(f"RewardTracker initialized with window size {window_size}")
    
    def _init_csv_logger(self):
        """Initialize CSV log file with headers"""
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Episode', 'Total Reward', 'Survival Reward', 'Score Reward', 
                'Coin Reward', 'Penalty Reward', 'Episode Length', 'Timestamp'
            ])
    
    def add_rewards(self, episode, total_reward, survival_reward, score_reward, 
                  coin_reward, penalty_reward, episode_length):
        """
        Add rewards for an episode
        
        Args:
            episode: Episode number
            total_reward: Total reward for the episode
            survival_reward: Reward from surviving
            score_reward: Reward from score increases
            coin_reward: Reward from collecting coins
            penalty_reward: Penalty rewards (negative)
            episode_length: Length of the episode
        """
        # Add rewards to tracking lists
        self.total_rewards.append(total_reward)
        self.survival_rewards.append(survival_reward)
        self.score_rewards.append(score_reward)
        self.coin_rewards.append(coin_reward)
        self.penalty_rewards.append(penalty_reward)
        
        # Add episode length
        self.episode_lengths.append(episode_length)
        
        # Add timestamp (seconds since start)
        self.timestamps.append(time.time() - self.start_time)
        
        # Log to CSV
        self._log_to_csv(episode, total_reward, survival_reward, score_reward,
                        coin_reward, penalty_reward, episode_length)
        
        # Return the moving average of total rewards if enough episodes
        if len(self.total_rewards) >= self.window_size:
            return np.mean(self.total_rewards[-self.window_size:])
        else:
            return np.mean(self.total_rewards)
    
    def _log_to_csv(self, episode, total_reward, survival_reward, score_reward,
                   coin_reward, penalty_reward, episode_length):
        """Log rewards to CSV file"""
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                episode, total_reward, survival_reward, score_reward,
                coin_reward, penalty_reward, episode_length,
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ])
    
    def plot_rewards(self, save=True, show=False):
        """
        Plot reward metrics
        
        Args:
            save: Whether to save the plot to disk
            show: Whether to display the plot
        """
        if not self.total_rewards:
            logger.warning("No rewards to plot")
            return
        
        # Create figure with subplots
        fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
        
        # Plot total rewards
        axs[0].plot(self.total_rewards, alpha=0.6, label='Total Rewards')
        
        # Add moving average if enough episodes
        if len(self.total_rewards) >= self.window_size:
            moving_avg = np.convolve(self.total_rewards, 
                                     np.ones(self.window_size)/self.window_size, 
                                     mode='valid')
            ma_x = np.arange(self.window_size-1, len(self.total_rewards))
            axs[0].plot(ma_x, moving_avg, 'r-', label=f'{self.window_size}-Episode Moving Average')
        
        axs[0].set_title('Total Rewards per Episode')
        axs[0].set_ylabel('Reward')
        axs[0].legend()
        axs[0].grid(True, alpha=0.3)
        
        # Plot reward components
        axs[1].plot(self.survival_rewards, label='Survival', alpha=0.7)
        axs[1].plot(self.score_rewards, label='Score', alpha=0.7)
        axs[1].plot(self.coin_rewards, label='Coins', alpha=0.7)
        axs[1].plot(self.penalty_rewards, label='Penalties', alpha=0.7)
        axs[1].set_title('Reward Components')
        axs[1].set_ylabel('Reward')
        axs[1].legend()
        axs[1].grid(True, alpha=0.3)
        
        # Plot episode lengths
        axs[2].plot(self.episode_lengths, label='Episode Length', alpha=0.7)
        
        # Add moving average if enough episodes
        if len(self.episode_lengths) >= self.window_size:
            length_ma = np.convolve(self.episode_lengths, 
                                   np.ones(self.window_size)/self.window_size, 
                                   mode='valid')
            axs[2].plot(ma_x, length_ma, 'r-', label=f'{self.window_size}-Episode Moving Average')
        
        axs[2].set_title('Episode Lengths')
        axs[2].set_xlabel('Episode')
        axs[2].set_ylabel('Steps')
        axs[2].legend()
        axs[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot if requested
        if save:
            save_path = os.path.join(self.log_dir, f"reward_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            plt.savefig(save_path)
            logger.info(f"Reward metrics plot saved to {save_path}")
        
        # Show plot if requested
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_reward_proportions(self, save=True, show=False):
        """
        Plot the proportion of different reward components as a stacked area chart
        
        Args:
            save: Whether to save the plot to disk
            show: Whether to display the plot
        """
        if not self.total_rewards:
            logger.warning("No rewards to plot")
            return
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Convert penalties to positive for stacking
        penalties_positive = np.array(self.penalty_rewards) * -1
        
        # Calculate absolute totals (ignoring signs) for proportions
        total_abs = (np.array(self.survival_rewards) + 
                     np.array(self.score_rewards) + 
                     np.array(self.coin_rewards) + 
                     penalties_positive)
        
        # Normalize components to percentages of total absolute reward
        survival_norm = 100 * np.array(self.survival_rewards) / np.maximum(total_abs, 1e-6)
        score_norm = 100 * np.array(self.score_rewards) / np.maximum(total_abs, 1e-6)
        coin_norm = 100 * np.array(self.coin_rewards) / np.maximum(total_abs, 1e-6)
        penalty_norm = 100 * penalties_positive / np.maximum(total_abs, 1e-6)
        
        # Create x-axis
        episodes = np.arange(len(self.total_rewards))
        
        # Create stacked area chart
        plt.stackplot(episodes, 
                     [survival_norm, score_norm, coin_norm, penalty_norm],
                     labels=['Survival', 'Score', 'Coins', 'Penalties'],
                     alpha=0.7)
        
        plt.title('Reward Component Proportions')
        plt.xlabel('Episode')
        plt.ylabel('Percentage of Total Absolute Reward')
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Save plot if requested
        if save:
            save_path = os.path.join(self.log_dir, f"reward_proportions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            plt.savefig(save_path)
            logger.info(f"Reward proportions plot saved to {save_path}")
        
        # Show plot if requested
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_time_efficiency(self, save=True, show=False):
        """
        Plot rewards versus training time
        
        Args:
            save: Whether to save the plot to disk
            show: Whether to display the plot
        """
        if not self.total_rewards or not self.timestamps:
            logger.warning("No rewards or timestamps to plot")
            return
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Create scatter plot of rewards vs time
        sc = plt.scatter(self.timestamps, self.total_rewards, 
                        c=np.arange(len(self.total_rewards)), 
                        cmap='viridis', alpha=0.7)
        
        # Add color bar for episode number
        cbar = plt.colorbar(sc)
        cbar.set_label('Episode')
        
        # Add trend line
        z = np.polyfit(self.timestamps, self.total_rewards, 1)
        p = np.poly1d(z)
        plt.plot(self.timestamps, p(self.timestamps), "r--", alpha=0.8, 
                label=f"Trend: {z[0]:.4f}x + {z[1]:.2f}")
        
        plt.title('Rewards vs Training Time')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Total Reward')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save plot if requested
        if save:
            save_path = os.path.join(self.log_dir, f"reward_vs_time_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            plt.savefig(save_path)
            logger.info(f"Reward vs time plot saved to {save_path}")
        
        # Show plot if requested
        if show:
            plt.show()
        else:
            plt.close()
    
    def get_statistics(self):
        """
        Get summary statistics of rewards
        
        Returns:
            Dictionary of reward statistics
        """
        if not self.total_rewards:
            return {}
        
        return {
            'total_episodes': len(self.total_rewards),
            'total_reward': {
                'mean': np.mean(self.total_rewards),
                'std': np.std(self.total_rewards),
                'min': np.min(self.total_rewards),
                'max': np.max(self.total_rewards),
                'latest': self.total_rewards[-1],
                'last_10_avg': np.mean(self.total_rewards[-10:]) if len(self.total_rewards) >= 10 else np.mean(self.total_rewards)
            },
            'episode_length': {
                'mean': np.mean(self.episode_lengths),
                'std': np.std(self.episode_lengths),
                'min': np.min(self.episode_lengths),
                'max': np.max(self.episode_lengths),
                'latest': self.episode_lengths[-1],
                'last_10_avg': np.mean(self.episode_lengths[-10:]) if len(self.episode_lengths) >= 10 else np.mean(self.episode_lengths)
            },
            'reward_components': {
                'survival': {
                    'mean': np.mean(self.survival_rewards),
                    'latest': self.survival_rewards[-1],
                    'percentage': np.mean(self.survival_rewards) / np.mean(np.abs(self.total_rewards)) * 100
                },
                'score': {
                    'mean': np.mean(self.score_rewards),
                    'latest': self.score_rewards[-1],
                    'percentage': np.mean(self.score_rewards) / np.mean(np.abs(self.total_rewards)) * 100
                },
                'coin': {
                    'mean': np.mean(self.coin_rewards),
                    'latest': self.coin_rewards[-1],
                    'percentage': np.mean(self.coin_rewards) / np.mean(np.abs(self.total_rewards)) * 100
                },
                'penalty': {
                    'mean': np.mean(self.penalty_rewards),
                    'latest': self.penalty_rewards[-1],
                    'percentage': np.mean(np.abs(self.penalty_rewards)) / np.mean(np.abs(self.total_rewards)) * 100
                }
            }
        }
    
    def print_statistics(self):
        """Print summary statistics to console"""
        stats = self.get_statistics()
        if not stats:
            logger.warning("No statistics available")
            return
        
        print("\n" + "="*50)
        print("REWARD STATISTICS".center(50))
        print("="*50)
        
        print(f"\nTotal Episodes: {stats['total_episodes']}")
        
        print("\nTOTAL REWARD:")
        print(f"  Mean: {stats['total_reward']['mean']:.2f}")
        print(f"  Std Dev: {stats['total_reward']['std']:.2f}")
        print(f"  Min: {stats['total_reward']['min']:.2f}")
        print(f"  Max: {stats['total_reward']['max']:.2f}")
        print(f"  Latest: {stats['total_reward']['latest']:.2f}")
        print(f"  Last 10 Avg: {stats['total_reward']['last_10_avg']:.2f}")
        
        print("\nEPISODE LENGTH:")
        print(f"  Mean: {stats['episode_length']['mean']:.2f}")
        print(f"  Std Dev: {stats['episode_length']['std']:.2f}")
        print(f"  Min: {stats['episode_length']['min']:.2f}")
        print(f"  Max: {stats['episode_length']['max']:.2f}")
        print(f"  Latest: {stats['episode_length']['latest']:.2f}")
        print(f"  Last 10 Avg: {stats['episode_length']['last_10_avg']:.2f}")
        
        print("\nREWARD COMPONENTS (% of total):")
        print(f"  Survival: {stats['reward_components']['survival']['mean']:.2f} ({stats['reward_components']['survival']['percentage']:.1f}%)")
        print(f"  Score: {stats['reward_components']['score']['mean']:.2f} ({stats['reward_components']['score']['percentage']:.1f}%)")
        print(f"  Coins: {stats['reward_components']['coin']['mean']:.2f} ({stats['reward_components']['coin']['percentage']:.1f}%)")
        print(f"  Penalties: {stats['reward_components']['penalty']['mean']:.2f} ({stats['reward_components']['penalty']['percentage']:.1f}%)")
        
        print("\n" + "="*50 + "\n")