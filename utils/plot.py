# utils/plot.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Plotting")

def set_plot_style():
    """Set the style for plots"""
    sns.set(style="darkgrid")
    sns.set_context("talk")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['figure.dpi'] = 100

def plot_rewards(rewards, window_size=10, title="Episode Rewards", save_path=None):
    """
    Plot episode rewards with a moving average
    
    Args:
        rewards: List of rewards per episode
        window_size: Window size for moving average
        title: Plot title
        save_path: Path to save the plot (optional)
    """
    set_plot_style()
    
    # Create figure
    plt.figure()
    
    # Plot raw rewards
    plt.plot(rewards, alpha=0.6, label="Raw Rewards")
    
    # Plot moving average if there are enough episodes
    if len(rewards) >= window_size:
        moving_avg = pd.Series(rewards).rolling(window_size).mean()
        plt.plot(moving_avg, color='red', label=f"{window_size}-Episode Moving Average")
    
    # Customize plot
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Rewards plot saved to {save_path}")
    
    plt.tight_layout()
    plt.show()

def plot_training_metrics(rewards, lengths, epsilons, losses=None, save_dir=None):
    """
    Plot multiple training metrics
    
    Args:
        rewards: List of rewards per episode
        lengths: List of episode lengths
        epsilons: List of epsilon values
        losses: List of training losses (optional)
        save_dir: Directory to save plots (optional)
    """
    set_plot_style()
    
    # Determine number of subplots
    num_plots = 3 if losses is None else 4
    
    # Create figure with subplots
    fig, axs = plt.subplots(num_plots, 1, figsize=(12, 4 * num_plots), sharex=True)
    
    # Plot rewards
    axs[0].plot(rewards, alpha=0.6, color='blue')
    if len(rewards) >= 10:
        moving_avg = pd.Series(rewards).rolling(10).mean()
        axs[0].plot(moving_avg, color='red', label="10-Episode Moving Average")
    axs[0].set_title("Episode Rewards")
    axs[0].set_ylabel("Reward")
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)
    
    # Plot episode lengths
    axs[1].plot(lengths, alpha=0.6, color='green')
    if len(lengths) >= 10:
        moving_avg = pd.Series(lengths).rolling(10).mean()
        axs[1].plot(moving_avg, color='red', label="10-Episode Moving Average")
    axs[1].set_title("Episode Lengths")
    axs[1].set_ylabel("Steps")
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)
    
    # Plot epsilon decay
    axs[2].plot(epsilons, color='purple')
    axs[2].set_title("Exploration Rate (Epsilon)")
    axs[2].set_ylabel("Epsilon")
    axs[2].grid(True, alpha=0.3)
    
    # Plot losses if provided
    if losses is not None:
        axs[3].plot(losses, alpha=0.6, color='orange')
        if len(losses) >= 10:
            moving_avg = pd.Series(losses).rolling(10).mean()
            axs[3].plot(moving_avg, color='red', label="10-Episode Moving Average")
        axs[3].set_title("Training Loss")
        axs[3].set_ylabel("Loss")
        axs[3].set_yscale('log')  # Log scale for losses
        axs[3].legend()
        axs[3].grid(True, alpha=0.3)
    
    # Set common x-axis label
    axs[-1].set_xlabel("Episode")
    
    # Save plots if directory provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_dir, f"training_metrics_{timestamp}.png")
        plt.savefig(save_path)
        logger.info(f"Training metrics plot saved to {save_path}")
    
    plt.tight_layout()
    plt.show()

def plot_state_value_heatmap(q_values, title="State Value Heatmap", save_path=None):
    """
    Plot a heatmap of state values from Q-values
    
    Args:
        q_values: Q-values array of shape (height, width, n_actions)
        title: Plot title
        save_path: Path to save the plot (optional)
    """
    set_plot_style()
    
    # Calculate state values (max Q-value for each state)
    state_values = np.max(q_values, axis=2)
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(state_values, cmap='viridis', annot=False)
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    
    # Add colorbar
    cbar = plt.colorbar()
    cbar.set_label("Value")
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path)
        logger.info(f"State value heatmap saved to {save_path}")
    
    plt.tight_layout()
    plt.show()

def plot_action_distribution(actions, n_actions=4, action_names=None, title="Action Distribution", save_path=None):
    """
    Plot distribution of actions taken by the agent
    
    Args:
        actions: List of actions taken
        n_actions: Number of possible actions
        action_names: Names of actions (optional)
        title: Plot title
        save_path: Path to save the plot (optional)
    """
    set_plot_style()
    
    # Default action names if not provided
    if action_names is None:
        action_names = [f"Action {i}" for i in range(n_actions)]
    
    # Count actions
    action_counts = np.bincount(actions, minlength=n_actions)
    
    # Calculate percentages
    action_percentages = action_counts / len(actions) * 100
    
    # Create bar plot
    plt.figure()
    bars = plt.bar(range(n_actions), action_counts, alpha=0.7)
    
    # Add percentage labels
    for i, (bar, percentage) in enumerate(zip(bars, action_percentages)):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{percentage:.1f}%",
            ha='center',
            va='bottom'
        )
    
    # Customize plot
    plt.title(title)
    plt.xlabel("Action")
    plt.ylabel("Count")
    plt.xticks(range(n_actions), action_names)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Action distribution plot saved to {save_path}")
    
    plt.tight_layout()
    plt.show()

def plot_reward_components(coin_rewards, survival_rewards, distance_rewards, title="Reward Components", save_path=None):
    """
    Plot breakdown of reward components
    
    Args:
        coin_rewards: Rewards from collecting coins
        survival_rewards: Rewards from surviving
        distance_rewards: Rewards from traveling distance
        title: Plot title
        save_path: Path to save the plot (optional)
    """
    set_plot_style()
    
    # Create stacked area plot
    plt.figure()
    
    # Create x-axis (episodes)
    episodes = range(len(coin_rewards))
    
    # Plot stacked areas
    plt.stackplot(
        episodes, 
        [coin_rewards, survival_rewards, distance_rewards],
        labels=['Coins', 'Survival', 'Distance'],
        alpha=0.7
    )
    
    # Customize plot
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Reward components plot saved to {save_path}")
    
    plt.tight_layout()
    plt.show()

def plot_learning_curve(csv_path, save_dir=None):
    """
    Plot learning curves from a CSV log file
    
    Args:
        csv_path: Path to CSV log file
        save_dir: Directory to save plots (optional)
    """
    try:
        # Load CSV data
        data = pd.read_csv(csv_path)
        
        set_plot_style()
        
        # Create subplots
        fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
        
        # Plot reward
        axs[0].plot(data['Episode'], data['Reward'], alpha=0.6, label='Reward')
        axs[0].plot(data['Episode'], data['Reward'].rolling(10).mean(), 'r-', label='10-Episode Average')
        axs[0].set_title('Training Reward')
        axs[0].set_ylabel('Reward')
        axs[0].legend()
        axs[0].grid(True, alpha=0.3)
        
        # Plot episode length
        axs[1].plot(data['Episode'], data['Length'], alpha=0.6, label='Length')
        axs[1].plot(data['Episode'], data['Length'].rolling(10).mean(), 'r-', label='10-Episode Average')
        axs[1].set_title('Episode Length')
        axs[1].set_ylabel('Steps')
        axs[1].legend()
        axs[1].grid(True, alpha=0.3)
        
        # Plot epsilon
        axs[2].plot(data['Episode'], data['Epsilon'], label='Epsilon')
        axs[2].set_title('Exploration Rate (Epsilon)')
        axs[2].set_ylabel('Epsilon')
        axs[2].set_xlabel('Episode')
        axs[2].legend()
        axs[2].grid(True, alpha=0.3)
        
        # Save plot if directory provided
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            filename = os.path.basename(csv_path).replace('.csv', '.png')
            save_path = os.path.join(save_dir, filename)
            plt.savefig(save_path)
            logger.info(f"Learning curve saved to {save_path}")
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        logger.error(f"Error plotting learning curve: {str(e)}")

def plot_frame_processing(original_frame, processed_frame, save_path=None):
    """
    Plot original and processed frames side by side
    
    Args:
        original_frame: Original RGB frame
        processed_frame: Processed frame (e.g., grayscale, resized)
        save_path: Path to save the plot (optional)
    """
    set_plot_style()
    
    # Create figure with subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot original frame
    axs[0].imshow(original_frame)
    axs[0].set_title("Original Frame")
    axs[0].axis('off')
    
    # Plot processed frame
    if len(processed_frame.shape) == 2:  # Grayscale
        axs[1].imshow(processed_frame, cmap='gray')
    else:  # RGB
        axs[1].imshow(processed_frame)
    axs[1].set_title("Processed Frame")
    axs[1].axis('off')
    
    # Add some processing details as text
    plt.figtext(
        0.5, 0.01,
        f"Original: {original_frame.shape}, Processed: {processed_frame.shape}",
        ha='center'
    )
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Frame processing plot saved to {save_path}")
    
    plt.tight_layout()
    plt.show()