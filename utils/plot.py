import matplotlib.pyplot as plt
import numpy as np

class PerformancePlotter:
    def __init__(self, save_dir='../logs'):
        self.save_dir = save_dir
    
    def plot_rewards(self, rewards, window_size=100):
        """Plot episode rewards with moving average."""
        plt.figure(figsize=(10, 5))
        plt.plot(rewards, alpha=0.3, label='Raw Rewards')
        
        # Calculate and plot moving average
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(moving_avg, label=f'Moving Average (window={window_size})')
        
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Training Rewards over Time')
        plt.legend()
        plt.savefig(f'{self.save_dir}/rewards.png')
        plt.close()
    
    def plot_steps(self, steps, window_size=100):
        """Plot episode lengths with moving average."""
        plt.figure(figsize=(10, 5))
        plt.plot(steps, alpha=0.3, label='Raw Steps')
        
        # Calculate and plot moving average
        moving_avg = np.convolve(steps, np.ones(window_size)/window_size, mode='valid')
        plt.plot(moving_avg, label=f'Moving Average (window={window_size})')
        
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.title('Episode Lengths over Time')
        plt.legend()
        plt.savefig(f'{self.save_dir}/steps.png')
        plt.close()
    
    def plot_loss(self, losses, window_size=100):
        """Plot training loss with moving average."""
        plt.figure(figsize=(10, 5))
        plt.plot(losses, alpha=0.3, label='Raw Loss')
        
        # Calculate and plot moving average
        moving_avg = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
        plt.plot(moving_avg, label=f'Moving Average (window={window_size})')
        
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        plt.title('Training Loss over Time')
        plt.legend()
        plt.savefig(f'{self.save_dir}/loss.png')
        plt.close()
    
    def plot_epsilon(self, epsilons):
        """Plot epsilon decay over training."""
        plt.figure(figsize=(10, 5))
        plt.plot(epsilons)
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        plt.title('Epsilon Decay over Training')
        plt.savefig(f'{self.save_dir}/epsilon.png')
        plt.close()