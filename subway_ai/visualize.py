# Visualization utility for Subway Surfers AI training results
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from subway_ai.config import *

def load_training_log(log_path=None):
    """Load training log data from CSV file"""
    if log_path is None:
        # Find the most recent log file
        results_dir = os.path.join(BASE_DIR, "results")
        if not os.path.exists(results_dir):
            print(f"Results directory not found: {results_dir}")
            return None
        
        log_files = [f for f in os.listdir(results_dir) if f.startswith("training_log_") and f.endswith(".csv")]
        if not log_files:
            print("No training log files found")
            return None
        
        # Sort by date (newest first)
        log_files.sort(reverse=True)
        log_path = os.path.join(results_dir, log_files[0])
    
    try:
        # Load the CSV data
        data = pd.read_csv(log_path)
        print(f"Loaded training log: {log_path}")
        print(f"Contains {len(data)} episodes")
        return data
    except Exception as e:
        print(f"Error loading training log {log_path}: {e}")
        return None

def plot_training_metrics(data, save_dir=None, show=True):
    """Plot various training metrics"""
    if data is None or len(data) == 0:
        print("No data to plot")
        return
    
    # Create save directory if needed
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Set up the figure
    plt.style.use('ggplot')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # --------------------------------------------------
    # Plot 1: Score over episodes
    plt.figure(figsize=(12, 6))
    plt.plot(data['Episode'], data['Score'], '-o', alpha=0.7, markersize=4)
    
    # Add smoothed trend line
    window_size = min(10, len(data) // 2) if len(data) > 10 else 3
    if window_size > 0:
        smooth_scores = data['Score'].rolling(window=window_size).mean()
        plt.plot(data['Episode'], smooth_scores, 'r-', linewidth=2, label=f'Moving Avg (window={window_size})')
    
    plt.title('Score Progression During Training', fontsize=16)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"scores_{timestamp}.png"), dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    # --------------------------------------------------
    # Plot 2: Rewards over episodes
    plt.figure(figsize=(12, 6))
    plt.plot(data['Episode'], data['TotalReward'], '-o', alpha=0.7, markersize=4)
    
    # Add smoothed trend line
    if window_size > 0:
        smooth_rewards = data['TotalReward'].rolling(window=window_size).mean()
        plt.plot(data['Episode'], smooth_rewards, 'r-', linewidth=2, label=f'Moving Avg (window={window_size})')
    
    plt.title('Total Reward Progression During Training', fontsize=16)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Total Reward', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"rewards_{timestamp}.png"), dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    # --------------------------------------------------
    # Plot 3: Epsilon decay
    plt.figure(figsize=(12, 6))
    plt.plot(data['Episode'], data['Epsilon'], '-', alpha=0.8)
    plt.title('Epsilon Decay During Training', fontsize=16)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Epsilon', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"epsilon_{timestamp}.png"), dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    # --------------------------------------------------
    # Plot 4: Steps per episode
    plt.figure(figsize=(12, 6))
    plt.plot(data['Episode'], data['Steps'], '-o', alpha=0.7, markersize=4)
    
    # Add smoothed trend line
    if window_size > 0:
        smooth_steps = data['Steps'].rolling(window=window_size).mean()
        plt.plot(data['Episode'], smooth_steps, 'r-', linewidth=2, label=f'Moving Avg (window={window_size})')
    
    plt.title('Steps per Episode During Training', fontsize=16)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Steps', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"steps_{timestamp}.png"), dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    # --------------------------------------------------
    # Plot 5: Combined metrics (normalized)
    plt.figure(figsize=(12, 6))
    
    # Normalize data for comparison
    norm_score = data['Score'] / data['Score'].max() if data['Score'].max() > 0 else data['Score']
    norm_steps = data['Steps'] / data['Steps'].max() if data['Steps'].max() > 0 else data['Steps']
    norm_reward = (data['TotalReward'] - data['TotalReward'].min()) / (data['TotalReward'].max() - data['TotalReward'].min()) if data['TotalReward'].max() > data['TotalReward'].min() else data['TotalReward'] / 100
    
    plt.plot(data['Episode'], norm_score, 'b-', alpha=0.7, label='Normalized Score')
    plt.plot(data['Episode'], norm_steps, 'g-', alpha=0.7, label='Normalized Steps')
    plt.plot(data['Episode'], norm_reward, 'r-', alpha=0.7, label='Normalized Reward')
    plt.plot(data['Episode'], data['Epsilon'], 'k--', alpha=0.7, label='Epsilon')
    
    plt.title('Combined Metrics During Training (Normalized)', fontsize=16)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Normalized Value', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"combined_{timestamp}.png"), dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    # --------------------------------------------------
    # Plot 6: Coins collected
    if 'Coins' in data.columns:
        plt.figure(figsize=(12, 6))
        plt.plot(data['Episode'], data['Coins'], '-o', alpha=0.7, markersize=4)
        
        # Add smoothed trend line
        if window_size > 0:
            smooth_coins = data['Coins'].rolling(window=window_size).mean()
            plt.plot(data['Episode'], smooth_coins, 'r-', linewidth=2, label=f'Moving Avg (window={window_size})')
        
        plt.title('Coins Collected During Training', fontsize=16)
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Coins', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, f"coins_{timestamp}.png"), dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
    
    print(f"Plots were {'saved to ' + save_dir if save_dir else 'displayed'}")

def analyze_training_data(data):
    """Analyze training data and print statistics"""
    if data is None or len(data) == 0:
        print("No data to analyze")
        return
    
    print("\n" + "="*50)
    print("TRAINING STATISTICS")
    print("="*50)
    
    # Basic statistics
    print(f"Total Episodes: {len(data)}")
    print(f"Training Duration: {data['DurationSecs'].sum() / 60:.2f} minutes")
    
    # Score statistics
    print("\nSCORE STATISTICS:")
    print(f"Max Score: {data['Score'].max()}")
    print(f"Average Score: {data['Score'].mean():.2f}")
    print(f"Episode with Best Score: {data.loc[data['Score'].idxmax(), 'Episode']}")
    
    # Steps statistics
    print("\nSTEPS STATISTICS:")
    print(f"Max Steps: {data['Steps'].max()}")
    print(f"Average Steps: {data['Steps'].mean():.2f}")
    print(f"Episode with Most Steps: {data.loc[data['Steps'].idxmax(), 'Episode']}")
    
    # Rewards statistics
    print("\nREWARD STATISTICS:")
    print(f"Max Reward: {data['TotalReward'].max():.2f}")
    print(f"Average Reward: {data['TotalReward'].mean():.2f}")
    print(f"Episode with Best Reward: {data.loc[data['TotalReward'].idxmax(), 'Episode']}")
    
    # Coins statistics
    if 'Coins' in data.columns:
        print("\nCOINS STATISTICS:")
        print(f"Max Coins: {data['Coins'].max()}")
        print(f"Average Coins: {data['Coins'].mean():.2f}")
        print(f"Episode with Most Coins: {data.loc[data['Coins'].idxmax(), 'Episode']}")
    
    # Training progress
    print("\nTRAINING PROGRESS:")
    # Split data into quarters to see progression
    quarter_size = len(data) // 4
    if quarter_size > 0:
        for i in range(4):
            start_idx = i * quarter_size
            end_idx = (i + 1) * quarter_size if i < 3 else len(data)
            quarter_data = data.iloc[start_idx:end_idx]
            print(f"Quarter {i+1}: Avg Score = {quarter_data['Score'].mean():.2f}, Avg Steps = {quarter_data['Steps'].mean():.2f}")
    
    print("="*50)

def main():
    """Main function to visualize training results"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize Subway Surfers AI training results')
    parser.add_argument('--log', type=str, help='Path to training log CSV file (default: most recent)')
    parser.add_argument('--save', type=str, help='Directory to save plots (default: results/plots)')
    parser.add_argument('--noshow', action='store_true', help='Do not display plots')
    args = parser.parse_args()
    
    # Set up save directory
    save_dir = args.save if args.save else os.path.join(BASE_DIR, "results", "plots")
    
    # Load data
    data = load_training_log(args.log)
    
    if data is not None:
        # Analyze data
        analyze_training_data(data)
        
        # Plot data
        plot_training_metrics(data, save_dir=save_dir, show=not args.noshow)
    else:
        print("No data available to visualize")

if __name__ == "__main__":
    main()