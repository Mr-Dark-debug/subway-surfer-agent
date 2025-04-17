# utils/plot_utils.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
from datetime import datetime
import logging
import csv
from matplotlib.ticker import MaxNLocator

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
    plt.rcParams['savefig.dpi'] = 150
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']

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
    
    # Calculate exponential weighted moving average
    if len(rewards) >= 3:  # Need at least a few points
        ewma = pd.Series(rewards).ewm(span=window_size).mean()
        plt.plot(ewma, color='green', linestyle='--', label=f"EWMA (span={window_size})")
    
    # Customize plot
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add trend line
    if len(rewards) >= 5:  # Need enough points for meaningful trend
        x = np.arange(len(rewards))
        z = np.polyfit(x, rewards, 1)
        p = np.poly1d(z)
        plt.plot(x, p(x), "b--", alpha=0.5, label=f"Trend: {z[0]:.4f}x + {z[1]:.2f}")
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
        
        # Add exponential moving average
        ewma = pd.Series(rewards).ewm(span=20).mean()
        axs[0].plot(ewma, color='green', linestyle='--', label="EWMA (span=20)")
    axs[0].set_title("Episode Rewards")
    axs[0].set_ylabel("Reward")
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)
    
    # Plot trend line for rewards
    if len(rewards) >= 5:
        x = np.arange(len(rewards))
        z = np.polyfit(x, rewards, 1)
        p = np.poly1d(z)
        axs[0].plot(x, p(x), "b--", alpha=0.5, label=f"Trend: {z[0]:.4f}x + {z[1]:.2f}")
        axs[0].legend()
    
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
        axs[3].set_yscale('log' if max(losses) / (min(losses) + 1e-10) > 100 else 'linear')  # Use log scale if range is large
        axs[3].legend()
        axs[3].grid(True, alpha=0.3)
    
    # Set common x-axis label
    axs[-1].set_xlabel("Episode")
    
    # Ensure integer x-axis ticks
    for ax in axs:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Add timestamp and save information
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fig.text(0.5, 0.01, f"Generated: {timestamp}", ha='center', fontsize=9, style='italic')
    
    # Adjust layout for better spacing
    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    
    # Save plots if directory provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_dir, f"training_metrics_{timestamp}.png")
        plt.savefig(save_path)
        logger.info(f"Training metrics plot saved to {save_path}")
    
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

def plot_reward_components(coin_rewards, survival_rewards, score_rewards, penalty_rewards=None, title="Reward Components", save_path=None):
    """
    Plot breakdown of reward components
    
    Args:
        coin_rewards: Rewards from collecting coins
        survival_rewards: Rewards from surviving
        score_rewards: Rewards from increasing score
        penalty_rewards: Penalty rewards (negative)
        title: Plot title
        save_path: Path to save the plot (optional)
    """
    set_plot_style()
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Create x-axis (episodes)
    episodes = range(len(coin_rewards))
    
    # Create array of components
    components = [survival_rewards, score_rewards, coin_rewards]
    labels = ['Survival', 'Score', 'Coins']
    
    # Add penalties if provided
    if penalty_rewards is not None:
        # Convert penalties to positive for stacking
        positive_penalties = [abs(p) for p in penalty_rewards]
        components.append(positive_penalties)
        labels.append('Penalties')
    
    # Create stacked area plot for positive components
    plt.stackplot(
        episodes, 
        components,
        labels=labels,
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
        
        # Check for required columns
        required_columns = ['Episode', 'Reward', 'Length', 'Epsilon']
        if not all(col in data.columns for col in required_columns):
            logger.error(f"CSV file missing required columns. Expected: {required_columns}, Got: {list(data.columns)}")
            return
        
        # Create subplots
        fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
        
        # Plot reward
        axs[0].plot(data['Episode'], data['Reward'], alpha=0.6, label='Reward')
        axs[0].plot(data['Episode'], data['Reward'].rolling(10).mean(), 'r-', label='10-Episode Average')
        
        # Add exponential moving average
        ewma = data['Reward'].ewm(span=20).mean()
        axs[0].plot(data['Episode'], ewma, 'g--', label='EWMA (span=20)')
        
        # Add trend line
        x = data['Episode'].values
        z = np.polyfit(x, data['Reward'].values, 1)
        p = np.poly1d(z)
        axs[0].plot(x, p(x), "b--", alpha=0.5, label=f"Trend: {z[0]:.4f}x + {z[1]:.2f}")
        
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
        
        # Add loss plot if available
        if 'Loss' in data.columns:
            fig.set_figheight(20)
            ax_loss = fig.add_subplot(4, 1, 4, sharex=axs[0])
            ax_loss.plot(data['Episode'], data['Loss'], alpha=0.6, label='Loss')
            if len(data) >= 10:
                ax_loss.plot(data['Episode'], data['Loss'].rolling(10).mean(), 'r-', label='10-Episode Average')
            ax_loss.set_title('Training Loss')
            ax_loss.set_ylabel('Loss')
            ax_loss.set_xlabel('Episode')
            ax_loss.legend()
            ax_loss.grid(True, alpha=0.3)
            
            # Use log scale for loss if range is large
            if data['Loss'].max() / (data['Loss'].min() + 1e-10) > 100:
                ax_loss.set_yscale('log')
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fig.text(0.5, 0.01, f"Generated: {timestamp}", ha='center', fontsize=9, style='italic')
        
        # Save plot if directory provided
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            filename = os.path.basename(csv_path).replace('.csv', '.png')
            save_path = os.path.join(save_dir, filename)
            plt.savefig(save_path)
            logger.info(f"Learning curve saved to {save_path}")
        
        plt.tight_layout(rect=[0, 0.02, 1, 0.98])
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

def plot_comparison(original, current, metric_name, title="Performance Comparison", save_path=None):
    """
    Plot a comparison between original and current performance
    
    Args:
        original: List of original metrics
        current: List of current metrics
        metric_name: Name of the metric
        title: Plot title
        save_path: Path to save the plot (optional)
    """
    set_plot_style()
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Create x-axis for both datasets
    x_original = np.arange(len(original))
    x_current = np.arange(len(current))
    
    # Plot original and current metrics
    plt.plot(x_original, original, label="Original", alpha=0.7)
    plt.plot(x_current, current, label="Current", alpha=0.7)
    
    # Calculate and plot moving averages
    if len(original) >= 10:
        ma_original = pd.Series(original).rolling(10).mean()
        plt.plot(x_original, ma_original, '--', label="Original (10-MA)")
    
    if len(current) >= 10:
        ma_current = pd.Series(current).rolling(10).mean()
        plt.plot(x_current, ma_current, '--', label="Current (10-MA)")
    
    # Calculate improvement
    if len(original) > 0 and len(current) > 0:
        original_avg = np.mean(original)
        current_avg = np.mean(current)
        improvement = (current_avg - original_avg) / original_avg * 100
        
        plt.figtext(
            0.5, 0.01,
            f"Improvement: {improvement:.2f}% ({original_avg:.2f} → {current_avg:.2f})",
            ha='center',
            fontsize=10,
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5')
        )
    
    # Customize plot
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel(metric_name)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Comparison plot saved to {save_path}")
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()

def create_training_summary_report(csv_path, save_dir=None):
    """
    Create a comprehensive training summary report from a CSV log file
    
    Args:
        csv_path: Path to CSV log file
        save_dir: Directory to save the report (optional)
    """
    try:
        # Load CSV data
        data = pd.read_csv(csv_path)
        
        # Create report directory
        if save_dir:
            report_dir = os.path.join(save_dir, "reports")
            os.makedirs(report_dir, exist_ok=True)
            
            # Report filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = os.path.join(report_dir, f"training_report_{timestamp}.txt")
            
            # Create HTML report file
            html_report = os.path.join(report_dir, f"training_report_{timestamp}.html")
        else:
            report_file = None
            html_report = None
        
        # Calculate statistics
        stats = {
            'episodes': len(data),
            'total_steps': data['Length'].sum(),
            'avg_reward': data['Reward'].mean(),
            'std_reward': data['Reward'].std(),
            'min_reward': data['Reward'].min(),
            'max_reward': data['Reward'].max(),
            'avg_length': data['Length'].mean(),
            'std_length': data['Length'].std(),
            'min_length': data['Length'].min(),
            'max_length': data['Length'].max(),
            'last_10_reward': data['Reward'].tail(10).mean(),
            'improvement': (data['Reward'].tail(10).mean() - data['Reward'].head(10).mean()) / max(1, abs(data['Reward'].head(10).mean())) * 100
        }
        
        # Generate report text
        report = f"Training Summary Report\n"
        report += f"=====================\n\n"
        report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"CSV File: {os.path.basename(csv_path)}\n\n"
        
        report += f"Basic Statistics\n"
        report += f"-----------------\n"
        report += f"Total Episodes: {stats['episodes']}\n"
        report += f"Total Steps: {stats['total_steps']}\n\n"
        
        report += f"Reward Statistics\n"
        report += f"-----------------\n"
        report += f"Average Reward: {stats['avg_reward']:.2f} ± {stats['std_reward']:.2f}\n"
        report += f"Min Reward: {stats['min_reward']:.2f}\n"
        report += f"Max Reward: {stats['max_reward']:.2f}\n"
        report += f"Last 10 Episodes Avg: {stats['last_10_reward']:.2f}\n"
        report += f"Improvement (First 10 vs Last 10): {stats['improvement']:.2f}%\n\n"
        
        report += f"Episode Length Statistics\n"
        report += f"------------------------\n"
        report += f"Average Length: {stats['avg_length']:.2f} ± {stats['std_length']:.2f}\n"
        report += f"Min Length: {stats['min_length']:.2f}\n"
        report += f"Max Length: {stats['max_length']:.2f}\n\n"
        
        # Add loss statistics if available
        if 'Loss' in data.columns:
            report += f"Loss Statistics\n"
            report += f"--------------\n"
            report += f"Average Loss: {data['Loss'].mean():.6f} ± {data['Loss'].std():.6f}\n"
            report += f"Min Loss: {data['Loss'].min():.6f}\n"
            report += f"Max Loss: {data['Loss'].max():.6f}\n"
            report += f"Last 10 Episodes Avg: {data['Loss'].tail(10).mean():.6f}\n\n"
        
        # Add time statistics if available
        if 'Episode Time' in data.columns:
            report += f"Time Statistics\n"
            report += f"--------------\n"
            report += f"Average Episode Time: {data['Episode Time'].mean():.2f}s ± {data['Episode Time'].std():.2f}s\n"
            report += f"Total Training Time: {data['Episode Time'].sum():.2f}s ({data['Episode Time'].sum()/3600:.2f}h)\n\n"
        
        # Print report to console
        print(report)
        
        # Save report to file if requested
        if report_file:
            with open(report_file, 'w') as f:
                f.write(report)
            logger.info(f"Training report saved to {report_file}")
        
        # Generate and save HTML report if requested
        if html_report:
            # Create HTML report with embedded plots
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Training Summary Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2 {{ color: #2c3e50; }}
                    .stats {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; }}
                    .plot {{ margin: 20px 0; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ text-align: left; padding: 8px; }}
                    tr:nth-child(even) {{ background-color: #f2f2f2; }}
                    th {{ background-color: #2c3e50; color: white; }}
                </style>
            </head>
            <body>
                <h1>Training Summary Report</h1>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>CSV File: {os.path.basename(csv_path)}</p>
                
                <h2>Basic Statistics</h2>
                <div class="stats">
                    <table>
                        <tr><th>Metric</th><th>Value</th></tr>
                        <tr><td>Total Episodes</td><td>{stats['episodes']}</td></tr>
                        <tr><td>Total Steps</td><td>{stats['total_steps']}</td></tr>
                    </table>
                </div>
                
                <h2>Reward Statistics</h2>
                <div class="stats">
                    <table>
                        <tr><th>Metric</th><th>Value</th></tr>
                        <tr><td>Average Reward</td><td>{stats['avg_reward']:.2f} ± {stats['std_reward']:.2f}</td></tr>
                        <tr><td>Min Reward</td><td>{stats['min_reward']:.2f}</td></tr>
                        <tr><td>Max Reward</td><td>{stats['max_reward']:.2f}</td></tr>
                        <tr><td>Last 10 Episodes Avg</td><td>{stats['last_10_reward']:.2f}</td></tr>
                        <tr><td>Improvement (First 10 vs Last 10)</td><td>{stats['improvement']:.2f}%</td></tr>
                    </table>
                </div>
                
                <h2>Episode Length Statistics</h2>
                <div class="stats">
                    <table>
                        <tr><th>Metric</th><th>Value</th></tr>
                        <tr><td>Average Length</td><td>{stats['avg_length']:.2f} ± {stats['std_length']:.2f}</td></tr>
                        <tr><td>Min Length</td><td>{stats['min_length']:.2f}</td></tr>
                        <tr><td>Max Length</td><td>{stats['max_length']:.2f}</td></tr>
                    </table>
                </div>
            """
            
            # Add loss statistics if available
            if 'Loss' in data.columns:
                html_content += f"""
                <h2>Loss Statistics</h2>
                <div class="stats">
                    <table>
                        <tr><th>Metric</th><th>Value</th></tr>
                        <tr><td>Average Loss</td><td>{data['Loss'].mean():.6f} ± {data['Loss'].std():.6f}</td></tr>
                        <tr><td>Min Loss</td><td>{data['Loss'].min():.6f}</td></tr>
                        <tr><td>Max Loss</td><td>{data['Loss'].max():.6f}</td></tr>
                        <tr><td>Last 10 Episodes Avg</td><td>{data['Loss'].tail(10).mean():.6f}</td></tr>
                    </table>
                </div>
                """
            
            # Add time statistics if available
            if 'Episode Time' in data.columns:
                html_content += f"""
                <h2>Time Statistics</h2>
                <div class="stats">
                    <table>
                        <tr><th>Metric</th><th>Value</th></tr>
                        <tr><td>Average Episode Time</td><td>{data['Episode Time'].mean():.2f}s ± {data['Episode Time'].std():.2f}s</td></tr>
                        <tr><td>Total Training Time</td><td>{data['Episode Time'].sum():.2f}s ({data['Episode Time'].sum()/3600:.2f}h)</td></tr>
                    </table>
                </div>
                """
            
            # Close HTML
            html_content += """
            </body>
            </html>
            """
            
            # Save HTML report
            with open(html_report, 'w') as f:
                f.write(html_content)
            logger.info(f"HTML training report saved to {html_report}")
        
        # Return statistics dictionary
        return stats
        
    except Exception as e:
        logger.error(f"Error creating training summary report: {str(e)}")
        return None