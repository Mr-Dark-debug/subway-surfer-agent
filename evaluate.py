#!/usr/bin/env python
"""
Subway Surfers AI - Evaluation Script

This script systematically evaluates a trained agent's performance,
collecting detailed metrics and generating comprehensive reports.
"""

import os
import argparse
import logging
import torch
import numpy as np
import json
import time
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tqdm import tqdm

# Configure logging
os.makedirs("logs", exist_ok=True)
log_file = f"subway_surfers_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/{log_file}"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Evaluation")

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained Subway Surfers AI agent")
    
    # Basic settings
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint file to load")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of episodes to evaluate")
    parser.add_argument("--browser_position", type=str, default="right", choices=["left", "right"],
                        help="Position of the browser window")
    parser.add_argument("--record_best", action="store_true",
                        help="Record video of the best episode")
    
    # Evaluation options
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                        help="Directory to save evaluation results")
    parser.add_argument("--compare", type=str, default=None,
                        help="Path to previous evaluation results for comparison")
    
    # Debug options
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode")
    
    return parser.parse_args()

def create_frame_stack_env(args):
    """
    Create and initialize the wrapped game environment
    
    Args:
        args: Command line arguments
        
    Returns:
        Frame stacking environment wrapper
    """
    from env.game_interaction import SubwaySurfersEnv
    
    # Create environment
    env = SubwaySurfersEnv(
        position=args.browser_position,
        use_existing_browser=False
    )
    
    # Wrap with frame stacking
    class FrameStackingEnv:
        def __init__(self, env, stack_size=4):
            self.env = env
            self.stack_size = stack_size
            self.frames = []
            self.actions = env.actions
            
        def reset(self):
            observation = self.env.reset()
            self.frames = []
            for _ in range(self.stack_size):
                self.frames.append(observation)
            return np.array(self.frames)
        
        def step(self, action):
            next_observation, reward, done, info = self.env.step(action)
            self.frames.pop(0)
            self.frames.append(next_observation)
            return np.array(self.frames), reward, done, info
        
        def close(self):
            self.env.close()
    
    return FrameStackingEnv(env, stack_size=4)

def load_agent(checkpoint_path, device):
    """
    Load agent from checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Torch device to use
        
    Returns:
        Loaded agent
    """
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint data
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model architecture information
    state_shape = checkpoint.get('state_shape', (4, 84, 84))
    n_actions = checkpoint.get('n_actions', 5)
    use_dueling = checkpoint.get('use_dueling', True)
    use_double = checkpoint.get('use_double', True)
    use_per = checkpoint.get('use_per', True)
    memory_efficient = checkpoint.get('memory_efficient', False)
    
    # Import appropriate agent type
    from model.agent import DQNAgent, AdaptiveDQNAgent
    
    # Create agent
    if 'adaptation_history' in checkpoint:
        agent = AdaptiveDQNAgent(
            state_shape=state_shape,
            n_actions=n_actions,
            use_dueling=use_dueling,
            use_double=use_double,
            use_per=use_per,
            device=device,
            memory_efficient=memory_efficient
        )
    else:
        agent = DQNAgent(
            state_shape=state_shape,
            n_actions=n_actions,
            use_dueling=use_dueling,
            use_double=use_double,
            use_per=use_per,
            device=device,
            memory_efficient=memory_efficient
        )
    
    # Load weights and parameters from checkpoint
    agent.load_checkpoint(checkpoint_path)
    
    return agent, checkpoint

def run_evaluation_episode(agent, env, episode_num, best_reward=None, record_best=False):
    """
    Run a single evaluation episode
    
    Args:
        agent: Trained agent
        env: Environment
        episode_num: Episode number
        best_reward: Current best reward (for recording best episode)
        record_best: Whether to record the best episode
        
    Returns:
        Dictionary of episode metrics
    """
    # Reset environment
    state = env.reset()
    
    # Initialize video writer if recording and this could be the best episode
    video_writer = None
    if record_best and best_reward is not None:
        os.makedirs("videos", exist_ok=True)
        video_path = f"videos/temp_episode_{episode_num}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_path, fourcc, 10.0, (1200, 600))
    
    # Initialize metrics
    episode_metrics = {
        'episode': episode_num,
        'total_reward': 0,
        'length': 0,
        'score': 0,
        'coins': 0,
        'survival_reward': 0,
        'score_reward': 0,
        'coin_reward': 0,
        'penalty_reward': 0,
        'time_reward': 0,
        'actions': {
            'noop': 0,
            'up': 0,
            'down': 0,
            'left': 0,
            'right': 0
        },
        'timestamps': []
    }
    
    # Record start time
    start_time = time.time()
    
    # Run episode
    done = False
    
    # Store frames and states for recording
    frames = []
    states = []
    q_values_history = []
    
    with tqdm(desc=f"Episode {episode_num}", unit="steps") as pbar:
        while not done:
            # Record timestamp
            episode_metrics['timestamps'].append(time.time() - start_time)
            
            # Convert state to tensor
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(agent.device)
            
            # Get Q-values
            with torch.no_grad():
                q_values = agent.policy_net(state_tensor).cpu().numpy()[0]
            
            # Select action (always greedy in evaluation mode)
            action = q_values.argmax()
            
            # Store state and Q-values for recording
            if video_writer is not None:
                states.append(state.copy())
                q_values_history.append(q_values.copy())
            
            # Take action in environment
            next_state, reward, done, info = env.step(action)
            
            # Capture frame for recording if needed
            if video_writer is not None:
                # Create a blank image
                frame = np.ones((600, 1200, 3), dtype=np.uint8) * 255
                
                # Add episode info
                cv2.putText(frame, f"Episode {episode_num} - Step {episode_metrics['length']}", 
                           (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                
                # Add reward info
                cv2.putText(frame, f"Reward: {episode_metrics['total_reward']:.2f}", 
                           (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                
                # Add score info
                cv2.putText(frame, f"Score: {info.get('score', 0)}", 
                           (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                
                # Add coin info
                cv2.putText(frame, f"Coins: {info.get('coins', 0)}", 
                           (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                
                frames.append(frame)
            
            # Update metrics
            episode_metrics['total_reward'] += reward
            episode_metrics['length'] += 1
            episode_metrics['score'] = info.get('score', episode_metrics['score'])
            episode_metrics['coins'] = info.get('coins', episode_metrics['coins'])
            
            # Update reward components
            episode_metrics['survival_reward'] += info.get('survival_reward', 0)
            episode_metrics['score_reward'] += info.get('score_reward', 0)
            episode_metrics['coin_reward'] += info.get('coin_reward', 0)
            episode_metrics['penalty_reward'] += info.get('game_over_penalty', 0)
            episode_metrics['time_reward'] += info.get('time_reward', 0)
            
            # Update action counts
            action_name = env.actions[action]
            episode_metrics['actions'][action_name] += 1
            
            # Update state
            state = next_state
            
            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({
                'reward': f"{episode_metrics['total_reward']:.2f}",
                'score': episode_metrics['score'],
                'coins': episode_metrics['coins']
            })
            
            # Safety check for max steps
            if episode_metrics['length'] >= 10000:
                logger.warning("Episode reached maximum steps (10000)")
                break
    
    # Calculate duration
    episode_metrics['duration'] = time.time() - start_time
    
    # Save recording if this is the best episode
    if video_writer is not None:
        # Check if this is the new best episode
        if episode_metrics['total_reward'] > best_reward:
            # Create a better video with states and Q-values
            for i, (frame, state, q_vals) in enumerate(zip(frames, states, q_values_history)):
                # Create state visualization
                grid_size = 2
                frame_size = 150
                grid_img = np.ones((frame_size * grid_size, frame_size * grid_size), dtype=np.uint8) * 255
                
                # Add each frame to the grid
                for j in range(min(len(state), 4)):
                    row = j // grid_size
                    col = j % grid_size
                    
                    # Resize state frame
                    frame_img = cv2.resize((state[j] * 255).astype(np.uint8), 
                                         (frame_size, frame_size))
                    
                    # Add to grid
                    grid_img[row*frame_size:(row+1)*frame_size, 
                            col*frame_size:(col+1)*frame_size] = frame_img
                
                # Convert grid to BGR
                grid_bgr = cv2.cvtColor(grid_img, cv2.COLOR_GRAY2BGR)
                
                # Add grid to frame
                frame[100:100+grid_size*frame_size, 20:20+grid_size*frame_size] = grid_bgr
                
                # Add Q-values visualization
                bar_height = 30
                bar_spacing = 10
                bar_width = 300
                
                # Normalize Q-values
                q_min = q_vals.min()
                q_max = q_vals.max()
                q_range = max(q_max - q_min, 1e-5)
                
                # Draw Q-value bars
                for j, (action_name, q_val) in enumerate(zip(env.actions, q_vals)):
                    # Bar position
                    y_pos = 100 + j * (bar_height + bar_spacing)
                    x_pos = 600
                    
                    # Normalized width
                    norm_val = (q_val - q_min) / q_range
                    width = int(norm_val * bar_width)
                    
                    # Color (green for max, blue for others)
                    color = (0, 180, 0) if j == action else (180, 0, 0)
                    
                    # Draw label
                    cv2.putText(frame, action_name, (x_pos, y_pos + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                    
                    # Draw bar
                    cv2.rectangle(frame, (x_pos + 70, y_pos), 
                                 (x_pos + 70 + width, y_pos + bar_height),
                                 color, -1)
                    
                    # Draw value
                    cv2.putText(frame, f"{q_val:.2f}", (x_pos + 75 + width, y_pos + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
                # Write frame to video
                video_writer.write(frame)
            
            # Close the temporary video
            video_writer.release()
            
            # Rename to best episode
            best_path = f"videos/best_episode_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            os.rename(f"videos/temp_episode_{episode_num}.mp4", best_path)
            logger.info(f"Saved best episode video to {best_path}")
        else:
            # Not the best, close and delete temporary video
            video_writer.release()
            if os.path.exists(f"videos/temp_episode_{episode_num}.mp4"):
                os.remove(f"videos/temp_episode_{episode_num}.mp4")
    
    return episode_metrics

def generate_evaluation_report(metrics, checkpoint_info, output_dir, compare_data=None):
    """
    Generate a comprehensive evaluation report
    
    Args:
        metrics: List of episode metrics dictionaries
        checkpoint_info: Information about the checkpoint
        output_dir: Directory to save the report
        compare_data: Previous evaluation data for comparison (optional)
        
    Returns:
        Path to the generated report
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate summary statistics
    summary = {
        'episodes': len(metrics),
        'total_reward': {
            'mean': np.mean([m['total_reward'] for m in metrics]),
            'std': np.std([m['total_reward'] for m in metrics]),
            'min': np.min([m['total_reward'] for m in metrics]),
            'max': np.max([m['total_reward'] for m in metrics])
        },
        'episode_length': {
            'mean': np.mean([m['length'] for m in metrics]),
            'std': np.std([m['length'] for m in metrics]),
            'min': np.min([m['length'] for m in metrics]),
            'max': np.max([m['length'] for m in metrics])
        },
        'score': {
            'mean': np.mean([m['score'] for m in metrics]),
            'std': np.std([m['score'] for m in metrics]),
            'min': np.min([m['score'] for m in metrics]),
            'max': np.max([m['score'] for m in metrics])
        },
        'coins': {
            'mean': np.mean([m['coins'] for m in metrics]),
            'std': np.std([m['coins'] for m in metrics]),
            'min': np.min([m['coins'] for m in metrics]),
            'max': np.max([m['coins'] for m in metrics])
        },
        'actions': {},
        'reward_components': {
            'survival': np.mean([m['survival_reward'] for m in metrics]),
            'score': np.mean([m['score_reward'] for m in metrics]),
            'coin': np.mean([m['coin_reward'] for m in metrics]),
            'penalty': np.mean([m['penalty_reward'] for m in metrics]),
            'time': np.mean([m['time_reward'] for m in metrics])
        },
        'duration': {
            'mean': np.mean([m['duration'] for m in metrics]),
            'total': sum([m['duration'] for m in metrics])
        }
    }
    
    # Calculate action distribution
    action_totals = {}
    for action in metrics[0]['actions'].keys():
        action_totals[action] = sum([m['actions'][action] for m in metrics])
    
    total_actions = sum(action_totals.values())
    summary['actions'] = {action: count / total_actions for action, count in action_totals.items()}
    
    # Convert to DataFrame for plotting
    df = pd.DataFrame(metrics)
    
    # Save detailed metrics as CSV
    df.to_csv(os.path.join(output_dir, 'detailed_metrics.csv'), index=False)
    
    # Save summary as JSON
    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        # Add checkpoint info
        output_summary = {
            'evaluation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'checkpoint': {
                'path': checkpoint_info.get('path', 'unknown'),
                'episode': checkpoint_info.get('episode', 0),
                'training_episodes': len(checkpoint_info.get('episode_rewards', [])),
                'parameters': checkpoint_info.get('hyperparameters', {})
            },
            'metrics': summary
        }
        
        json.dump(output_summary, f, indent=2)
    
    # Generate plots
    generate_evaluation_plots(df, summary, output_dir, compare_data)
    
    # Generate HTML report
    html_report_path = os.path.join(output_dir, 'evaluation_report.html')
    generate_html_report(summary, checkpoint_info, html_report_path, compare_data)
    
    return html_report_path

def generate_evaluation_plots(df, summary, output_dir, compare_data=None):
    """
    Generate evaluation plots
    
    Args:
        df: DataFrame of episode metrics
        summary: Summary statistics
        output_dir: Directory to save plots
        compare_data: Previous evaluation data for comparison (optional)
    """
    # Set style
    sns.set(style="whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)
    
    # Create plots directory
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Reward plot
    plt.figure()
    sns.lineplot(data=df, x='episode', y='total_reward', marker='o')
    plt.axhline(y=summary['total_reward']['mean'], color='r', linestyle='--', 
               label=f"Mean: {summary['total_reward']['mean']:.2f}")
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'rewards.png'))
    plt.close()
    
    # 2. Episode length plot
    plt.figure()
    sns.lineplot(data=df, x='episode', y='length', marker='o')
    plt.axhline(y=summary['episode_length']['mean'], color='r', linestyle='--',
               label=f"Mean: {summary['episode_length']['mean']:.2f}")
    plt.title('Episode Lengths')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'lengths.png'))
    plt.close()
    
    # 3. Action distribution
    plt.figure()
    actions = list(summary['actions'].keys())
    action_freqs = [summary['actions'][a] * 100 for a in actions]  # Convert to percentages
    
    # Sort by frequency
    sorted_indices = np.argsort(action_freqs)[::-1]
    sorted_actions = [actions[i] for i in sorted_indices]
    sorted_freqs = [action_freqs[i] for i in sorted_indices]
    
    sns.barplot(x=sorted_actions, y=sorted_freqs)
    plt.title('Action Distribution')
    plt.xlabel('Action')
    plt.ylabel('Frequency (%)')
    plt.xticks(rotation=45)
    
    # Add percentage labels
    for i, freq in enumerate(sorted_freqs):
        plt.text(i, freq + 0.5, f"{freq:.1f}%", ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'action_distribution.png'))
    plt.close()
    
    # 4. Reward components
    plt.figure()
    components = list(summary['reward_components'].keys())
    values = [summary['reward_components'][c] for c in components]
    
    # Calculate percentages
    total = sum(abs(v) for v in values)
    percentages = [abs(v) / total * 100 for v in values]
    
    # Sort by absolute value
    sorted_indices = np.argsort(percentages)[::-1]
    sorted_components = [components[i] for i in sorted_indices]
    sorted_values = [values[i] for i in sorted_indices]
    sorted_percentages = [percentages[i] for i in sorted_indices]
    
    # Use different colors for positive/negative values
    colors = ['green' if v >= 0 else 'red' for v in sorted_values]
    
    bars = plt.bar(sorted_components, sorted_values, color=colors)
    plt.title('Reward Components')
    plt.ylabel('Average Reward')
    plt.xticks(rotation=45)
    
    # Add value labels
    for i, v in enumerate(sorted_values):
        plt.text(i, v + 0.1 if v >= 0 else v - 0.5, f"{v:.2f}\n({sorted_percentages[i]:.1f}%)", 
                ha='center', va='bottom' if v >= 0 else 'top', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'reward_components.png'))
    plt.close()
    
    # 5. Score progression
    plt.figure()
    sns.lineplot(data=df, x='episode', y='score', marker='o')
    plt.title('Final Scores')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.savefig(os.path.join(plots_dir, 'scores.png'))
    plt.close()
    
    # 6. Coin collection
    plt.figure()
    sns.lineplot(data=df, x='episode', y='coins', marker='o')
    plt.title('Coins Collected')
    plt.xlabel('Episode')
    plt.ylabel('Coins')
    plt.savefig(os.path.join(plots_dir, 'coins.png'))
    plt.close()
    
    # 7. Comparison plots if previous data is available
    if compare_data:
        # Load previous summary
        try:
            with open(os.path.join(compare_data, 'summary.json'), 'r') as f:
                prev_summary = json.load(f)
                
            # Reward comparison
            plt.figure()
            curr_reward = summary['total_reward']['mean']
            prev_reward = prev_summary['metrics']['total_reward']['mean']
            
            labels = ['Previous', 'Current']
            rewards = [prev_reward, curr_reward]
            
            bars = plt.bar(labels, rewards)
            plt.title('Reward Comparison')
            plt.ylabel('Average Reward')
            
            # Color bars based on improvement
            if curr_reward > prev_reward:
                bars[1].set_color('green')
            elif curr_reward < prev_reward:
                bars[1].set_color('red')
            
            # Add improvement percentage
            improvement = (curr_reward - prev_reward) / abs(prev_reward) * 100
            plt.text(1, curr_reward, f"{curr_reward:.2f}\n({improvement:+.1f}%)", 
                    ha='center', va='bottom' if improvement >= 0 else 'top')
            plt.text(0, prev_reward, f"{prev_reward:.2f}", ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'reward_comparison.png'))
            plt.close()
            
            # Score comparison
            plt.figure()
            curr_score = summary['score']['mean']
            prev_score = prev_summary['metrics']['score']['mean']
            
            labels = ['Previous', 'Current']
            scores = [prev_score, curr_score]
            
            bars = plt.bar(labels, scores)
            plt.title('Score Comparison')
            plt.ylabel('Average Score')
            
            # Color bars based on improvement
            if curr_score > prev_score:
                bars[1].set_color('green')
            elif curr_score < prev_score:
                bars[1].set_color('red')
            
            # Add improvement percentage
            improvement = (curr_score - prev_score) / abs(prev_score) * 100 if prev_score != 0 else 0
            plt.text(1, curr_score, f"{curr_score:.2f}\n({improvement:+.1f}%)", 
                    ha='center', va='bottom' if improvement >= 0 else 'top')
            plt.text(0, prev_score, f"{prev_score:.2f}", ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'score_comparison.png'))
            plt.close()
            
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Error creating comparison plots: {e}")

def generate_html_report(summary, checkpoint_info, output_path, compare_data=None):
    """
    Generate an HTML report with all evaluation results
    
    Args:
        summary: Summary statistics
        checkpoint_info: Information about the checkpoint
        output_path: Path to save the HTML report
        compare_data: Previous evaluation data for comparison (optional)
    """
    # Load previous summary if available
    prev_summary = None
    if compare_data:
        try:
            with open(os.path.join(compare_data, 'summary.json'), 'r') as f:
                prev_summary = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            logger.warning(f"Could not load previous summary from {compare_data}")
    
    # Get relative paths for plots
    plot_paths = {
        'rewards': 'plots/rewards.png',
        'lengths': 'plots/lengths.png',
        'action_dist': 'plots/action_distribution.png',
        'reward_comp': 'plots/reward_components.png',
        'scores': 'plots/scores.png',
        'coins': 'plots/coins.png'
    }
    
    if compare_data and prev_summary:
        plot_paths['reward_comparison'] = 'plots/reward_comparison.png'
        plot_paths['score_comparison'] = 'plots/score_comparison.png'
    
    # Format time duration
    total_seconds = summary['duration']['total']
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    duration_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
    
    # Generate HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Subway Surfers RL Evaluation Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }}
            h1, h2, h3 {{
                color: #2c3e50;
            }}
            .header {{
                background-color: #3498db;
                color: white;
                padding: 20px;
                text-align: center;
                margin-bottom: 30px;
                border-radius: 5px;
            }}
            .section {{
                margin-bottom: 30px;
                background-color: #f9f9f9;
                padding: 20px;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 20px;
            }}
            th, td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            .plot-container {{
                display: flex;
                flex-wrap: wrap;
                justify-content: space-around;
                margin: 20px 0;
            }}
            .plot {{
                width: 45%;
                margin-bottom: 20px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                border-radius: 5px;
                overflow: hidden;
            }}
            .plot img {{
                width: 100%;
                height: auto;
            }}
            .plot-title {{
                background-color: #f2f2f2;
                padding: 10px;
                font-weight: bold;
                text-align: center;
            }}
            .improvement {{
                color: green;
                font-weight: bold;
            }}
            .deterioration {{
                color: red;
                font-weight: bold;
            }}
            .footer {{
                text-align: center;
                margin-top: 30px;
                padding: 10px;
                font-size: 12px;
                color: #7f8c8d;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Subway Surfers RL Evaluation Report</h1>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="section">
            <h2>Evaluation Summary</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                    {f"<th>Previous</th><th>Change</th>" if prev_summary else ""}
                </tr>
                <tr>
                    <td>Number of Episodes</td>
                    <td>{summary['episodes']}</td>
                    {f"<td>{prev_summary['metrics']['episodes']}</td><td>-</td>" if prev_summary else ""}
                </tr>
                <tr>
                    <td>Average Reward</td>
                    <td>{summary['total_reward']['mean']:.2f} ± {summary['total_reward']['std']:.2f}</td>
                    {f"<td>{prev_summary['metrics']['total_reward']['mean']:.2f}</td><td class='{'improvement' if summary['total_reward']['mean'] > prev_summary['metrics']['total_reward']['mean'] else 'deterioration'}'>{((summary['total_reward']['mean'] - prev_summary['metrics']['total_reward']['mean']) / abs(prev_summary['metrics']['total_reward']['mean']) * 100):.1f}%</td>" if prev_summary else ""}
                </tr>
                <tr>
                    <td>Average Episode Length</td>
                    <td>{summary['episode_length']['mean']:.1f} ± {summary['episode_length']['std']:.1f} steps</td>
                    {f"<td>{prev_summary['metrics']['episode_length']['mean']:.1f}</td><td class='{'improvement' if summary['episode_length']['mean'] > prev_summary['metrics']['episode_length']['mean'] else 'deterioration'}'>{((summary['episode_length']['mean'] - prev_summary['metrics']['episode_length']['mean']) / abs(prev_summary['metrics']['episode_length']['mean']) * 100):.1f}%</td>" if prev_summary else ""}
                </tr>
                <tr>
                    <td>Average Score</td>
                    <td>{summary['score']['mean']:.1f} ± {summary['score']['std']:.1f}</td>
                    {f"<td>{prev_summary['metrics']['score']['mean']:.1f}</td><td class='{'improvement' if summary['score']['mean'] > prev_summary['metrics']['score']['mean'] else 'deterioration'}'>{((summary['score']['mean'] - prev_summary['metrics']['score']['mean']) / abs(prev_summary['metrics']['score']['mean']) * 100 if prev_summary['metrics']['score']['mean'] != 0 else 0):.1f}%</td>" if prev_summary else ""}
                </tr>
                <tr>
                    <td>Average Coins</td>
                    <td>{summary['coins']['mean']:.1f} ± {summary['coins']['std']:.1f}</td>
                    {f"<td>{prev_summary['metrics']['coins']['mean']:.1f}</td><td class='{'improvement' if summary['coins']['mean'] > prev_summary['metrics']['coins']['mean'] else 'deterioration'}'>{((summary['coins']['mean'] - prev_summary['metrics']['coins']['mean']) / abs(prev_summary['metrics']['coins']['mean']) * 100 if prev_summary['metrics']['coins']['mean'] != 0 else 0):.1f}%</td>" if prev_summary else ""}
                </tr>
                <tr>
                    <td>Best Reward</td>
                    <td>{summary['total_reward']['max']:.2f}</td>
                    {f"<td>{prev_summary['metrics']['total_reward']['max']:.2f}</td><td class='{'improvement' if summary['total_reward']['max'] > prev_summary['metrics']['total_reward']['max'] else 'deterioration'}'>{((summary['total_reward']['max'] - prev_summary['metrics']['total_reward']['max']) / abs(prev_summary['metrics']['total_reward']['max']) * 100):.1f}%</td>" if prev_summary else ""}
                </tr>
                <tr>
                    <td>Worst Reward</td>
                    <td>{summary['total_reward']['min']:.2f}</td>
                    {f"<td>{prev_summary['metrics']['total_reward']['min']:.2f}</td><td>-</td>" if prev_summary else ""}
                </tr>
                <tr>
                    <td>Total Evaluation Time</td>
                    <td>{duration_str}</td>
                    {f"<td>-</td><td>-</td>" if prev_summary else ""}
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>Model Information</h2>
            <table>
                <tr>
                    <th>Parameter</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Checkpoint Path</td>
                    <td>{checkpoint_info.get('path', 'Unknown')}</td>
                </tr>
                <tr>
                    <td>Training Episodes</td>
                    <td>{len(checkpoint_info.get('episode_rewards', []))}</td>
                </tr>
                <tr>
                    <td>Dueling DQN</td>
                    <td>{'Yes' if checkpoint_info.get('use_dueling', False) else 'No'}</td>
                </tr>
                <tr>
                    <td>Double DQN</td>
                    <td>{'Yes' if checkpoint_info.get('use_double', False) else 'No'}</td>
                </tr>
                <tr>
                    <td>Prioritized Experience Replay</td>
                    <td>{'Yes' if checkpoint_info.get('use_per', False) else 'No'}</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>Performance Visualizations</h2>
            
            <div class="plot-container">
                <div class="plot">
                    <div class="plot-title">Episode Rewards</div>
                    <img src="{plot_paths['rewards']}" alt="Episode Rewards">
                </div>
                
                <div class="plot">
                    <div class="plot-title">Episode Lengths</div>
                    <img src="{plot_paths['lengths']}" alt="Episode Lengths">
                </div>
            </div>
            
            <div class="plot-container">
                <div class="plot">
                    <div class="plot-title">Action Distribution</div>
                    <img src="{plot_paths['action_dist']}" alt="Action Distribution">
                </div>
                
                <div class="plot">
                    <div class="plot-title">Reward Components</div>
                    <img src="{plot_paths['reward_comp']}" alt="Reward Components">
                </div>
            </div>
            
            <div class="plot-container">
                <div class="plot">
                    <div class="plot-title">Game Scores</div>
                    <img src="{plot_paths['scores']}" alt="Game Scores">
                </div>
                
                <div class="plot">
                    <div class="plot-title">Coins Collected</div>
                    <img src="{plot_paths['coins']}" alt="Coins Collected">
                </div>
            </div>
            
            {
                f'''
                <h3>Comparison with Previous Evaluation</h3>
                <div class="plot-container">
                    <div class="plot">
                        <div class="plot-title">Reward Comparison</div>
                        <img src="{plot_paths['reward_comparison']}" alt="Reward Comparison">
                    </div>
                    
                    <div class="plot">
                        <div class="plot-title">Score Comparison</div>
                        <img src="{plot_paths['score_comparison']}" alt="Score Comparison">
                    </div>
                </div>
                ''' if compare_data and prev_summary else ""
            }
        </div>
        
        <div class="section">
            <h2>Action Analysis</h2>
            <p>This section shows the distribution of actions taken by the agent.</p>
            
            <table>
                <tr>
                    <th>Action</th>
                    <th>Frequency</th>
                    {f"<th>Previous</th><th>Change</th>" if prev_summary and 'actions' in prev_summary['metrics'] else ""}
                </tr>
                {"".join([f"<tr><td>{action}</td><td>{summary['actions'][action]*100:.1f}%</td>{f'<td>{prev_summary['metrics']['actions'].get(action, 0)*100:.1f}%</td><td>{(summary['actions'][action] - prev_summary['metrics']['actions'].get(action, 0))*100:.1f}%</td>' if prev_summary and 'actions' in prev_summary['metrics'] else ''}</tr>" for action in summary['actions']])}
            </table>
        </div>
        
        <div class="section">
            <h2>Reward Components</h2>
            <p>This section breaks down the contributions of different components to the total reward.</p>
            
            <table>
                <tr>
                    <th>Component</th>
                    <th>Average Value</th>
                    <th>Percentage</th>
                    {f"<th>Previous</th><th>Change</th>" if prev_summary and 'reward_components' in prev_summary['metrics'] else ""}
                </tr>
                {generate_reward_component_rows(summary, prev_summary)}
            </table>
        </div>
        
        <div class="footer">
            <p>Subway Surfers RL Evaluation - Generated by Evaluation Script</p>
        </div>
    </body>
    </html>
    """
    
    # Write HTML to file
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    return output_path

def generate_reward_component_rows(summary, prev_summary):
    """Generate HTML rows for reward components table"""
    components = summary['reward_components']
    total = sum(abs(v) for v in components.values())
    
    rows = []
    for comp, value in components.items():
        percentage = abs(value) / total * 100 if total != 0 else 0
        
        # Calculate change if previous data is available
        change_html = ""
        if prev_summary and 'reward_components' in prev_summary['metrics'] and comp in prev_summary['metrics']['reward_components']:
            prev_value = prev_summary['metrics']['reward_components'][comp]
            change = value - prev_value
            change_class = 'improvement' if (value > prev_value and comp != 'penalty') or (value < prev_value and comp == 'penalty') else 'deterioration'
            change_html = f'<td>{prev_value:.2f}</td><td class="{change_class}">{change:+.2f}</td>'
        
        rows.append(f"<tr><td>{comp.capitalize()}</td><td>{value:.2f}</td><td>{percentage:.1f}%</td>{change_html}</tr>")
    
    return "".join(rows)

def main():
    # Parse arguments
    args = parse_args()
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    try:
        # Display welcome message
        print("\n" + "="*80)
        print("Subway Surfers AI - Evaluation Mode".center(80))
        print("="*80)
        print(f"Checkpoint: {args.checkpoint}")
        print(f"Episodes: {args.episodes}")
        print(f"Output Directory: {args.output_dir}")
        if args.compare:
            print(f"Comparing with: {args.compare}")
        print("-"*80 + "\n")
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(args.output_dir, f"eval_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Create environment
        logger.info("Creating environment...")
        env = create_frame_stack_env(args)
        
        # Load agent
        logger.info(f"Loading agent from checkpoint: {args.checkpoint}")
        agent, checkpoint = load_agent(args.checkpoint, device)
        
        # Add checkpoint path to info
        checkpoint_info = {
            'path': args.checkpoint,
            **checkpoint
        }
        
        # Set agent to evaluation mode and disable exploration
        agent.epsilon = 0
        if hasattr(agent, 'policy_net'):
            agent.policy_net.eval()
        
        # Run evaluation episodes
        episode_metrics = []
        best_reward = float('-inf')
        
        for episode in range(args.episodes):
            logger.info(f"Starting evaluation episode {episode+1}/{args.episodes}")
            
            # Run episode
            metrics = run_evaluation_episode(
                agent, env, episode+1, 
                best_reward=best_reward if args.record_best else None,
                record_best=args.record_best
            )
            
            # Update best reward
            if metrics['total_reward'] > best_reward:
                best_reward = metrics['total_reward']
            
            # Store metrics
            episode_metrics.append(metrics)
            
            # Log episode metrics
            logger.info(f"Episode {episode+1} - Reward: {metrics['total_reward']:.2f}, "
                       f"Length: {metrics['length']}, Score: {metrics['score']}, "
                       f"Coins: {metrics['coins']}")
        
        # Close environment
        env.close()
        
        # Generate evaluation report
        logger.info("Generating evaluation report...")
        report_path = generate_evaluation_report(
            episode_metrics, checkpoint_info, output_dir,
            compare_data=args.compare
        )
        
        # Display final results
        print("\n" + "="*80)
        print("Evaluation Results".center(80))
        print("="*80)
        
        # Calculate summary statistics
        avg_reward = np.mean([m['total_reward'] for m in episode_metrics])
        avg_length = np.mean([m['length'] for m in episode_metrics])
        avg_score = np.mean([m['score'] for m in episode_metrics])
        
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Average Episode Length: {avg_length:.1f}")
        print(f"Average Score: {avg_score:.1f}")
        print(f"Best Episode Reward: {best_reward:.2f}")
        print(f"Reports saved to: {os.path.abspath(output_dir)}")
        print(f"Detailed HTML report: {os.path.abspath(report_path)}")
        
        if args.record_best:
            print(f"Best episode video saved to {os.path.abspath('videos/')}")
        
        print("="*80 + "\n")
        
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}", exc_info=True)
        print(f"\nError: {str(e)}")
    finally:
        # Clean up
        if 'env' in locals():
            env.close()
        
        # Close all OpenCV windows
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()