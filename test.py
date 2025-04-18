#!/usr/bin/env python
"""
Subway Surfers AI - Visualization Script

This script loads a trained agent from a checkpoint and visualizes its performance
without any training.
"""

import os
import argparse
import logging
import torch
import numpy as np
import time
import cv2
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm

# Configure logging
os.makedirs("logs", exist_ok=True)
log_file = f"subway_surfers_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/{log_file}"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Test")

def parse_args():
    parser = argparse.ArgumentParser(description="Test a trained Subway Surfers AI agent")
    
    # Basic settings
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint file to load")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of episodes to run")
    parser.add_argument("--browser_position", type=str, default="right", choices=["left", "right"],
                        help="Position of the browser window")
    parser.add_argument("--record", action="store_true",
                        help="Record videos of gameplay")
    
    # Display options
    parser.add_argument("--show_q_values", action="store_true",
                        help="Show Q-values on screen during gameplay")
    parser.add_argument("--show_states", action="store_true",
                        help="Show preprocessed states on screen")
    
    # Debug options
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode")
    parser.add_argument("--sleep_between_actions", type=float, default=0.0,
                        help="Sleep time between actions for better visualization")
    
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
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
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
    
    return agent

def visualize_q_values(q_values, actions, selected_action, img_shape=(400, 600)):
    """Create a visualization of Q-values"""
    # Create a blank image
    img = np.ones((img_shape[0], img_shape[1], 3), dtype=np.uint8) * 255
    
    # Parameters
    bar_height = 40
    bar_margin = 10
    max_bar_width = img_shape[1] - 100
    text_offset = 5
    
    # Normalize Q-values for visualization
    q_min = q_values.min()
    q_max = q_values.max()
    q_range = max(q_max - q_min, 1e-5)  # Avoid division by zero
    
    # Add title
    cv2.putText(img, "Action Q-Values", (20, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # Draw bars for each action
    for i, (action, q_val) in enumerate(zip(actions, q_values)):
        y_pos = 70 + i * (bar_height + bar_margin)
        
        # Calculate bar width
        norm_q = (q_val - q_min) / q_range
        bar_width = int(norm_q * max_bar_width)
        
        # Determine bar color (green for selected action, blue for others)
        color = (0, 180, 0) if i == selected_action else (180, 0, 0)
        
        # Draw action label
        cv2.putText(img, action, (20, y_pos + bar_height//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Draw bar
        cv2.rectangle(img, (80, y_pos), (80 + bar_width, y_pos + bar_height),
                     color, -1)
        
        # Draw Q-value
        cv2.putText(img, f"{q_val:.2f}", (90 + bar_width, y_pos + bar_height//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Add selected action text
    y_pos = 70 + len(actions) * (bar_height + bar_margin) + 20
    cv2.putText(img, f"Selected: {actions[selected_action]}", (20, y_pos),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    return img

def visualize_state(state, img_shape=(400, 400)):
    """Create a visualization of the stacked state frames"""
    # Number of frames in the stack
    n_frames = state.shape[0]
    
    # Create a blank image
    img = np.ones((img_shape[0], img_shape[1], 3), dtype=np.uint8) * 255
    
    # Grid size
    grid_size = int(np.ceil(np.sqrt(n_frames)))
    
    # Calculate frame size in the grid
    frame_height = img_shape[0] // grid_size
    frame_width = img_shape[1] // grid_size
    
    # Place each frame in the grid
    for i in range(n_frames):
        row = i // grid_size
        col = i % grid_size
        
        # Get the frame and resize to fit the grid
        frame = state[i]
        frame = cv2.resize((frame * 255).astype(np.uint8), 
                          (frame_width, frame_height))
        
        # Convert grayscale to BGR for display
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        # Insert frame into the image
        y_start = row * frame_height
        y_end = (row + 1) * frame_height
        x_start = col * frame_width
        x_end = (col + 1) * frame_width
        
        img[y_start:y_end, x_start:x_end] = frame_bgr
        
        # Add frame number
        cv2.putText(img, f"t-{n_frames-i-1}", (x_start + 5, y_start + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    return img

def run_episode(agent, env, episode_num, args):
    """
    Run a single episode with visualization
    
    Args:
        agent: Trained agent
        env: Environment
        episode_num: Episode number
        args: Command line arguments
        
    Returns:
        Tuple of (total_reward, episode_length)
    """
    # Reset environment
    state = env.reset()
    
    # Initialize video writer if recording
    video_writer = None
    if args.record:
        os.makedirs("videos", exist_ok=True)
        video_path = f"videos/episode_{episode_num}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_path, fourcc, 10.0, (1200, 600))
        logger.info(f"Recording video to {video_path}")
    
    # Prepare visualization windows
    if args.show_q_values:
        cv2.namedWindow('Q-Values', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Q-Values', 400, 600)
    
    if args.show_states:
        cv2.namedWindow('State', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('State', 400, 400)
    
    # Run episode
    done = False
    total_reward = 0
    steps = 0
    
    with tqdm(desc=f"Episode {episode_num}", unit="steps") as pbar:
        while not done:
            # Convert state to tensor
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(agent.device)
            
            # Get Q-values
            with torch.no_grad():
                q_values = agent.policy_net(state_tensor).cpu().numpy()[0]
            
            # Select action (always greedy in test mode)
            action = q_values.argmax()
            
            # Visualize Q-values if enabled
            if args.show_q_values:
                q_vis = visualize_q_values(q_values, env.actions, action)
                cv2.imshow('Q-Values', q_vis)
            
            # Visualize state if enabled
            if args.show_states:
                state_vis = visualize_state(state)
                cv2.imshow('State', state_vis)
            
            # Combine visualizations for video
            if video_writer is not None:
                # Create a combined frame
                combined = np.ones((600, 1200, 3), dtype=np.uint8) * 255
                
                # Add state visualization
                state_vis = visualize_state(state)
                combined[100:500, 50:450] = state_vis
                
                # Add Q-values visualization
                q_vis = visualize_q_values(q_values, env.actions, action)
                combined[0:600, 600:1200] = q_vis
                
                # Add info text
                cv2.putText(combined, f"Episode: {episode_num}, Step: {steps}", (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                cv2.putText(combined, f"Reward: {total_reward:.2f}", (50, 580),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                
                # Write to video
                video_writer.write(combined)
            
            # Wait for key press if visualization is enabled
            if args.show_q_values or args.show_states:
                key = cv2.waitKey(1)
                if key == 27:  # ESC key
                    logger.info("Test stopped by user")
                    break
            
            # Take action in environment
            next_state, reward, done, info = env.step(action)
            
            # Sleep between actions if specified (for better visualization)
            if args.sleep_between_actions > 0:
                time.sleep(args.sleep_between_actions)
            
            # Update state
            state = next_state
            total_reward += reward
            steps += 1
            
            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({
                'reward': f"{total_reward:.2f}",
                'score': info.get('score', 0),
                'coins': info.get('coins', 0)
            })
            
            # Safety check for max steps
            if steps >= 10000:
                logger.warning("Episode reached maximum steps (10000)")
                break
    
    # Close video writer
    if video_writer is not None:
        video_writer.release()
    
    # Close visualization windows
    if args.show_q_values:
        cv2.destroyWindow('Q-Values')
    if args.show_states:
        cv2.destroyWindow('State')
    
    return total_reward, steps

def main():
    # Parse arguments
    args = parse_args()
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    try:
        # Display welcome message
        print("\n" + "="*80)
        print("Subway Surfers AI - Test Mode".center(80))
        print("="*80)
        print(f"Checkpoint: {args.checkpoint}")
        print(f"Episodes: {args.episodes}")
        print(f"Recording: {'Yes' if args.record else 'No'}")
        print("-"*80 + "\n")
        
        # Create environment
        logger.info("Creating environment...")
        env = create_frame_stack_env(args)
        
        # Load agent
        logger.info(f"Loading agent from checkpoint: {args.checkpoint}")
        agent = load_agent(args.checkpoint, device)
        
        # Set agent to evaluation mode and disable exploration
        agent.epsilon = 0
        if hasattr(agent, 'policy_net'):
            agent.policy_net.eval()
        
        # Run episodes
        all_rewards = []
        all_lengths = []
        
        for episode in range(args.episodes):
            logger.info(f"Starting episode {episode+1}/{args.episodes}")
            
            # Run episode
            reward, length = run_episode(agent, env, episode+1, args)
            
            # Log results
            logger.info(f"Episode {episode+1} - Reward: {reward:.2f}, Length: {length}")
            all_rewards.append(reward)
            all_lengths.append(length)
        
        # Close environment
        env.close()
        
        # Display final results
        avg_reward = np.mean(all_rewards)
        avg_length = np.mean(all_lengths)
        print("\n" + "="*80)
        print("Test Results".center(80))
        print("="*80)
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Average Episode Length: {avg_length:.1f}")
        print(f"Best Episode Reward: {max(all_rewards):.2f}")
        print("="*80 + "\n")
        
        # Plot results
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, args.episodes+1), all_rewards, 'o-')
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.grid(True, alpha=0.3)
        
        # Save plot
        os.makedirs("results", exist_ok=True)
        plot_path = f"results/test_rewards_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_path)
        print(f"Reward plot saved to {plot_path}")
        
        if args.record:
            print(f"Videos saved to {os.path.abspath('videos/')}")
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        logger.error(f"Error during test: {str(e)}", exc_info=True)
        print(f"\nError: {str(e)}")
    finally:
        # Clean up
        if 'env' in locals():
            env.close()
        
        # Close all OpenCV windows
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()