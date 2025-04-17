# main.py - Complete implementation
import os
import logging
import argparse
import torch
import numpy as np
from datetime import datetime
import time
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import custom modules
from env.game_interaction import SubwaySurfersEnv
from env.capture_state import StateCapture
from model.agent import DQNAgent
from model.training import Trainer
from utils.utils import check_gpu, optimize_for_gpu, measure_inference_time, monitor_memory_usage, format_time
from utils.plot_utils import plot_training_metrics, plot_learning_curve, create_training_summary_report

# Configure logging
log_file = f"subway_surfers_rl_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/{log_file}"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Main")

class FrameStackingEnv:
    """
    Wrapper environment that handles frame stacking for DQN
    """
    def __init__(self, env, state_capture, stack_size=4):
        self.env = env
        self.state_capture = state_capture
        self.stack_size = stack_size
        self.frames = []
        self.actions = env.actions
        
    def reset(self):
        # Reset the base environment
        _ = self.env.reset()
        
        # Reset state capture
        self.state_capture.reset()
        
        # Reset the frame stack
        self.frames = []
        
        # Get initial state by stacking the same frame
        frame = self.env.capture_screen()
        processed_frame = self.env.preprocess_frame(frame)
        
        # Clear frame stack and add initial frame multiple times
        self.frames = [processed_frame] * self.stack_size
        
        # Return stacked frames as the state
        return np.array(self.frames)
    
    def step(self, action):
        # Take action in the environment
        next_frame, reward, done, info = self.env.step(action)
        
        # Store preprocessed frame
        processed_frame = next_frame  # Already preprocessed by env.step()
        
        # Update frame stack (remove oldest, add newest)
        self.frames.pop(0)
        self.frames.append(processed_frame)
        
        # Return state as stacked frames
        return np.array(self.frames), reward, done, info
    
    def close(self):
        self.env.close()
    
    def visualize_agent_state(self, state, step_num, episode_num):
        """Forward to the environment's visualize method"""
        self.env.visualize_agent_state(state, step_num, episode_num)
    
    def update_training_stats(self, epsilon=None, loss=None, avg_reward=None):
        """Forward to the environment's update_training_stats method"""
        if hasattr(self.env, 'update_training_stats'):
            self.env.update_training_stats(epsilon, loss, avg_reward)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train a RL agent to play Subway Surfers")
    
    # Environment settings
    parser.add_argument("--game_url", type=str, default="https://poki.com/en/g/subway-surfers", 
                        help="URL to the Subway Surfers game")
    parser.add_argument("--browser_position", type=str, default="right", choices=["left", "right"],
                        help="Position of the browser window for split-screen viewing")
    
    # Training settings
    parser.add_argument("--max_episodes", type=int, default=1000, 
                        help="Maximum number of episodes to train for")
    parser.add_argument("--max_steps", type=int, default=5000, 
                        help="Maximum steps per episode")
    parser.add_argument("--save_interval", type=int, default=10, 
                        help="Save checkpoint every N episodes")
    parser.add_argument("--eval_interval", type=int, default=20, 
                        help="Evaluate agent every N episodes")
    parser.add_argument("--visual_feedback", action="store_true",
                        help="Show visual feedback during training")
    
    # Model settings
    parser.add_argument("--use_dueling", action="store_true", 
                        help="Use Dueling DQN architecture")
    parser.add_argument("--use_double", action="store_true", 
                        help="Use Double DQN algorithm")
    parser.add_argument("--use_per", action="store_true", 
                        help="Use Prioritized Experience Replay")
    parser.add_argument("--frame_stack", type=int, default=4,
                        help="Number of frames to stack for state representation")
    
    # Checkpoint settings
    parser.add_argument("--load_checkpoint", type=str, default=None, 
                        help="Path to checkpoint file to load")
    
    # Hyperparameters
    parser.add_argument("--learning_rate", type=float, default=0.0001, 
                        help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, 
                        help="Discount factor for future rewards")
    parser.add_argument("--batch_size", type=int, default=32, 
                        help="Batch size for training")
    parser.add_argument("--target_update", type=int, default=10, 
                        help="Update target network every N episodes")
    parser.add_argument("--memory_capacity", type=int, default=10000, 
                        help="Capacity of replay buffer")
    parser.add_argument("--epsilon_start", type=float, default=1.0,
                        help="Starting value of epsilon for exploration")
    parser.add_argument("--epsilon_end", type=float, default=0.01,
                        help="Minimum value of epsilon")
    parser.add_argument("--epsilon_decay", type=float, default=0.9995,
                        help="Decay rate of epsilon per episode")
    
    # Debug settings
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode with extra logging and visualizations")
    parser.add_argument("--record_video", action="store_true",
                        help="Record video of gameplay")
    parser.add_argument("--detailed_monitoring", action="store_true",
                        help="Enable detailed state monitoring with visualizations")
    parser.add_argument("--show_regions", action="store_true",
                        help="Show game regions in real-time with bounding boxes")
    parser.add_argument("--monitor_memory", action="store_true",
                        help="Monitor memory usage during training")
    
    # Parse arguments
    args = parser.parse_args()
    
    return args

def create_environment(args):
    """Create and initialize the game environment"""
    logger.info("Creating environment...")
    
    # Create directories for debug output
    os.makedirs("debug_images", exist_ok=True)
    os.makedirs("debug_images/states", exist_ok=True)
    
    # Create environment
    try:
        env = SubwaySurfersEnv(
            game_url=args.game_url,
            render_mode="human" if args.visual_feedback or args.show_regions else None,
            position=args.browser_position
        )
        
        # Set up state capture
        state_capture = StateCapture(
            game_region=env.game_region,
            target_size=(84, 84),
            stack_frames=args.frame_stack,
            debug=args.debug
        )
        
        # Wrap environment with frame stacking
        stacked_env = FrameStackingEnv(env, state_capture, stack_size=args.frame_stack)
        
        logger.info(f"Environment created successfully with {args.frame_stack} frame stacking")
        return stacked_env
    
    except Exception as e:
        logger.error(f"Error creating environment: {str(e)}")
        raise

def create_agent(env, args, device):
    """Create and initialize the DQN agent"""
    logger.info("Creating agent...")
    
    # Determine state shape (frames, height, width)
    state_shape = (args.frame_stack, 84, 84)
    n_actions = len(env.actions)  # Number of possible actions
    
    # Optimize hyperparameters for GPU if available
    batch_size, memory_capacity = optimize_for_gpu(
        batch_size=args.batch_size,
        state_shape=state_shape
    )
    
    logger.info(f"Using batch size: {batch_size}, memory capacity: {memory_capacity}")
    
    # Create agent
    try:
        # Create checkpoint directory
        os.makedirs("logs/checkpoints", exist_ok=True)
        
        agent = DQNAgent(
            state_shape=state_shape,
            n_actions=n_actions,
            checkpoint_dir="./logs/checkpoints/",
            use_dueling=args.use_dueling,
            use_double=args.use_double,
            use_per=args.use_per,
            device=device
        )
        
        # Update hyperparameters from command line arguments
        agent.gamma = args.gamma
        agent.batch_size = batch_size
        agent.learning_rate = args.learning_rate
        agent.target_update = args.target_update
        agent.memory_capacity = memory_capacity
        agent.epsilon_start = args.epsilon_start
        agent.epsilon_end = args.epsilon_end
        agent.epsilon_decay = args.epsilon_decay
        agent.epsilon = args.epsilon_start
        
        # Update optimizer learning rate
        for param_group in agent.optimizer.param_groups:
            param_group['lr'] = args.learning_rate
        
        # Load checkpoint if specified
        start_episode = 0
        if args.load_checkpoint:
            if os.path.exists(args.load_checkpoint):
                start_episode, _ = agent.load_checkpoint(args.load_checkpoint)
                logger.info(f"Loaded checkpoint: {args.load_checkpoint}")
            else:
                logger.warning(f"Checkpoint not found: {args.load_checkpoint}")
        
        logger.info(f"Agent created with {n_actions} actions and state shape {state_shape}")
        
        return agent, start_episode
        
    except Exception as e:
        logger.error(f"Error creating agent: {str(e)}")
        raise

def record_gameplay_video(env, agent, output_path, max_frames=1000):
    """Record a video of the agent playing the game"""
    logger.info("Recording gameplay video...")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30
    frame_size = (env.env.game_region[2], env.env.game_region[3])  # Get from base env
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    
    # Reset the environment
    state = env.reset()
    
    # Convert state from stacked frames to single state tensor if needed
    if isinstance(state, np.ndarray) and len(state.shape) == 3:
        # No need to unsqueeze here, select_action will handle it
        state_tensor = state
    else:
        state_tensor = state
    
    frames_recorded = 0
    done = False
    total_reward = 0
    
    try:
        # Record gameplay until done or max frames reached
        while not done and frames_recorded < max_frames:
            # Capture raw frame for video from base environment
            raw_frame = env.env.capture_screen()
            
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR)
            
            # Add frame number and reward
            cv2.putText(frame_bgr, f"Frame: {frames_recorded}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame_bgr, f"Reward: {total_reward:.2f}", 
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Write frame to video
            video_writer.write(frame_bgr)
            
            # Select action
            action = agent.select_action(state_tensor, training=False)
            
            # Take action in environment
            next_state, reward, done, _ = env.step(action)
            
            # Update state and accumulate reward
            state_tensor = next_state
            total_reward += reward
            
            frames_recorded += 1
            
    except Exception as e:
        logger.error(f"Error during video recording: {str(e)}")
    
    finally:
        # Release video writer
        video_writer.release()
        logger.info(f"Gameplay video saved to {output_path} ({frames_recorded} frames)")
        
        return total_reward, frames_recorded

def main():
    """Main function"""
    # Parse command line arguments
    args = parse_args()
    
    # Display welcome message
    print("\n" + "="*80)
    print("Subway Surfers Reinforcement Learning Training".center(80))
    print("="*80 + "\n")
    
    # Create necessary directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("logs/checkpoints", exist_ok=True)
    os.makedirs("logs/videos", exist_ok=True)
    os.makedirs("logs/plots", exist_ok=True)
    os.makedirs("debug_images", exist_ok=True)
    
    # Log all arguments
    logger.info(f"Training with arguments: {vars(args)}")
    
    # Check GPU
    device = check_gpu()
    
    # Log start time
    start_time = time.time()
    
    env = None
    
    try:
        # Create environment
        env = create_environment(args)
        
        # Create agent
        agent, start_episode = create_agent(env, args, device)
        
        # Create trainer
        trainer = Trainer(
            agent=agent,
            env=env,
            log_dir="./logs/",
            save_interval=args.save_interval,
            eval_interval=args.eval_interval,
            max_episodes=args.max_episodes,
            max_steps_per_episode=args.max_steps,
            detailed_monitoring=args.detailed_monitoring
        )
        
        # Measure model inference time
        logger.info("Measuring inference time...")
        dummy_input_shape = (1, *agent.state_shape)  # Batch size of 1
        inference_time = measure_inference_time(
            model=agent.policy_net,
            input_shape=dummy_input_shape,
            num_trials=100,
            device=device
        )
        logger.info(f"Average inference time: {inference_time:.2f} ms")
        
        # Monitor initial memory usage
        if args.monitor_memory:
            monitor_memory_usage()
        
        # Train the agent
        logger.info(f"Starting training from episode {start_episode}...")
        print("\nStarting training. Press Ctrl+C to stop.\n")
        
        # Set up memory monitoring timer
        if args.monitor_memory:
            last_memory_check = time.time()
            memory_check_interval = 300  # Check every 5 minutes
        
        # Initial timestamp for ETA calculation
        training_start = time.time()
        
        # Train the agent
        try:
            rewards = trainer.train(start_episode=start_episode)
            
            # Calculate total training time
            training_time = time.time() - start_time
            hours, remainder = divmod(training_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            logger.info(f"Training completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
            
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            print("\nTraining interrupted. Saving final metrics...\n")
            rewards = trainer.episode_rewards
        
        # Plot final metrics
        logger.info("Plotting final metrics...")
        plot_training_metrics(
            rewards=trainer.episode_rewards,
            lengths=trainer.episode_lengths,
            epsilons=[agent.epsilon_start * (agent.epsilon_decay ** i) for i in range(len(trainer.episode_rewards))],
            losses=trainer.losses,
            save_dir="./logs/plots"
        )
        
        # Log final performance statistics
        if hasattr(agent, 'log_performance_stats'):
            agent.log_performance_stats()
            
        if hasattr(trainer, 'log_performance_stats'):
            trainer.log_performance_stats()
        
        # Plot learning curve from CSV
        logger.info("Creating training summary report...")
        create_training_summary_report(
            csv_path=trainer.csv_path,
            save_dir="./logs/"
        )
        
        # Record a video of the trained agent
        if args.record_video:
            video_path = f"logs/videos/gameplay_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            final_reward, frames_recorded = record_gameplay_video(env, agent, video_path)
            logger.info(f"Recorded gameplay video with reward: {final_reward:.2f} over {frames_recorded} frames")
        
        # Final memory usage check
        if args.monitor_memory:
            monitor_memory_usage()
        
        logger.info("Training complete!")
        print("\n" + "="*80)
        print("Training completed successfully!".center(80))
        print(f"Training time: {int(hours)}h {int(minutes)}m {int(seconds)}s".center(80))
        print(f"Checkpoints saved in: {os.path.abspath('logs/checkpoints')}".center(80))
        print(f"Final reward: {rewards[-1]:.2f}".center(80))
        print("="*80 + "\n")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        print("\nTraining interrupted. Saving checkpoint...\n")
        
        # Save checkpoint on interrupt
        if 'agent' in locals() and 'trainer' in locals():
            checkpoint_path = agent.save_checkpoint(
                episode=trainer.current_episode if hasattr(trainer, 'current_episode') else 0,
                rewards=trainer.episode_rewards[-min(10, len(trainer.episode_rewards)):] if hasattr(trainer, 'episode_rewards') and trainer.episode_rewards else [],
                avg_reward=np.mean(trainer.episode_rewards[-min(10, len(trainer.episode_rewards)):]) if hasattr(trainer, 'episode_rewards') and trainer.episode_rewards else 0,
                suffix="interrupted"
            )
            print(f"Checkpoint saved to: {checkpoint_path}\n")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True)
        print(f"\nError during training: {str(e)}\n")
    finally:
        # Clean up
        if env is not None:
            env.close()
            logger.info("Environment closed")

if __name__ == "__main__":
    main()