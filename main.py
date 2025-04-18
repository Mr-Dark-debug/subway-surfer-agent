# main.py - Optimized implementation for training a RL agent to play Subway Surfers
import os
import logging
import argparse
import torch
import numpy as np
from datetime import datetime
import time
import matplotlib.pyplot as plt
import gc
import psutil
import json
import shutil
import warnings
from tqdm import tqdm
import platform

# Filter excessive warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Import custom modules
from env.game_interaction import SubwaySurfersEnv
from model.agent import DQNAgent, AdaptiveDQNAgent
from model.training import Trainer
from utils.utils import check_gpu, optimize_for_gpu

# Configure logging
os.makedirs("logs", exist_ok=True)
log_file = f"subway_surfers_rl_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
    def __init__(self, env, stack_size=4):
        """
        Initialize frame stacking environment wrapper
        
        Args:
            env: Base environment
            stack_size: Number of frames to stack
        """
        self.env = env
        self.stack_size = stack_size
        self.frames = []
        self.actions = env.actions
        
    def reset(self):
        """Reset environment and return initial stacked state"""
        # Reset the base environment
        observation = self.env.reset()
        
        # Reset the frame stack
        self.frames = []
        
        # Stack the initial frame multiple times
        for _ in range(self.stack_size):
            self.frames.append(observation)
        
        # Return stacked frames as the state
        return np.array(self.frames)
    
    def step(self, action):
        """
        Take a step in the environment
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (stacked_state, reward, done, info)
        """
        # Take action in the environment
        next_observation, reward, done, info = self.env.step(action)
        
        # Update frame stack (remove oldest, add newest)
        self.frames.pop(0)
        self.frames.append(next_observation)
        
        # Return state as stacked frames
        return np.array(self.frames), reward, done, info
    
    def close(self):
        """Close the environment"""
        self.env.close()
    
    def visualize_agent_state(self, state, step_num, episode_num):
        """Forward to the environment's visualize method"""
        self.env.visualize_agent_state(state, step_num, episode_num)
    
    def update_training_stats(self, epsilon=None, loss=None, avg_reward=None):
        """Forward to the environment's update_training_stats method"""
        if hasattr(self.env, 'update_training_stats'):
            self.env.update_training_stats(epsilon, loss, avg_reward)
    
    def log_performance_stats(self):
        """Forward to the environment's log_performance_stats method"""
        if hasattr(self.env, 'log_performance_stats'):
            self.env.log_performance_stats()

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train a RL agent to play Subway Surfers")
    
    # Environment settings
    parser.add_argument("--game_url", type=str, default="https://poki.com/en/g/subway-surfers", 
                        help="URL to the game")
    parser.add_argument("--browser_position", type=str, default="right", choices=["left", "right"],
                        help="Position of the browser window for split-screen viewing")
    parser.add_argument("--use_existing_browser", action="store_true",
                        help="Use existing browser window with the game already opened")
    
    # Training settings
    parser.add_argument("--max_episodes", type=int, default=500, 
                        help="Maximum number of episodes to train for")
    parser.add_argument("--max_steps", type=int, default=5000, 
                        help="Maximum steps per episode")
    parser.add_argument("--save_interval", type=int, default=10, 
                        help="Save checkpoint every N episodes")
    parser.add_argument("--eval_interval", type=int, default=20, 
                        help="Evaluate agent every N episodes")
    
    # Model settings
    parser.add_argument("--use_dueling", action="store_true", default=True,
                        help="Use Dueling DQN architecture")
    parser.add_argument("--use_double", action="store_true", default=True,
                        help="Use Double DQN algorithm")
    parser.add_argument("--use_per", action="store_true", default=True,
                        help="Use Prioritized Experience Replay")
    parser.add_argument("--frame_stack", type=int, default=4,
                        help="Number of frames to stack for state representation")
    parser.add_argument("--memory_efficient", action="store_true", default=True,
                        help="Use memory-efficient implementations")
    parser.add_argument("--adaptive", action="store_true", default=True,
                        help="Use adaptive hyperparameter tuning")
    
    # Checkpoint settings
    parser.add_argument("--load_checkpoint", type=str, default=None, 
                        help="Path to checkpoint file to load")
    parser.add_argument("--model_dir", type=str, default="models",
                        help="Directory to save trained model")
    
    # Hyperparameters
    parser.add_argument("--learning_rate", type=float, default=0.0005, 
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
    parser.add_argument("--epsilon_end", type=float, default=0.05,
                        help="Minimum value of epsilon")
    parser.add_argument("--epsilon_decay", type=float, default=0.995,
                        help="Decay rate of epsilon per episode")
    
    # Debug settings
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode with extra logging and visualizations")
    parser.add_argument("--detailed_monitoring", action="store_true",
                        help="Enable detailed state monitoring with visualizations")
    parser.add_argument("--skip_browser", action="store_true",
                        help="Skip browser initialization (for debugging only)")
    
    # Parse arguments
    args = parser.parse_args()
    
    return args

def create_environment(args):
    """
    Create and initialize the game environment
    
    Args:
        args: Command line arguments
        
    Returns:
        Environment instance
    """
    logger.info("Creating environment...")
    
    # Create directories for debug output
    os.makedirs("debug_images", exist_ok=True)
    os.makedirs("debug_images/states", exist_ok=True)
    
    # Create environment
    try:
        if args.skip_browser:
            # Create a dummy environment for debugging
            from unittest.mock import MagicMock
            env = MagicMock()
            env.actions = ['noop', 'up', 'down', 'left', 'right']
            env.reset.return_value = np.zeros((84, 84), dtype=np.float32)
            env.step.return_value = (
                np.zeros((84, 84), dtype=np.float32),
                1.0,  # reward
                False,  # done
                {'score': 100, 'coins': 1}  # info
            )
            logger.warning("Using dummy environment (browser skipped)")
        else:
            # Create real environment
            env = SubwaySurfersEnv(
                game_url=args.game_url,
                position=args.browser_position,
                use_existing_browser=args.use_existing_browser
            )
            
        # Wrap environment with frame stacking
        stacked_env = FrameStackingEnv(env, stack_size=args.frame_stack)
        
        logger.info(f"Environment created successfully with {args.frame_stack} frame stacking")
        return stacked_env
    
    except Exception as e:
        logger.error(f"Error creating environment: {str(e)}")
        raise

def create_agent(env, args, device):
    """
    Create and initialize the DQN agent
    
    Args:
        env: Game environment
        args: Command line arguments
        device: Torch device (CPU/GPU)
        
    Returns:
        Tuple of (agent, start_episode)
    """
    logger.info("Creating agent...")
    
    # Determine state shape (frames, height, width)
    state_shape = (args.frame_stack, 84, 84)
    n_actions = len(env.actions)  # Number of possible actions
    
    # Optimize hyperparameters for GPU if available
    if device.type == 'cuda':
        batch_size, memory_capacity = optimize_for_gpu(
            batch_size=args.batch_size,
            state_shape=state_shape
        )
    else:
        batch_size = args.batch_size
        memory_capacity = args.memory_capacity
    
    logger.info(f"Using batch size: {batch_size}, memory capacity: {memory_capacity}")
    
    # Create agent
    try:
        # Create checkpoint directory
        os.makedirs("logs/checkpoints", exist_ok=True)
        
        # Use adaptive agent if requested
        if args.adaptive:
            logger.info("Creating AdaptiveDQNAgent with dynamic hyperparameter tuning")
            agent = AdaptiveDQNAgent(
                state_shape=state_shape,
                n_actions=n_actions,
                checkpoint_dir="./logs/checkpoints/",
                use_dueling=args.use_dueling,
                use_double=args.use_double,
                use_per=args.use_per,
                device=device,
                memory_efficient=args.memory_efficient
            )
        else:
            logger.info("Creating standard DQNAgent")
            agent = DQNAgent(
                state_shape=state_shape,
                n_actions=n_actions,
                checkpoint_dir="./logs/checkpoints/",
                use_dueling=args.use_dueling,
                use_double=args.use_double,
                use_per=args.use_per,
                device=device,
                memory_efficient=args.memory_efficient
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

def checkpoint_callback(checkpoint_path, episode):
    """
    Callback function for when a checkpoint is saved
    Copies the checkpoint to the models directory for easier access
    
    Args:
        checkpoint_path: Path to the saved checkpoint
        episode: Current episode number
    """
    try:
        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
        
        # Create a simplified filename for the destination
        dest_filename = f"subway_surfers_ep{episode}.pt"
        dest_path = os.path.join("models", dest_filename)
        
        # Copy the checkpoint
        shutil.copy(checkpoint_path, dest_path)
        logger.info(f"Checkpoint copied to {dest_path}")
        
        # Create a 'latest' link
        latest_path = os.path.join("models", "latest.pt")
        if os.path.exists(latest_path):
            os.remove(latest_path)
        shutil.copy(checkpoint_path, latest_path)
        
    except Exception as e:
        logger.warning(f"Error in checkpoint callback: {e}")

def save_experiment_config(args, model_dir):
    """
    Save experiment configuration to JSON file
    
    Args:
        args: Command line arguments
        model_dir: Directory to save the configuration
    """
    config = vars(args)
    config['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    config['platform'] = {
        'python_version': platform.python_version(),
        'system': platform.system(),
        'processor': platform.processor()
    }
    
    # Add GPU info if available
    if torch.cuda.is_available():
        config['gpu'] = {
            'name': torch.cuda.get_device_name(0),
            'count': torch.cuda.device_count(),
            'memory': torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        }
    
    # Save to file
    os.makedirs(model_dir, exist_ok=True)
    config_path = os.path.join(model_dir, 'experiment_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Experiment configuration saved to {config_path}")

def show_startup_message():
    """Display an ASCII art startup message"""
    art = """
    _____       _                         _____             __                
   / ___/__ __ (_)__    ___ _  ___  __ __/ ___/__ __ _____ / _|___ ____ ___  
  _\\ \\  / // // / _ \\  / _ `/ / _ \\/ // /\\ \\  / // // __// _// -_) __// -_) 
 /___/  \\_,_//_/_//_/  \\_,_/ / .__/\\_,_/___/  \\_,_//_/  /_/  \\__/_/   \\__/  
                            /_/                                             
     _____     _        __                                          __ 
    / ___/_ __/ /  ___ / /  ___ _  ___  ___ ___ ___ _ __ ___  ___ _/ /_
   / (_ // __/ _ \\/ -_) _ \\/ _ `/ / _ \\/ -_|_-</ -_) // / _ \\/ _ `/ __/
   \\___//_/ /_//_/\\__/_//_/\\_,_/ / .__/\\__/___/\\__/\\_,_/ .__/\\_,_/\\__/ 
                                /_/                    /_/              
    """
    
    print("\n" + art)
    
    # Add system info
    print("\nSystem Information:")
    print(f"  Python: {platform.python_version()}")
    print(f"  OS: {platform.system()} {platform.release()}")
    
    # Add PyTorch and CUDA info
    print("\nPyTorch Information:")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        
    # Memory info
    vm = psutil.virtual_memory()
    print("\nMemory Information:")
    print(f"  Total RAM: {vm.total / (1024**3):.1f} GB")
    print(f"  Available RAM: {vm.available / (1024**3):.1f} GB")
    
    # Add separator
    print("\n" + "="*80 + "\n")

def main():
    """Main function"""
    # Parse command line arguments
    args = parse_args()
    
    # Display welcome message
    show_startup_message()
    
    # Create necessary directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("logs/checkpoints", exist_ok=True)
    os.makedirs("debug_images", exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Log all arguments
    logger.info(f"Training with arguments: {vars(args)}")
    
    # Save experiment configuration
    import platform  # for system info
    save_experiment_config(args, args.model_dir)
    
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
            detailed_monitoring=args.detailed_monitoring,
            checkpoint_callback=checkpoint_callback  # Register callback for checkpoints
        )
        
        # Train the agent
        logger.info(f"Starting training from episode {start_episode}...")
        print("\nStarting training. Press Ctrl+C to stop gracefully.\n")
        
        # Train the agent
        rewards = trainer.train(start_episode=start_episode)
        
        # Calculate total training time
        training_time = time.time() - start_time
        hours, remainder = divmod(training_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        logger.info(f"Training completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
        
        # Create training summary
        summary_path = trainer.create_training_summary()
        print(f"\nTraining summary created: {summary_path}\n")
        
        # Copy final model to the specified model directory
        final_model_path = os.path.join(args.model_dir, f"subway_surfers_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt")
        
        # Find latest checkpoint
        latest_checkpoint = None
        checkpoints_dir = "logs/checkpoints"
        if os.path.exists(checkpoints_dir):
            checkpoints = [os.path.join(checkpoints_dir, f) for f in os.listdir(checkpoints_dir) if f.endswith(".pt")]
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=os.path.getmtime)
        
        # Copy the checkpoint if found
        if latest_checkpoint:
            shutil.copy(latest_checkpoint, final_model_path)
            logger.info(f"Final model saved to {final_model_path}")
        
        # Clean up
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
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
        
        # Display final message
        print("\n" + "="*80)
        print("Training session ended".center(80))
        
        # Calculate training time
        training_time = time.time() - start_time
        training_hours, remainder = divmod(training_time, 3600)
        training_minutes, training_seconds = divmod(remainder, 60)
        
        print(f"Training time: {int(training_hours)}h {int(training_minutes)}m {int(training_seconds)}s".center(80))
        
        # Show info about where to find results
        print("\nResults saved in:")
        print(f"  - Checkpoints: {os.path.abspath('logs/checkpoints')}")
        print(f"  - Training logs: {os.path.abspath('logs')}")
        print(f"  - Final model: {os.path.abspath(args.model_dir)}")
        print(f"  - Debug images: {os.path.abspath('debug_images')}")
        print("="*80 + "\n")

if __name__ == "__main__":
    main()