# model/training.py
import torch
import numpy as np
import time
import os
import logging
import gc
from datetime import datetime
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt
import psutil
import signal
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Training")

class Trainer:
    def __init__(self, agent, env, log_dir="./logs/", save_interval=10, 
                 eval_interval=5, max_episodes=1000, max_steps_per_episode=10000, 
                 detailed_monitoring=False, checkpoint_callback=None):
        """
        Initialize the trainer
        
        Args:
            agent: DQNAgent instance
            env: Game environment
            log_dir: Directory to save logs
            save_interval: Number of episodes between saving checkpoints
            eval_interval: Number of episodes between evaluations
            max_episodes: Maximum number of episodes to train for
            max_steps_per_episode: Maximum steps per episode
            detailed_monitoring: Whether to save detailed state visualizations
            checkpoint_callback: Optional callback function to receive checkpoint path when saved
        """
        self.agent = agent
        self.env = env
        self.log_dir = log_dir
        self.save_interval = save_interval
        self.eval_interval = eval_interval
        self.max_episodes = max_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.detailed_monitoring = detailed_monitoring
        self.checkpoint_callback = checkpoint_callback
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Performance metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.eval_rewards = []
        self.losses = []
        self.current_episode = 0
        
        # Reward components tracking
        self.survival_rewards = []
        self.score_rewards = []
        self.coin_rewards = []
        self.penalty_rewards = []
        self.time_rewards = []
        
        # Timers for profiling
        self.episode_times = []
        self.step_times = []
        self.action_times = []
        self.learn_times = []
        
        # CSV logger
        self.csv_path = os.path.join(log_dir, f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        self.init_csv_logger()
        
        # Create directories for debug output
        os.makedirs(os.path.join(log_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(log_dir, "eval"), exist_ok=True)
        os.makedirs(os.path.join(log_dir, "plots"), exist_ok=True)
        
        # Memory monitoring
        self.memory_usage = []
        
        # Recovery state
        self.is_graceful_exit = False
        self.recovery_checkpoint = None
        self.recovery_episode = 0
        
        # Register signal handlers for graceful shutdown
        self._register_signal_handlers()
        
        logger.info("Trainer initialized")
        logger.info(f"Max episodes: {max_episodes}")
        logger.info(f"Save interval: {save_interval} episodes")
        logger.info(f"Evaluation interval: {eval_interval} episodes")
    
    def _register_signal_handlers(self):
        """Register signal handlers for graceful shutdown"""
        # Register SIGINT (Ctrl+C) and SIGTERM handler
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, sig, frame):
        """Handle signals for graceful shutdown"""
        signal_name = "SIGINT" if sig == signal.SIGINT else "SIGTERM"
        logger.info(f"Received {signal_name} signal - initiating graceful shutdown")
        self.is_graceful_exit = True
        
        # The training loop will check this flag and save a checkpoint
    
    def init_csv_logger(self):
        """Initialize CSV log file with headers"""
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Episode', 'Reward', 'Length', 'Epsilon', 'Loss', 'Total Steps',
                'Episode Time', 'Average Step Time', 'Average Learn Time',
                'Survival Reward', 'Score Reward', 'Coin Reward', 'Penalty Reward',
                'Time Reward', 'Memory Usage (MB)', 'GPU Memory (MB)',
                'Timestamp'
            ])
    
    def log_episode(self, episode, reward, length, losses, reward_components, episode_time, avg_step_time, avg_learn_time, memory_usage=None, gpu_memory=None):
        """
        Log episode data to CSV
        
        Args:
            episode: Episode number
            reward: Total reward for the episode
            length: Number of steps in the episode
            losses: List of losses for the episode
            reward_components: Dictionary of reward components
            episode_time: Total time for the episode
            avg_step_time: Average time per step
            avg_learn_time: Average time for learning
            memory_usage: RAM usage in MB (optional)
            gpu_memory: GPU memory usage in MB (optional)
        """
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            
            # Calculate average loss if available
            avg_loss = np.mean(losses) if losses else 0
            
            # Extract reward components
            survival_reward = reward_components.get('survival_reward', 0)
            score_reward = reward_components.get('score_reward', 0)
            coin_reward = reward_components.get('coin_reward', 0)
            penalty_reward = reward_components.get('game_over_penalty', 0)
            time_reward = reward_components.get('time_reward', 0)
            
            writer.writerow([
                episode, reward, length, self.agent.epsilon, avg_loss, self.agent.total_steps,
                episode_time, avg_step_time, avg_learn_time,
                survival_reward, score_reward, coin_reward, penalty_reward,
                time_reward, memory_usage, gpu_memory,
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ])
    
    def get_memory_usage(self):
        """Get current memory usage"""
        mem_info = {
            'ram': psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)  # MB
        }
        
        # Add GPU memory if available
        if torch.cuda.is_available():
            mem_info['gpu_allocated'] = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
            mem_info['gpu_reserved'] = torch.cuda.memory_reserved() / (1024 * 1024)  # MB
        
        return mem_info
    
    def plot_memory_usage(self, save_dir=None):
        """
        Plot memory usage over time
        
        Args:
            save_dir: Directory to save the plot (optional)
        """
        if not self.memory_usage:
            return
            
        if save_dir is None:
            save_dir = os.path.join(self.log_dir, "plots")
            
        os.makedirs(save_dir, exist_ok=True)
        
        # Extract data
        episodes = range(len(self.memory_usage))
        ram_usage = [mem['ram'] for mem in self.memory_usage]
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Plot RAM usage
        plt.plot(episodes, ram_usage, label='RAM Usage (MB)')
        
        # Plot GPU memory if available
        if 'gpu_allocated' in self.memory_usage[0]:
            gpu_allocated = [mem['gpu_allocated'] for mem in self.memory_usage]
            gpu_reserved = [mem['gpu_reserved'] for mem in self.memory_usage]
            
            plt.plot(episodes, gpu_allocated, label='GPU Allocated (MB)')
            plt.plot(episodes, gpu_reserved, label='GPU Reserved (MB)')
        
        plt.title('Memory Usage During Training')
        plt.xlabel('Episode')
        plt.ylabel('Memory (MB)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save and close
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(os.path.join(save_dir, f"memory_usage_{timestamp}.png"))
        plt.close()
    
    def plot_reward_components(self, save_dir=None):
        """
        Plot reward components over training
        
        Args:
            save_dir: Directory to save the plot (optional)
        """
        if not self.survival_rewards:
            return
            
        if save_dir is None:
            save_dir = os.path.join(self.log_dir, "plots")
            
        os.makedirs(save_dir, exist_ok=True)
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Prepare data
        episodes = range(1, len(self.survival_rewards) + 1)
        survival = np.array(self.survival_rewards)
        score = np.array(self.score_rewards)
        coins = np.array(self.coin_rewards)
        penalties = np.abs(np.array(self.penalty_rewards))  # Make positive for stacking
        time_rewards = np.array(self.time_rewards) if self.time_rewards else np.zeros_like(survival)
        
        # Create stacked area plot
        plt.stackplot(
            episodes,
            [survival, score, coins, time_rewards, penalties],
            labels=['Survival', 'Score', 'Coins', 'Time', 'Penalties'],
            alpha=0.7
        )
        
        plt.title('Reward Components')
        plt.xlabel('Episode')
        plt.ylabel('Reward Magnitude')
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Save and close
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(os.path.join(save_dir, f"reward_components_{timestamp}.png"))
        plt.close()
    
    def train(self, start_episode=0):
        """
        Main training loop
        
        Args:
            start_episode: Episode to start from (for resuming training)
            
        Returns:
            List of episode rewards
        """
        logger.info(f"Starting training from episode {start_episode}")
        self.current_episode = start_episode
        
        # For storing last reward components
        last_reward_components = {}
        
        try:
            for episode in range(start_episode, self.max_episodes):
                # Break if graceful exit was requested
                if self.is_graceful_exit:
                    # Mark this episode as the recovery point
                    self.recovery_episode = episode
                    break
                    
                self.current_episode = episode
                
                # Reset environment and get initial state
                state = self.env.reset()
                
                # Initialize episode metrics
                episode_reward = 0
                episode_loss = []
                episode_start_time = time.time()
                episode_step_times = []
                episode_learn_times = []
                
                # Reset reward components for this episode
                episode_reward_components = {
                    'survival_reward': 0,
                    'score_reward': 0,
                    'coin_reward': 0,
                    'game_over_penalty': 0,
                    'time_reward': 0
                }
                
                # Progress bar for this episode
                pbar = tqdm(total=self.max_steps_per_episode, desc=f"Episode {episode}")
                
                # Episode loop
                for step in range(self.max_steps_per_episode):
                    step_start_time = time.time()
                    
                    # Select and perform action
                    action = self.agent.select_action(state)
                    
                    # Take action in environment
                    next_state, reward, done, info = self.env.step(action)
                    
                    # Store transition in replay memory
                    self.agent.remember(state, action, reward, next_state, done)
                    
                    # Move to next state
                    state = next_state
                    
                    # Visualize state periodically if detailed monitoring is enabled
                    if self.detailed_monitoring and episode % 5 == 0 and step % 50 == 0 and step > 0:
                        self.env.visualize_agent_state(state, step, episode)
                        
                    # Accumulate rewards
                    episode_reward += reward
                    
                    # Update reward components
                    for component in ['survival_reward', 'score_reward', 'coin_reward', 'game_over_penalty', 'time_reward']:
                        if component in info:
                            episode_reward_components[component] += info[component]
                    
                    # Learn from experiences
                    learn_start_time = time.time()
                    loss = self.agent.learn()
                    learn_time = time.time() - learn_start_time
                    
                    if loss is not None:
                        episode_loss.append(loss)
                        episode_learn_times.append(learn_time * 1000)  # Convert to ms
                    
                    # Record step time
                    step_time = time.time() - step_start_time
                    episode_step_times.append(step_time * 1000)  # Convert to ms
                    
                    # Update progress bar
                    pbar.update(1)
                    pbar.set_postfix({
                        'reward': f"{episode_reward:.2f}",
                        'epsilon': f"{self.agent.epsilon:.2f}",
                        'loss': f"{np.mean(episode_loss) if episode_loss else 0:.4f}"
                    })
                    
                    # Check if episode is done
                    if done:
                        break
                
                # Additional safety check in case env.step never returns done=True
                if step >= self.max_steps_per_episode - 1:
                    logger.warning(f"Episode {episode} reached max steps without completing")
                
                # Close progress bar
                pbar.close()
                
                # Calculate episode time
                episode_time = time.time() - episode_start_time
                
                # Update environment with training stats
                if hasattr(self.env, 'update_training_stats'):
                    self.env.update_training_stats(
                        epsilon=self.agent.epsilon,
                        loss=np.mean(episode_loss) if episode_loss else 0,
                        avg_reward=np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else episode_reward
                    )
                
                # Adaptively adjust hyperparameters if agent supports it
                if hasattr(self.agent, 'adapt_hyperparameters'):
                    self.agent.adapt_hyperparameters()
                
                # Update target network periodically
                if episode % self.agent.target_update == 0:
                    self.agent.update_target_network()
                
                # Decay exploration rate
                self.agent.decay_epsilon()
                
                # Record metrics
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(step + 1)
                if episode_loss:
                    avg_loss = np.mean(episode_loss)
                    self.losses.append(avg_loss)
                else:
                    avg_loss = 0
                
                # Record reward components
                self.survival_rewards.append(episode_reward_components['survival_reward'])
                self.score_rewards.append(episode_reward_components['score_reward'])
                self.coin_rewards.append(episode_reward_components['coin_reward'])
                self.penalty_rewards.append(episode_reward_components['game_over_penalty'])
                self.time_rewards.append(episode_reward_components['time_reward'])
                
                # Store reward components for reference
                last_reward_components = episode_reward_components
                
                # Record timing metrics
                self.episode_times.append(episode_time)
                self.step_times.append(episode_step_times)
                self.learn_times.append(episode_learn_times if episode_learn_times else [0])
                
                # Get memory usage
                memory_info = self.get_memory_usage()
                self.memory_usage.append(memory_info)
                
                # Update agent's episode reward history
                self.agent.add_episode_reward(episode_reward, score=info.get('score', None))
                
                # Calculate average times
                avg_step_time = np.mean(episode_step_times) if episode_step_times else 0
                avg_learn_time = np.mean(episode_learn_times) if episode_learn_times else 0
                
                # Log episode data
                logger.info(f"Episode {episode} - Reward: {episode_reward:.2f}, Steps: {step+1}, Epsilon: {self.agent.epsilon:.4f}, Loss: {avg_loss:.4f}, Time: {episode_time:.2f}s")
                self.log_episode(
                    episode=episode, 
                    reward=episode_reward, 
                    length=step+1, 
                    losses=episode_loss, 
                    reward_components=episode_reward_components,
                    episode_time=episode_time, 
                    avg_step_time=avg_step_time, 
                    avg_learn_time=avg_learn_time,
                    memory_usage=memory_info.get('ram'),
                    gpu_memory=memory_info.get('gpu_allocated')
                )
                
                # Save checkpoint periodically
                if episode % self.save_interval == 0 or episode == self.max_episodes - 1:
                    checkpoint_path = self.agent.save_checkpoint(
                        episode=episode,
                        rewards=self.episode_rewards[-self.save_interval:] if len(self.episode_rewards) >= self.save_interval else self.episode_rewards,
                        avg_reward=np.mean(self.episode_rewards[-self.save_interval:]) if len(self.episode_rewards) >= self.save_interval else np.mean(self.episode_rewards) if self.episode_rewards else 0
                    )
                    
                    # If checkpoint callback is provided, call it
                    if self.checkpoint_callback:
                        self.checkpoint_callback(checkpoint_path, episode)
                
                # Run evaluation periodically
                if episode % self.eval_interval == 0 and episode > 0:
                    eval_reward = self.evaluate()
                    self.eval_rewards.append(eval_reward)
                    logger.info(f"Evaluation at episode {episode} - Reward: {eval_reward:.2f}")
                    
                # Log performance statistics periodically
                if episode % 20 == 0:
                    if hasattr(self.agent, 'log_performance_stats'):
                        self.agent.log_performance_stats()
                    
                    if hasattr(self.env, 'log_performance_stats'):
                        self.env.log_performance_stats()
                    
                    if hasattr(self.agent.memory, 'log_performance_stats'):
                        self.agent.memory.log_performance_stats()
                
                # Plot metrics periodically
                if episode % 50 == 0 and episode > 0:
                    self.plot_training_metrics()
                    self.plot_reward_components()
                    self.plot_memory_usage()
                    
                    # Force garbage collection
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training error: {str(e)}", exc_info=True)
            
            # Create error report
            error_report = {
                'error': str(e),
                'traceback': traceback.format_exc(),
                'episode': self.current_episode,
                'total_steps': self.agent.total_steps,
                'last_reward_components': last_reward_components
            }
            
            # Save error report
            os.makedirs(os.path.join(self.log_dir, "errors"), exist_ok=True)
            error_file = os.path.join(self.log_dir, "errors", f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
            with open(error_file, 'w') as f:
                f.write(f"Error: {str(e)}\n\n")
                f.write(f"Traceback:\n{traceback.format_exc()}\n\n")
                f.write(f"Episode: {self.current_episode}\n")
                f.write(f"Total Steps: {self.agent.total_steps}\n")
                f.write(f"Last Reward Components:\n")
                for k, v in last_reward_components.items():
                    f.write(f"  {k}: {v}\n")
            
            logger.info(f"Error report saved to {error_file}")
        finally:
            # Final actions regardless of how training ended
            
            # Save final checkpoint if we didn't exit due to keyboard interrupt
            if not self.is_graceful_exit:
                try:
                    self.recovery_checkpoint = self.agent.save_checkpoint(
                        episode=self.current_episode,
                        rewards=self.episode_rewards[-min(10, len(self.episode_rewards)):] if self.episode_rewards else [],
                        avg_reward=np.mean(self.episode_rewards[-min(10, len(self.episode_rewards)):]) if self.episode_rewards else 0,
                        suffix="recovery"
                    )
                    logger.info(f"Recovery checkpoint saved: {self.recovery_checkpoint}")
                    
                    # If checkpoint callback is provided, call it
                    if self.checkpoint_callback:
                        self.checkpoint_callback(self.recovery_checkpoint, self.current_episode)
                except Exception as e:
                    logger.error(f"Failed to save recovery checkpoint: {str(e)}")
            
            # Final evaluation
            try:
                final_eval_reward = self.evaluate()
                logger.info(f"Final evaluation - Reward: {final_eval_reward:.2f}")
            except Exception as e:
                logger.error(f"Error during final evaluation: {str(e)}")
            
            # Final plots
            try:
                self.plot_training_metrics()
                self.plot_reward_components()
                self.plot_memory_usage()
            except Exception as e:
                logger.error(f"Error creating final plots: {str(e)}")
            
            return self.episode_rewards
    
    def evaluate(self, num_episodes=5):
        """
        Evaluate the agent without exploration
        
        Args:
            num_episodes: Number of episodes to evaluate
            
        Returns:
            Average reward over evaluation episodes
        """
        logger.info(f"Evaluating agent for {num_episodes} episodes")
        eval_rewards = []
        eval_lengths = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            steps = 0
            done = False
            
            while not done:
                # Select best action without exploration
                action = self.agent.select_action(state, training=False)
                
                # Take action in environment
                next_state, reward, done, info = self.env.step(action)
                
                # Move to next state
                state = next_state
                
                # Accumulate rewards
                episode_reward += reward
                steps += 1
                
                # Safety check - limit evaluation steps
                if steps >= self.max_steps_per_episode:
                    logger.warning(f"Evaluation episode reached max steps ({self.max_steps_per_episode})")
                    break
            
            eval_rewards.append(episode_reward)
            eval_lengths.append(steps)
            logger.debug(f"Evaluation episode {episode} - Reward: {episode_reward:.2f}, Steps: {steps}")
        
        avg_reward = np.mean(eval_rewards)
        avg_length = np.mean(eval_lengths)
        logger.info(f"Evaluation complete - Average reward: {avg_reward:.2f}, Average length: {avg_length:.2f}")
        
        # Save evaluation metrics
        eval_data = {
            'episode': self.current_episode,
            'rewards': eval_rewards,
            'avg_reward': avg_reward,
            'lengths': eval_lengths,
            'avg_length': avg_length,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Append to evaluation log
        eval_log_path = os.path.join(self.log_dir, "eval/eval_log.csv")
        
        # Create file with header if it doesn't exist
        if not os.path.exists(eval_log_path):
            os.makedirs(os.path.dirname(eval_log_path), exist_ok=True)
            with open(eval_log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Episode', 'Average Reward', 'Average Length', 'Timestamp'])
        
        # Append evaluation data
        with open(eval_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([self.current_episode, avg_reward, avg_length, eval_data['timestamp']])
        
        return avg_reward
    
    def plot_training_metrics(self, save_dir=None):
        """
        Plot training metrics
        
        Args:
            save_dir: Directory to save plots (optional)
        """
        if not self.episode_rewards:
            return
            
        if save_dir is None:
            save_dir = os.path.join(self.log_dir, "plots")
            
        os.makedirs(save_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Plot rewards
        plt.figure(figsize=(10, 6))
        plt.plot(self.episode_rewards, alpha=0.6, label='Episode Reward')
        
        if len(self.episode_rewards) >= 10:
            window = 10
            rolling_mean = np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid')
            plt.plot(range(window-1, len(self.episode_rewards)), rolling_mean, 'r-', label=f'{window}-Episode Moving Average')
        
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Training Rewards')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, f"rewards_{timestamp}.png"))
        plt.close()
        
        # Plot episode lengths
        plt.figure(figsize=(10, 6))
        plt.plot(self.episode_lengths, alpha=0.6, label='Episode Length')
        
        if len(self.episode_lengths) >= 10:
            window = 10
            rolling_mean = np.convolve(self.episode_lengths, np.ones(window)/window, mode='valid')
            plt.plot(range(window-1, len(self.episode_lengths)), rolling_mean, 'r-', label=f'{window}-Episode Moving Average')
        
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.title('Episode Lengths')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, f"lengths_{timestamp}.png"))
        plt.close()
        
        # Plot epsilon decay
        if hasattr(self.agent, 'epsilon_history') and self.agent.epsilon_history:
            plt.figure(figsize=(10, 6))
            plt.plot(self.agent.epsilon_history)
            plt.xlabel('Episode')
            plt.ylabel('Epsilon')
            plt.title('Exploration Rate Decay')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(save_dir, f"epsilon_{timestamp}.png"))
            plt.close()
        
        # Plot combined metrics
        fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
        
        # Rewards
        axs[0].plot(self.episode_rewards, alpha=0.6)
        if len(self.episode_rewards) >= 10:
            window = 10
            rolling_mean = np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid')
            axs[0].plot(range(window-1, len(self.episode_rewards)), rolling_mean, 'r-', label=f'{window}-Episode Moving Average')
        axs[0].set_ylabel('Reward')
        axs[0].set_title('Training Rewards')
        axs[0].legend()
        axs[0].grid(True, alpha=0.3)
        
        # Lengths
        axs[1].plot(self.episode_lengths, alpha=0.6)
        if len(self.episode_lengths) >= 10:
            window = 10
            rolling_mean = np.convolve(self.episode_lengths, np.ones(window)/window, mode='valid')
            axs[1].plot(range(window-1, len(self.episode_lengths)), rolling_mean, 'r-', label=f'{window}-Episode Moving Average')
        axs[1].set_ylabel('Steps')
        axs[1].set_title('Episode Lengths')
        axs[1].legend()
        axs[1].grid(True, alpha=0.3)
        
        # Losses
        if self.losses:
            axs[2].plot(self.losses, alpha=0.6)
            if len(self.losses) >= 10:
                window = 10
                rolling_mean = np.convolve(self.losses, np.ones(window)/window, mode='valid')
                axs[2].plot(range(window-1, len(self.losses)), rolling_mean, 'r-', label=f'{window}-Episode Moving Average')
            axs[2].set_ylabel('Loss')
            axs[2].set_title('Training Loss')
            axs[2].legend()
            axs[2].grid(True, alpha=0.3)
            
            # Use log scale for better visualization if values span multiple orders of magnitude
            if max(self.losses) / (min(self.losses) + 1e-10) > 100:
                axs[2].set_yscale('log')
        
        plt.xlabel('Episode')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"metrics_combined_{timestamp}.png"))
        plt.close()
    
    def get_performance_stats(self):
        """Get performance statistics for the training process"""
        stats = {
            'episode_time': {
                'mean': np.mean(self.episode_times) if self.episode_times else 0,
                'std': np.std(self.episode_times) if self.episode_times else 0,
                'min': np.min(self.episode_times) if self.episode_times else 0,
                'max': np.max(self.episode_times) if self.episode_times else 0
            },
            'step_time': {
                'mean': np.mean([np.mean(times) for times in self.step_times]) if self.step_times else 0,
                'std': np.std([np.mean(times) for times in self.step_times]) if self.step_times else 0,
                'min': np.min([np.mean(times) for times in self.step_times]) if self.step_times else 0,
                'max': np.max([np.mean(times) for times in self.step_times]) if self.step_times else 0
            },
            'learn_time': {
                'mean': np.mean([np.mean(times) for times in self.learn_times if times]) if self.learn_times else 0,
                'std': np.std([np.mean(times) for times in self.learn_times if times]) if self.learn_times else 0,
                'min': np.min([np.mean(times) for times in self.learn_times if times]) if self.learn_times else 0,
                'max': np.max([np.mean(times) for times in self.learn_times if times]) if self.learn_times else 0
            },
            'episodes_completed': len(self.episode_rewards),
            'total_steps': sum(self.episode_lengths) if self.episode_lengths else 0,
            'total_training_time': sum(self.episode_times) if self.episode_times else 0,
            'reward_components': {
                'survival': np.mean(self.survival_rewards) if self.survival_rewards else 0,
                'score': np.mean(self.score_rewards) if self.score_rewards else 0,
                'coins': np.mean(self.coin_rewards) if self.coin_rewards else 0,
                'penalties': np.mean(self.penalty_rewards) if self.penalty_rewards else 0,
                'time': np.mean(self.time_rewards) if self.time_rewards else 0
            }
        }
        
        # Add memory stats if available
        if self.memory_usage:
            stats['memory'] = {
                'ram': {
                    'mean': np.mean([mem['ram'] for mem in self.memory_usage]),
                    'max': np.max([mem['ram'] for mem in self.memory_usage])
                }
            }
            
            if 'gpu_allocated' in self.memory_usage[0]:
                stats['memory']['gpu'] = {
                    'mean': np.mean([mem['gpu_allocated'] for mem in self.memory_usage]),
                    'max': np.max([mem['gpu_allocated'] for mem in self.memory_usage])
                }
        
        return stats
    
    def log_performance_stats(self):
        """Log performance statistics to the log file"""
        stats = self.get_performance_stats()
        
        logger.info("Training Performance Statistics:")
        logger.info(f"  Episodes Completed: {stats['episodes_completed']}")
        logger.info(f"  Total Steps: {stats['total_steps']}")
        logger.info(f"  Total Training Time: {stats['total_training_time']:.2f}s ({stats['total_training_time']/3600:.2f}h)")
        logger.info(f"  Average Episode Time: {stats['episode_time']['mean']:.2f}s (±{stats['episode_time']['std']:.2f}s)")
        logger.info(f"  Average Step Time: {stats['step_time']['mean']:.2f}ms (±{stats['step_time']['std']:.2f}ms)")
        logger.info(f"  Average Learn Time: {stats['learn_time']['mean']:.2f}ms (±{stats['learn_time']['std']:.2f}ms)")
        
        # Log reward components
        logger.info("  Average Reward Components:")
        logger.info(f"    Survival: {stats['reward_components']['survival']:.2f}")
        logger.info(f"    Score: {stats['reward_components']['score']:.2f}")
        logger.info(f"    Coins: {stats['reward_components']['coins']:.2f}")
        logger.info(f"    Penalties: {stats['reward_components']['penalties']:.2f}")
        if 'time' in stats['reward_components']:
            logger.info(f"    Time: {stats['reward_components']['time']:.2f}")
        
        # Log memory stats if available
        if 'memory' in stats:
            logger.info(f"  Average RAM Usage: {stats['memory']['ram']['mean']:.2f}MB (max: {stats['memory']['ram']['max']:.2f}MB)")
            if 'gpu' in stats['memory']:
                logger.info(f"  Average GPU Memory: {stats['memory']['gpu']['mean']:.2f}MB (max: {stats['memory']['gpu']['max']:.2f}MB)")
    
    def create_training_summary(self, save_path=None):
        """
        Create a comprehensive training summary report
        
        Args:
            save_path: Path to save the summary report (optional)
        """
        if not self.episode_rewards:
            logger.warning("No training data to summarize")
            return
            
        if save_path is None:
            os.makedirs(os.path.join(self.log_dir, "reports"), exist_ok=True)
            save_path = os.path.join(self.log_dir, "reports", f"training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        
        stats = self.get_performance_stats()
        
        # Format timedelta
        total_seconds = stats['total_training_time']
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        time_str = f"{hours}h {minutes}m {seconds}s"
        
        # Create summary report
        report = [
            "="*80,
            "SUBWAY SURFERS RL TRAINING SUMMARY".center(80),
            "="*80,
            "",
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Training Time: {time_str}",
            "",
            "TRAINING METRICS",
            "-"*80,
            f"Episodes Completed: {stats['episodes_completed']} of {self.max_episodes}",
            f"Total Environment Steps: {stats['total_steps']:,}",
            "",
            "PERFORMANCE",
            "-"*80,
        ]
        
        if self.episode_rewards:
            report.extend([
                f"Final Episode Reward: {self.episode_rewards[-1]:.2f}",
                f"Best Episode Reward: {max(self.episode_rewards):.2f}",
                f"Average Episode Reward: {np.mean(self.episode_rewards):.2f}",
                f"Last 10 Episodes Average: {np.mean(self.episode_rewards[-10:]):.2f}" if len(self.episode_rewards) >= 10 else "",
                "",
                "EPISODE LENGTHS",
                "-"*80,
                f"Average Episode Length: {np.mean(self.episode_lengths):.2f} steps",
                f"Longest Episode: {max(self.episode_lengths)} steps",
                f"Shortest Episode: {min(self.episode_lengths)} steps",
                "",
                "REWARD COMPONENTS",
                "-"*80,
                f"Survival Rewards: {np.mean(self.survival_rewards):.2f}",
                f"Score Rewards: {np.mean(self.score_rewards):.2f}",
                f"Coin Rewards: {np.mean(self.coin_rewards):.2f}",
                f"Penalty Rewards: {np.mean(self.penalty_rewards):.2f}",
            ])
            
            if self.time_rewards:
                report.append(f"Time Rewards: {np.mean(self.time_rewards):.2f}")
        
        report.extend([
            "",
            "TIMING METRICS",
            "-"*80,
            f"Average Episode Time: {stats['episode_time']['mean']:.2f}s (±{stats['episode_time']['std']:.2f}s)",
            f"Average Step Time: {stats['step_time']['mean']:.2f}ms (±{stats['step_time']['std']:.2f}ms)",
            f"Average Learn Time: {stats['learn_time']['mean']:.2f}ms (±{stats['learn_time']['std']:.2f}ms)",
            "",
            "MEMORY USAGE",
            "-"*80,
        ])
        
        if 'memory' in stats:
            report.extend([
                f"Average RAM Usage: {stats['memory']['ram']['mean']:.2f}MB (max: {stats['memory']['ram']['max']:.2f}MB)",
            ])
            
            if 'gpu' in stats['memory']:
                report.extend([
                    f"Average GPU Memory: {stats['memory']['gpu']['mean']:.2f}MB (max: {stats['memory']['gpu']['max']:.2f}MB)",
                ])
        
        # Add model info
        if hasattr(self.agent, 'policy_net'):
            policy_params = sum(p.numel() for p in self.agent.policy_net.parameters())
            report.extend([
                "",
                "MODEL INFORMATION",
                "-"*80,
                f"Model Parameters: {policy_params:,}",
                f"Final Epsilon: {self.agent.epsilon:.4f}",
                f"Using Dueling DQN: {'Yes' if hasattr(self.agent, 'use_dueling') and self.agent.use_dueling else 'No'}",
                f"Using Double DQN: {'Yes' if hasattr(self.agent, 'use_double') and self.agent.use_double else 'No'}",
                f"Using Prioritized Experience Replay: {'Yes' if hasattr(self.agent, 'use_per') and self.agent.use_per else 'No'}",
                f"Memory Efficiency Optimizations: {'Yes' if hasattr(self.agent, 'memory_efficient') and self.agent.memory_efficient else 'No'}",
            ])
        
        report.extend([
            "",
            "="*80,
        ])
        
        # Write report to file
        with open(save_path, 'w') as f:
            f.write('\n'.join(report))
        
        logger.info(f"Training summary saved to {save_path}")
        return save_path