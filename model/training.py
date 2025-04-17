# model/training.py
import torch
import numpy as np
import time
import os
import logging
from datetime import datetime
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm

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
                 detailed_monitoring=False):
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
        """
        self.agent = agent
        self.env = env
        self.log_dir = log_dir
        self.save_interval = save_interval
        self.eval_interval = eval_interval
        self.max_episodes = max_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.detailed_monitoring = detailed_monitoring
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Performance metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.eval_rewards = []
        self.losses = []
        self.current_episode = 0
        
        # Timers for profiling
        self.episode_times = []
        self.step_times = []
        self.action_times = []
        self.learn_times = []
        
        # CSV logger
        self.csv_path = os.path.join(log_dir, f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        self.init_csv_logger()
        
        # Create directories for debug output
        os.makedirs(os.path.join(log_dir, "plots"), exist_ok=True)
        os.makedirs(os.path.join(log_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(log_dir, "eval"), exist_ok=True)
        
        logger.info("Trainer initialized")
        logger.info(f"Max episodes: {max_episodes}")
        logger.info(f"Save interval: {save_interval} episodes")
        logger.info(f"Evaluation interval: {eval_interval} episodes")
    
    def init_csv_logger(self):
        """Initialize CSV log file with headers"""
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Episode', 'Reward', 'Length', 'Epsilon', 
                'Loss', 'Total Steps', 'Episode Time', 
                'Average Step Time', 'Average Learn Time', 'Timestamp'
            ])
    
    def log_episode(self, episode, reward, length, loss, episode_time, avg_step_time, avg_learn_time):
        """
        Log episode data to CSV
        
        Args:
            episode: Episode number
            reward: Total reward for the episode
            length: Number of steps in the episode
            loss: Average loss for the episode
            episode_time: Total time for the episode
            avg_step_time: Average time per step
            avg_learn_time: Average time for learning
        """
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                episode, reward, length, self.agent.epsilon,
                loss, self.agent.total_steps, episode_time, 
                avg_step_time, avg_learn_time,
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ])
    
    def plot_metrics(self):
        """Plot training metrics and save to log directory"""
        # Create figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot episode rewards
        axs[0, 0].plot(self.episode_rewards)
        axs[0, 0].set_title('Episode Rewards')
        axs[0, 0].set_xlabel('Episode')
        axs[0, 0].set_ylabel('Reward')
        
        # Add moving average if enough episodes
        if len(self.episode_rewards) >= 10:
            moving_avg = np.convolve(self.episode_rewards, np.ones(10)/10, mode='valid')
            axs[0, 0].plot(range(9, len(self.episode_rewards)), moving_avg, 'r-', label='10-Episode Average')
            axs[0, 0].legend()
        
        # Plot episode lengths
        axs[0, 1].plot(self.episode_lengths)
        axs[0, 1].set_title('Episode Lengths')
        axs[0, 1].set_xlabel('Episode')
        axs[0, 1].set_ylabel('Steps')
        
        # Add moving average if enough episodes
        if len(self.episode_lengths) >= 10:
            moving_avg = np.convolve(self.episode_lengths, np.ones(10)/10, mode='valid')
            axs[0, 1].plot(range(9, len(self.episode_lengths)), moving_avg, 'r-', label='10-Episode Average')
            axs[0, 1].legend()
        
        # Plot epsilon decay
        episodes = list(range(len(self.episode_rewards)))
        epsilons = [self.agent.epsilon_start * (self.agent.epsilon_decay ** i) for i in episodes]
        epsilons = [max(e, self.agent.epsilon_end) for e in epsilons]
        axs[1, 0].plot(episodes, epsilons)
        axs[1, 0].set_title('Epsilon Decay')
        axs[1, 0].set_xlabel('Episode')
        axs[1, 0].set_ylabel('Epsilon')
        
        # Plot losses
        if self.losses:
            axs[1, 1].plot(self.losses)
            axs[1, 1].set_title('Training Loss')
            axs[1, 1].set_xlabel('Episode')
            axs[1, 1].set_ylabel('Loss')
            
            # Use log scale if losses vary widely
            if max(self.losses) / (min(self.losses) + 1e-10) > 100:
                axs[1, 1].set_yscale('log')
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, f"plots/training_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"))
        plt.close()
        
        # Also plot time metrics
        if self.episode_times:
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            
            # Plot episode times
            axs[0].plot(self.episode_times)
            axs[0].set_title('Episode Times')
            axs[0].set_xlabel('Episode')
            axs[0].set_ylabel('Time (s)')
            
            # Plot step times
            axs[1].plot([np.mean(times) for times in self.step_times])
            axs[1].set_title('Average Step Times')
            axs[1].set_xlabel('Episode')
            axs[1].set_ylabel('Time (ms)')
            
            # Plot learn times
            axs[2].plot([np.mean(times) for times in self.learn_times])
            axs[2].set_title('Average Learn Times')
            axs[2].set_xlabel('Episode')
            axs[2].set_ylabel('Time (ms)')
            
            # Adjust layout and save
            plt.tight_layout()
            plt.savefig(os.path.join(self.log_dir, f"plots/time_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"))
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
        
        try:
            for episode in range(start_episode, self.max_episodes):
                self.current_episode = episode
                
                # Reset environment and get initial state
                state = self.env.reset()
                
                # Initialize episode metrics
                episode_reward = 0
                episode_loss = []
                episode_start_time = time.time()
                episode_step_times = []
                episode_learn_times = []
                
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
                
                # Close progress bar
                pbar.close()
                
                # Calculate episode time
                episode_time = time.time() - episode_start_time
                
                # Update environment with training stats (for GUI display)
                if hasattr(self.env, 'update_training_stats'):
                    self.env.update_training_stats(
                        epsilon=self.agent.epsilon,
                        loss=np.mean(episode_loss) if episode_loss else 0,
                        avg_reward=np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else episode_reward
                    )
                
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
                
                # Record timing metrics
                self.episode_times.append(episode_time)
                self.step_times.append(episode_step_times)
                self.learn_times.append(episode_learn_times if episode_learn_times else [0])
                
                # Update agent's episode reward history
                self.agent.add_episode_reward(episode_reward)
                
                # Calculate average times
                avg_step_time = np.mean(episode_step_times) if episode_step_times else 0
                avg_learn_time = np.mean(episode_learn_times) if episode_learn_times else 0
                
                # Log episode data
                logger.info(f"Episode {episode} - Reward: {episode_reward:.2f}, Steps: {step+1}, Epsilon: {self.agent.epsilon:.4f}, Loss: {avg_loss:.4f}, Time: {episode_time:.2f}s")
                self.log_episode(episode, episode_reward, step+1, avg_loss, episode_time, avg_step_time, avg_learn_time)
                
                # Save checkpoint periodically
                if episode % self.save_interval == 0:
                    self.agent.save_checkpoint(
                        episode=episode,
                        rewards=self.episode_rewards[-self.save_interval:],
                        avg_reward=np.mean(self.episode_rewards[-self.save_interval:])
                    )
                
                # Run evaluation periodically
                if episode % self.eval_interval == 0:
                    eval_reward = self.evaluate()
                    self.eval_rewards.append(eval_reward)
                    logger.info(f"Evaluation at episode {episode} - Reward: {eval_reward:.2f}")
                
                # Plot metrics periodically
                if episode % 20 == 0:
                    self.plot_metrics()
                    
                    # Also log performance statistics
                    if hasattr(self.agent, 'log_performance_stats'):
                        self.agent.log_performance_stats()
                    
                    if hasattr(self.agent.memory, 'log_performance_stats'):
                        self.agent.memory.log_performance_stats()
                    
                    if hasattr(self.agent.policy_net, 'log_performance_stats'):
                        self.agent.policy_net.log_performance_stats()
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training error: {str(e)}", exc_info=True)
        finally:
            # Final evaluation
            final_eval_reward = self.evaluate()
            logger.info(f"Final evaluation - Reward: {final_eval_reward:.2f}")
            
            # Save final model
            self.agent.save_checkpoint(
                episode=episode,
                rewards=self.episode_rewards[-self.save_interval:],
                avg_reward=np.mean(self.episode_rewards[-self.save_interval:]),
                suffix="final"
            )
            
            # Plot final metrics
            self.plot_metrics()
            
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
            with open(eval_log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Episode', 'Average Reward', 'Average Length', 'Timestamp'])
        
        # Append evaluation data
        with open(eval_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([self.current_episode, avg_reward, avg_length, eval_data['timestamp']])
        
        return avg_reward
    
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
                'mean': np.mean([np.mean(times) for times in self.learn_times]) if self.learn_times else 0,
                'std': np.std([np.mean(times) for times in self.learn_times]) if self.learn_times else 0,
                'min': np.min([np.mean(times) for times in self.learn_times]) if self.learn_times else 0,
                'max': np.max([np.mean(times) for times in self.learn_times]) if self.learn_times else 0
            },
            'episodes_completed': len(self.episode_rewards),
            'total_steps': sum(self.episode_lengths),
            'total_training_time': sum(self.episode_times) if self.episode_times else 0
        }
        
        return stats
    
    def log_performance_stats(self):
        """Log performance statistics to the log file"""
        stats = self.get_performance_stats()
        
        logger.info("Training Performance Statistics:")
        logger.info(f"  Episodes Completed: {stats['episodes_completed']}")
        logger.info(f"  Total Steps: {stats['total_steps']}")
        logger.info(f"  Total Training Time: {stats['total_training_time']:.2f}s")
        logger.info(f"  Average Episode Time: {stats['episode_time']['mean']:.2f}s (±{stats['episode_time']['std']:.2f}s)")
        logger.info(f"  Average Step Time: {stats['step_time']['mean']:.2f}ms (±{stats['step_time']['std']:.2f}ms)")
        logger.info(f"  Average Learn Time: {stats['learn_time']['mean']:.2f}ms (±{stats['learn_time']['std']:.2f}ms)")