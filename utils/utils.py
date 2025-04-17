# utils/utils.py
import torch
import logging
import time
import numpy as np
import os
import platform
import psutil
import matplotlib.pyplot as plt
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Utils")

def check_gpu():
    """
    Check if GPU is available and return the appropriate device.
    Also logs system information for diagnostics.
    
    Returns:
        torch.device: Device to use (cuda or cpu)
    """
    # Log system information
    log_system_info()
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        
        # Get GPU information
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # Convert to GB
        
        logger.info(f"GPU found: {gpu_name} with {gpu_mem:.2f} GB VRAM")
        
        # Log CUDA version
        cuda_version = torch.version.cuda
        logger.info(f"CUDA Version: {cuda_version}")
        
        # Log PyTorch CUDA capabilities
        logger.info(f"PyTorch CUDA available: {torch.cuda.is_available()}")
        logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
        logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
    else:
        device = torch.device("cpu")
        logger.info("No GPU found, using CPU.")
    
    return device

def log_system_info():
    """Log system information for diagnostics"""
    logger.info("System Information:")
    logger.info(f"  OS: {platform.system()} {platform.version()}")
    logger.info(f"  Python Version: {platform.python_version()}")
    logger.info(f"  PyTorch Version: {torch.__version__}")
    
    # CPU information
    logger.info(f"  CPU: {platform.processor()}")
    logger.info(f"  CPU Cores: {psutil.cpu_count(logical=False)} Physical, {psutil.cpu_count()} Logical")
    
    # Memory information
    mem_info = psutil.virtual_memory()
    logger.info(f"  RAM: {mem_info.total / (1024 ** 3):.2f} GB Total, {mem_info.available / (1024 ** 3):.2f} GB Available")

def optimize_for_gpu(batch_size, state_shape, min_batch_size=32, max_batch_size=512):
    """
    Adjust hyperparameters based on GPU availability and memory
    
    Args:
        batch_size: Initial batch size
        state_shape: Shape of state tensor (frames, height, width)
        min_batch_size: Minimum batch size to use
        max_batch_size: Maximum batch size to use
        
    Returns:
        tuple: (optimized_batch_size, memory_capacity)
    """
    # Default memory capacity - will be adjusted based on GPU
    memory_capacity = 50000
    
    if torch.cuda.is_available():
        # Get GPU memory in GB
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        logger.info(f"Optimizing hyperparameters for {gpu_mem_gb:.2f} GB GPU VRAM")
        
        # Calculate state tensor size in bytes
        # Each frame is a float32 (4 bytes)
        state_size = 4 * np.prod(state_shape)  # bytes per state
        
        # Transition size includes state, next_state, action, reward, done
        # action is int64 (8 bytes), reward is float32 (4 bytes), done is bool (1 byte)
        transition_size = 2 * state_size + 8 + 4 + 1
        
        # Reserve memory for model, optimizers, etc. (rough estimate)
        reserved_memory = 1.0  # GB
        
        # Calculate available memory for replay buffer
        available_mem_gb = max(0.1, gpu_mem_gb - reserved_memory)
        
        # Calculate max capacity based on available memory (with safety factor of 0.8)
        memory_capacity = int((available_mem_gb * 1024**3 * 0.8) / transition_size)
        
        # Adjust batch size based on GPU memory
        # For smaller GPUs (< 4GB), use smaller batch size
        if gpu_mem_gb < 4:
            batch_size = max(min_batch_size, batch_size // 2)
        # For larger GPUs (> 8GB), use larger batch size
        elif gpu_mem_gb > 8:
            batch_size = min(max_batch_size, batch_size * 2)
        
        logger.info(f"Adjusted for GPU: Batch size = {batch_size}, Memory capacity = {memory_capacity}")
    else:
        logger.info("No GPU detected, using default hyperparameters")
        # For CPU, use smaller batch size and memory capacity
        batch_size = max(min_batch_size, batch_size // 2)
        memory_capacity = min(10000, memory_capacity)
    
    return batch_size, memory_capacity

def measure_inference_time(model, input_shape, num_trials=100, device='cpu'):
    """
    Measure the average inference time of a model
    
    Args:
        model: Neural network model
        input_shape: Shape of input tensor (batch_size, frames, height, width)
        num_trials: Number of trials to average over
        device: Device to run the model on (cuda or cpu)
        
    Returns:
        float: Average inference time in milliseconds
    """
    model.to(device)
    model.eval()
    
    # Create dummy input tensor
    dummy_input = torch.randn(input_shape).to(device)
    
    # Warm-up runs
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)
            
    # Measurement runs
    start_times = []
    end_times = []
    
    for _ in range(num_trials):
        # Make sure all CUDA operations are finished
        if device == torch.device("cuda"):
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy_input)
        
        # Ensure GPU operations are finished if using CUDA
        if device == torch.device("cuda"):
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        start_times.append(start_time)
        end_times.append(end_time)
        
    # Calculate average inference time in milliseconds
    inference_times = [(e - s) * 1000 for s, e in zip(start_times, end_times)]
    avg_time_ms = np.mean(inference_times)
    std_time_ms = np.std(inference_times)
    
    # Log detailed inference time statistics
    logger.info(f"Inference Time Statistics (ms):")
    logger.info(f"  Mean: {avg_time_ms:.2f}")
    logger.info(f"  Std Dev: {std_time_ms:.2f}")
    logger.info(f"  Min: {min(inference_times):.2f}")
    logger.info(f"  Max: {max(inference_times):.2f}")
    
    # Plot histogram of inference times
    plt.figure(figsize=(10, 5))
    plt.hist(inference_times, bins=20, alpha=0.7)
    plt.axvline(avg_time_ms, color='r', linestyle='dashed', linewidth=2, label=f'Mean: {avg_time_ms:.2f} ms')
    plt.title(f"Inference Time Distribution (n={num_trials})")
    plt.xlabel("Time (ms)")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Create plots directory if it doesn't exist
    os.makedirs("logs/plots", exist_ok=True)
    
    # Save plot
    plt.savefig(f"logs/plots/inference_time_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.close()
    
    return avg_time_ms

def save_debug_image(image, filename, subfolder=None):
    """
    Save an image for debugging purposes
    
    Args:
        image: Image as numpy array
        filename: Name of the file
        subfolder: Optional subfolder within debug_images
    """
    # Create debug_images directory if it doesn't exist
    base_dir = "debug_images"
    os.makedirs(base_dir, exist_ok=True)
    
    # If subfolder is specified, create it
    if subfolder:
        save_dir = os.path.join(base_dir, subfolder)
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir = base_dir
    
    # Full path to save the image
    full_path = os.path.join(save_dir, filename)
    
    # Save the image
    plt.imsave(full_path, image)
    
    return full_path

def format_time(seconds):
    """
    Format time in seconds to a readable string (e.g., 1h 30m 45s)
    
    Args:
        seconds: Time in seconds
        
    Returns:
        str: Formatted time string
    """
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # Format without leading zeros
    if hours > 0:
        return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
    elif minutes > 0:
        return f"{int(minutes)}m {int(seconds)}s"
    else:
        return f"{int(seconds)}s"

def estimate_remaining_time(elapsed_time, episodes_completed, total_episodes):
    """
    Estimate remaining time based on elapsed time and progress
    
    Args:
        elapsed_time: Time elapsed so far (seconds)
        episodes_completed: Number of episodes completed
        total_episodes: Total number of episodes
        
    Returns:
        str: Formatted remaining time estimate
    """
    if episodes_completed == 0:
        return "Unknown"
    
    # Calculate time per episode
    time_per_episode = elapsed_time / episodes_completed
    
    # Estimate remaining time
    remaining_episodes = total_episodes - episodes_completed
    remaining_time = time_per_episode * remaining_episodes
    
    return format_time(remaining_time)

def calculate_ewma(values, alpha=0.1):
    """
    Calculate Exponentially Weighted Moving Average (EWMA)
    
    Args:
        values: List of values
        alpha: Smoothing factor (0 < alpha < 1)
        
    Returns:
        list: EWMA values
    """
    ewma = [values[0]]  # Initialize with first value
    
    for i in range(1, len(values)):
        ewma.append(alpha * values[i] + (1 - alpha) * ewma[i-1])
    
    return ewma

def monitor_memory_usage():
    """Monitor and log memory usage"""
    # Get GPU memory usage if available
    if torch.cuda.is_available():
        # Get allocated memory in MB
        allocated_memory = torch.cuda.memory_allocated() / (1024 ** 2)
        # Get cached memory in MB
        cached_memory = torch.cuda.memory_reserved() / (1024 ** 2)
        
        logger.info(f"GPU Memory: {allocated_memory:.2f} MB allocated, {cached_memory:.2f} MB cached")
    
    # Get system memory usage
    mem_info = psutil.virtual_memory()
    used_memory = mem_info.used / (1024 ** 3)  # GB
    total_memory = mem_info.total / (1024 ** 3)  # GB
    memory_percent = mem_info.percent
    
    logger.info(f"System Memory: {used_memory:.2f} GB / {total_memory:.2f} GB ({memory_percent}%)")