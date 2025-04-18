#!/usr/bin/env python
"""
Subway Surfers AI Training Script - Simplified Launcher

This script provides a simple way to run the Subway Surfers AI training with different configurations.
"""

import subprocess
import argparse
import os
import sys
import platform
import psutil
import torch
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="Run Subway Surfers AI Training")
    
    # Basic options
    parser.add_argument("--mode", type=str, choices=["train", "test", "evaluate"], default="train",
                        help="Mode: train a new model, test/visualize an existing model, or evaluate a model")
    parser.add_argument("--episodes", type=int, default=500,
                        help="Number of episodes to train for")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint file to load")
    
    # Performance and hardware options
    parser.add_argument("--performance", type=str, choices=["low", "medium", "high", "auto"], default="auto",
                        help="Performance setting (affects resource usage)")
    parser.add_argument("--browser-position", type=str, default="right", choices=["left", "right"],
                        help="Position of browser window")
    
    # Debug and monitoring options
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode")
    parser.add_argument("--visualize", action="store_true",
                        help="Enable detailed visualization during training")
    
    # Advanced options
    parser.add_argument("--no-adaptive", action="store_true",
                        help="Disable adaptive hyperparameter tuning")
    parser.add_argument("--no-memory-efficient", action="store_true",
                        help="Disable memory efficiency optimizations")
    parser.add_argument("--skip-browser", action="store_true",
                        help="Skip browser (for debugging only)")
    
    return parser.parse_args()

def get_system_specs():
    """Get system specifications to optimize performance settings"""
    try:
        import psutil
        
        specs = {
            "cpu_cores": os.cpu_count(),
            "memory_gb": round(psutil.virtual_memory().total / (1024**3), 1),
            "has_gpu": torch.cuda.is_available(),
        }
        
        if specs["has_gpu"]:
            try:
                specs["gpu_name"] = torch.cuda.get_device_name(0)
                specs["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)
            except:
                specs["gpu_name"] = "Unknown"
                specs["gpu_memory_gb"] = 0
                
        return specs
    except ImportError:
        print("Warning: psutil not installed. Using default specs.")
        return {
            "cpu_cores": os.cpu_count(),
            "memory_gb": 8.0,
            "has_gpu": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
            "gpu_memory_gb": 4.0 if torch.cuda.is_available() else 0
        }

def autodetect_performance_settings(specs):
    """Automatically determine performance settings based on hardware specs"""
    # For GPU systems
    if specs["has_gpu"]:
        if specs["gpu_memory_gb"] >= 8:
            return "high"
        elif specs["gpu_memory_gb"] >= 4:
            return "medium"
        else:
            return "low"
    # For CPU-only systems
    else:
        if specs["memory_gb"] >= 16 and specs["cpu_cores"] >= 8:
            return "medium"
        else:
            return "low"

def apply_performance_settings(cmd, performance, specs):
    """Apply performance settings to command line arguments"""
    # Low performance profile (minimal resource usage)
    if performance == "low":
        cmd.append("--memory_efficient")  # Boolean flag, no value
        cmd.extend(["--batch_size", "16"])
        cmd.extend(["--memory_capacity", "5000"])
        cmd.extend(["--frame_stack", "3"])
    
    # Medium performance profile (balanced)
    elif performance == "medium":
        cmd.append("--memory_efficient")  # Boolean flag, no value
        cmd.extend(["--batch_size", "32"])
        cmd.extend(["--memory_capacity", "10000"])
        cmd.extend(["--frame_stack", "4"])
    
    # High performance profile (maximized quality)
    elif performance == "high":
        cmd.extend(["--batch_size", "64"])
        cmd.extend(["--memory_capacity", "20000"])
        cmd.extend(["--frame_stack", "4"])
    
    return cmd

def main():
    # Check for required libraries
    try:
        import psutil
    except ImportError:
        print("Warning: psutil library not found. Some features may be limited.")
    
    args = parse_args()
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get system specs for auto-configuration
    specs = get_system_specs()
    
    # Auto-detect performance settings if set to auto
    performance = args.performance
    if performance == "auto":
        performance = autodetect_performance_settings(specs)
        print(f"Auto-detected performance setting: {performance}")
    
    # Base command
    cmd = [sys.executable]
    
    # Create model directory with timestamp
    model_dir = f"models/{timestamp}"
    os.makedirs(model_dir, exist_ok=True)
    
    # Handle different modes
    if args.mode == "train":
        # Training mode
        cmd.append("main.py")
        cmd.extend(["--max_episodes", str(args.episodes)])
        cmd.extend(["--browser_position", args.browser_position])
        cmd.extend(["--model_dir", model_dir])
        
        # Apply performance settings
        cmd = apply_performance_settings(cmd, performance, specs)
        
        # Add checkpoints if specified
        if args.checkpoint:
            cmd.extend(["--load_checkpoint", args.checkpoint])
        
        # Debug and visualization
        if args.debug:
            cmd.append("--debug")
        
        if args.visualize:
            cmd.append("--detailed_monitoring")
        
        # Optional flags
        if args.no_adaptive:
            # Don't add --adaptive flag
            pass
        else:
            cmd.append("--adaptive")  # Boolean flag, no value
        
        if args.no_memory_efficient:
            # Don't add --memory_efficient flag
            pass
        # Note: memory_efficient is already handled in apply_performance_settings
        
        if args.skip_browser:
            cmd.append("--skip_browser")
        
        # Always use best algorithm settings
        cmd.append("--use_dueling")
        cmd.append("--use_double")
        cmd.append("--use_per")
        
    elif args.mode == "test":
        # Test/visualization mode
        cmd.append("test.py")
        
        # Must have a checkpoint
        if not args.checkpoint:
            print("Error: test mode requires a checkpoint. Use --checkpoint to specify.")
            return
        
        cmd.extend(["--checkpoint", args.checkpoint])
        cmd.extend(["--browser_position", args.browser_position])
        
        # Debug mode
        if args.debug:
            cmd.append("--debug")
        
    elif args.mode == "evaluate":
        # Evaluation mode
        cmd.append("evaluate.py")
        
        # Must have a checkpoint
        if not args.checkpoint:
            print("Error: evaluate mode requires a checkpoint. Use --checkpoint to specify.")
            return
        
        cmd.extend(["--checkpoint", args.checkpoint])
        cmd.extend(["--browser_position", args.browser_position])
        cmd.extend(["--episodes", str(args.episodes)])
        
        # Debug mode
        if args.debug:
            cmd.append("--debug")
    
    # Print command
    print("=" * 80)
    print("Subway Surfers AI Training".center(80))
    print("=" * 80)
    print(f"Mode: {args.mode}")
    print(f"Performance setting: {performance}")
    print(f"System: {specs['cpu_cores']} CPU cores, {specs['memory_gb']} GB RAM")
    if specs["has_gpu"]:
        print(f"GPU: {specs['gpu_name']} ({specs['gpu_memory_gb']} GB)")
    else:
        print("GPU: None (CPU mode)")
    print("-" * 80)
    print(f"Running command: {' '.join(cmd)}")
    if args.mode == "train":
        print(f"Models will be saved to: {os.path.abspath(model_dir)}")
    print("=" * 80)
    
    # Run the command
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nCommand interrupted by user. Your progress is saved in the checkpoint files.")
    except Exception as e:
        print(f"\nError during execution: {str(e)}")

if __name__ == "__main__":
    main()