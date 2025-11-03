#!/usr/bin/env python3
"""
Example script demonstrating how to use the refine-pipeline processing system.
This script shows different ways to run the dataset processing with various configurations.
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and print the description."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")

def main():
    """Main function to demonstrate different usage scenarios."""
    script_dir = Path(__file__).parent
    main_script = script_dir / "main.py"
    
    if not main_script.exists():
        print(f"Error: {main_script} not found!")
        sys.exit(1)
    
    print("Refine-Pipeline Processing Examples")
    print("===================================")
    
    # Example 1: Sample mode for testing
    run_command([
        sys.executable, str(main_script),
        "--sample_mode",
        "--sample_size", "3"
    ], "Sample mode with 3 images using config.yaml")
    
    # Example 2: Process with custom max images
    print("\n" + "="*60)
    print("Example 2: Process 10 images with debug output")
    print("="*60)
    print("Command:")
    print(f"python3 {main_script} --max_images 10 --debug")
    print("\n(Not running this example to save time, but this is how you would do it)")
    
    # Example 3: Full processing using configuration file
    print("\n" + "="*60)
    print("Example 3: Full dataset processing using config.yaml")
    print("="*60)
    print("Command for full dataset processing:")
    print(f"python3 {main_script}")
    print("\nThis uses all settings from config.yaml including:")
    print("  - Percentage-based prompt distribution")
    print("  - One prompt per image")
    print("  - Only images present in all metadata files")
    print("  - Configurable batch size and workers")
    
    # Example 4: Custom configuration file
    print("\n" + "="*60)
    print("Example 4: Using custom configuration file")
    print("="*60)
    print("Command for custom config:")
    print(f"python3 {main_script} --config custom_config.yaml")
    print("\nTo create a custom config, copy config.yaml and modify:")
    print("  - prompt distribution percentages")
    print("  - processing parameters")
    print("  - API settings")
    
    print("\n" + "="*60)
    print("Key Features:")
    print("- YAML configuration with hierarchical parameters")
    print("- Percentage-based prompt distribution (configurable)")
    print("- One prompt per image based on dataset distribution")
    print("- Only processes images present in ALL metadata files")
    print("- Automatic progress saving and resume capability")
    print("- Multiprocessing for high throughput")
    print("")
    print("Configuration Tips:")
    print("- Edit config.yaml to adjust prompt distribution percentages")
    print("- Set random_seed for reproducible prompt assignment")
    print("- Adjust num_workers based on CPU cores and API limits")
    print("- Use sample_mode: true in config for testing")
    print("="*60)

if __name__ == "__main__":
    main()
