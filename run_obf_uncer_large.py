"""
Run all the OBF/Uncer experiments with different budget ratios.
"""

import subprocess
import sys

def run_obf_experiments(grid_name, sample_size):

    if grid_name == "bus39":
        M_DP = 1e4
        M_RD = 1e4
    elif grid_name == "bus57":
        M_DP = 5e4
        M_RD = 5e4
    elif grid_name == "bus118":
        M_DP = 1e5
        M_RD = 1e5
    else:
        raise ValueError(f"Unsupported grid in script: {grid_name}")

    budget_ratio = [0.05]
    for ratio in budget_ratio:
        print(f"\nRunning OBF experiment with budget ratio: {ratio}")
        command = [
            "python", "paper_exp/obf_uncer.py", 
            f"grid={grid_name}", 
            f"operation={grid_name}_continuous", 
            "exp=obf_uncer", 
            "exp.train_config.verbose=false",
            f"exp.budget_ratio={ratio}", 
            f"exp.train_config.M_DP={M_DP}", 
            f"exp.train_config.M_RD={M_RD}", 
            f"exp.no_train_sample={sample_size}"
        ]
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running OBF experiment with budget ratio {ratio}: {e}")
            sys.exit(1)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run OBF experiments with specified grid and sample size.")
    parser.add_argument("--grid", type=str, required=True, help="Grid to use (e.g., bus39)")
    parser.add_argument("--sample_size", type=int, default=10, help="Number of training samples (default: 1000)")
    args = parser.parse_args()

    run_obf_experiments(grid_name=args.grid, sample_size=args.sample_size)