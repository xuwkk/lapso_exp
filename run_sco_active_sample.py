import subprocess
import sys
import time

def run_sco_experiments(solver, grid):
    if solver == "GUROBI":
        net_list = [
            "nn_very_large", "nn_large_1", "nn_large_2", "nn_large_3"
            ]
    else:
        net_list = [
            "nn_very_large", "nn_large_1", "nn_large_2", "nn_large_3"
        ]
    
    if grid == "bus14":
        renewable_rescale = 1.4
        over_value = 0.05
    elif grid == "bus39":
        renewable_rescale = 0.6
        over_value = 0.3
    elif grid == "bus57":
        renewable_rescale = 0.6
        over_value = 0.5
    elif grid == "bus118":
        renewable_rescale = 0.7
        over_value = 0.5
    elif grid == "bus300":
        renewable_rescale = 0.3
        over_value = 1.5
    else:
        raise ValueError(f"Unknown grid: {grid}")
    
    for idx, type in enumerate(net_list):
        command = [
            "python", "paper_exp/sco_active_sample.py", 
            f"grid={grid}",
            f"operation={grid}_discrete", 
            "exp=sco_active_sample", 
            f"exp.type={type}",
            f"exp.optimization.solver={solver}",
            f"exp.train_new_nn=true",
            f"exp.renewable_rescale={renewable_rescale}",
            f"exp.over_value={over_value}",
            # Only need to generate the training dataset once per grid
            "exp.generate_new_dataset=false" if idx == 0 else "exp.generate_new_dataset=false",
            "exp.generate_new_aug=true" if idx == 0 else "exp.generate_new_aug=false"
        ]
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running experiment {type}: {e}")
            sys.exit(1)
        
        time.sleep(2)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run SCO experiments with specified solver and grid.")
    parser.add_argument("--solver", type=str, default="GUROBI", help="Solver to use (e.g., GUROBI)")
    parser.add_argument("--grid", type=str, required=True, help="Grid to use (e.g., bus39)")
    args = parser.parse_args()

    run_sco_experiments(grid=args.grid, solver=args.solver.upper())