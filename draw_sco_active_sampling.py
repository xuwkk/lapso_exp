"""
Visualize on the sco experiment with active learning strategies
"""


import numpy as np
import matplotlib.pyplot as plt

rc = {"font.family" : "serif", 
      "mathtext.fontset" : "stix",
      }
plt.rcParams.update(rc)
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
font = {'size'   : 20}
plt.rc('font', **font)

# Define colors
BLUE = '#2C73D2'
RED = '#FF6666'
GREEN = '#2ac195'
PURPLE = '#8290bb'
YELLOW = '#cda23d'

# BLUE = '#4444FF'
# RED = '#FF4444'
# GREEN = '#44FF44'
# PURPLE = '#FF44FF'
# YELLOW = '#FFFF44'

sco_dir = "paper_exp/sco_smart_sample_result"
grid_list = ["bus14", "bus39", "bus57", "bus118", "bus300"]
nn_list = ["nn_very_large", "nn_large_1", "nn_large_2", "nn_large_3"]
solver_list = ["GUROBI"]
# solver_list = ["MOSEK"]

solver_ = solver_list[0]

computation_time = {}
trainable_param = {}
sr = {}
no_binary = {}
ur = {}
no_param = {}

for grid_ in grid_list:
    computation_time[grid_] = []
    trainable_param[grid_] = []
    sr[grid_] = []
    no_binary[grid_] = []
    ur[grid_] = []
    no_param[grid_] = []

    for idx, nn_ in enumerate(nn_list):
        print(f"\n====== Grid: {grid_}, NN: {nn_}, Solver: {solver_} ======")
        ori_dir_ = f"{sco_dir}/{grid_}/{nn_}/{solver_}/ori_result.npy"
        sco_dir_ = f"{sco_dir}/{grid_}/{nn_}/{solver_}/sco_result.npy"
        
        ori = np.load(ori_dir_, allow_pickle=True).item()
        sco = np.load(sco_dir_, allow_pickle=True).item()

        no_day = len(ori['cost'])
        no_hour = np.prod(ori['gscr_cls'].shape)

        print(f"No. of Days: {no_day}, No. of Hours: {no_hour}")

        ori_cost = np.mean(np.array(ori['cost']))
        sco_cost = np.mean(np.array(sco['cost']))

        print("Average Cost $:")
        print(f"Original: {np.round(ori_cost * 100, 2)}, SCO: {np.round(sco_cost * 100, 2)}")

        # for i in range(len(ori['gscr'][0])):
        #     print(f">>>>Hour {i}: {round(ori['gscr'][0][i],2)} -> {round(sco['gscr'][0][i],2)}")
        #     print(f"{round(ori['gscr_cls'][0][i],2)} -> {round(sco['gscr_cls'][0][i],2)}")
        #     print(f"no_on_gen: {np.sum(ori['ug'][0][i])} -> {np.sum(sco['ug'][0][i])}")
        #     ori_on_idx = np.where(ori['ug'][0][i] == 1)[0] + 1
        #     sco_on_idx = np.where(sco['ug'][0][i] == 1)[0] + 1
        #     # sco idx that is not in ori idx
        #     sco_on_but_not_ori = np.setdiff1d(sco_on_idx, ori_on_idx)
        #     print(sco_on_but_not_ori)
        #     print(f"rc: {np.sum(ori['solarc'][0][i])} -> {np.sum(sco['solarc'][0][i])}")

        # stable hourly
        ori_cls_stable_idx_hour = np.where(ori['gscr_cls'].flatten() <= 0)[0] # NOTE: for classification only
        sco_cls_stable_idx_hour = np.where(sco['gscr_cls'].flatten() <= 0)[0]
        ori_gscr_stable_idx_hour = np.where(ori['gscr'].flatten() >= 2.5)[0]
        sco_gscr_stable_idx_hour = np.where(sco['gscr'].flatten() >= 2.5)[0]
        
        # unstable hourly
        ori_cls_unstable_idx_hour = np.where(ori['gscr_cls'].flatten() > 0)[0]
        sco_cls_unstable_idx_hour = np.where(sco['gscr_cls'].flatten() > 0)[0]
        ori_gscr_unstable_idx_hour = np.where(ori['gscr'].flatten() < 2.5)[0]
        sco_gscr_unstable_idx_hour = np.where(sco['gscr'].flatten() < 2.5)[0]
        
        print("UR-GSCR-HOUR (%): ")
        print(f"Original: {np.round(len(ori_gscr_unstable_idx_hour)/no_hour * 100, 2)}, SCO: {np.round(len(sco_gscr_unstable_idx_hour)/no_hour * 100, 2)}")

        SR_HOUR = len(np.intersect1d(ori_gscr_unstable_idx_hour, sco_gscr_stable_idx_hour)) / len(ori_gscr_unstable_idx_hour)
        DR_HOUR = len(np.intersect1d(ori_gscr_stable_idx_hour, sco_gscr_unstable_idx_hour)) / len(ori_gscr_stable_idx_hour)
        
        print(f"SR-HOUR: {np.round(SR_HOUR * 100, 4)}, DR-HOUR: {np.round(DR_HOUR * 100, 4)}")
        
        # Overreaction, on ori: gscr stable -> cls unstable
        OR_HOUR = len(np.intersect1d(ori_gscr_stable_idx_hour, ori_cls_unstable_idx_hour)) / len(ori_gscr_stable_idx_hour)

        print(f"OR-HOUR: {np.round(OR_HOUR * 100, 5)}")
        
        print(f"No. of Binary: ")
        print(f"Original: {ori['no_binary_var']}, SCO: {sco['no_binary_var']}")
        
        print(f"Computation Time: ")
        print(f"Original: {np.round(ori['time'], 3)}, SCO: {np.round(sco['time'], 3)}")
        
        print(f"No. of Trainable Parameters: {sco['no_trainable_param']}")

        # Per grid summary
        computation_time[grid_].append(np.round(sco['time'],2))
        trainable_param[grid_].append(np.round(sco['no_trainable_param'],2))
        sr[grid_].append(np.round(SR_HOUR * 100,2))
        no_binary[grid_].append(np.round(sco['no_binary_var'],2))
        ur[grid_].append(np.round(len(ori_gscr_unstable_idx_hour)/no_hour * 100,2))
        no_param[grid_].append(sco['no_trainable_param'])

print("\n\n====== Summary ======")

print("\nSR-HOUR: ")
print(sr)
print("\nNo. of Binary: ")
print(no_binary)
print("\nComputation Time (s): ")
print(computation_time)
print("\nNo. of Trainable Parameters: ")
print(no_param)

print("\nTrainable Parameters: ")
print(trainable_param)
print("\nUR-HOUR (%): ")
print(ur)


# Draw 3D plot for computational time x-axis is the grid, y-axis is the nn, z-axis is the time
# Example: assume computation_time is a dict {grid_name: [t_nn1, t_nn2, ...]}
# grid_list = ['Grid1', 'Grid2', 'Grid3']
# nn_list = ['NN1', 'NN2', 'NN3', 'NN4']

grid_list_disp = ["14", "39", "57", "118", "300"]
nn_list_disp = ["T", "S", "M", "L"]

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_proj_type('ortho')  # orthographic = no perspective distortion

# Create meshgrid for bar positions
_x = np.arange(len(grid_list))
_y = np.arange(len(nn_list))
_xx, _yy = np.meshgrid(_x, _y)
x, y = _xx.ravel(), _yy.ravel()

# Flatten computation_time data in the correct order
top = np.array([computation_time[grid][i] for i in range(len(nn_list)) for grid in grid_list])

bottom = np.zeros_like(top)
width = depth = 0.45

# Define colors for each bus (assuming 4 buses)
colors = [BLUE, RED, YELLOW, GREEN, PURPLE]  # Red, Green, Blue, Yellow
color_map = [colors[i % len(colors)] for i in range(len(top))]

ax.bar3d(x, y, bottom, width, depth, top, shade=True, color=color_map,
         edgecolor='black', linewidth=0.3, alpha=0.55)

# Labeling
font_size = 26
ax.set_xticks(_x + width / 2)
ax.set_xticklabels(grid_list_disp, fontsize=font_size, fontweight='bold')
ax.set_yticks(_y + depth / 2)
ax.set_yticklabels(nn_list_disp, fontsize=font_size, fontweight='bold')
ax.set_zticks(np.arange(0,3700,800))  # Set z ticks evenly
ax.set_zticklabels(np.arange(0,3700,800), fontsize=font_size, fontweight='bold')  # Set z tick labels

ax.set_xlabel('Grid', labelpad=20, fontweight='bold', fontsize=font_size)
ax.set_ylabel('NN', labelpad=20, fontweight='bold', fontsize=font_size)
ax.set_zlabel('Computation Time (s)', labelpad=30, fontweight='bold', fontsize=font_size)

# Increase the distance of z ticks to the plot
ax.zaxis.set_tick_params(pad=10)

# --- White background ---
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# --- Remove grey 3D panes and make them transparent ---
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
plt.savefig(f'{sco_dir}/computation_time_3d.pdf', bbox_inches='tight', pad_inches=0.5)
plt.show()

"""
Plot time vs no. of binary variables
"""

for idx, grid_name in enumerate(grid_list):
    fig, ax1 = plt.subplots(figsize=(6, 4), dpi=300)

    # Plot No. of Binary Variables on left y-axis with markers and grid
    ax1.plot(nn_list_disp, np.array(no_binary[grid_name]), color=BLUE, marker='s', linewidth=2, markersize=8, label='No. of Binary')
    ax1.set_ylabel(r'NO. Binary ($\times 10$)', color=BLUE)
    ax1.set_ylim(no_binary[grid_name][0]*0.9, no_binary[grid_name][-1] * 1.3)
    ax1.tick_params(axis='y', labelcolor=BLUE)
    ax1.grid(True, linestyle='--', alpha=0.7)
    # plt.xticks(rotation=15)

    # Plot Time on right y-axis
    ax2 = ax1.twinx()
    ax2.plot(nn_list_disp, computation_time[grid_name], color=RED, marker='o', linewidth=2, markersize=8, label='Time')
    ax2.set_ylabel('Compute Time (s)', color=RED)
    ax2.tick_params(axis='y', labelcolor=RED)
    ax2.set_ylim(0.1, max(computation_time[grid_name]) * 5)
    ax2.set_yscale('log')

    # Add legend with white background
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    if idx == 0:
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left',
            bbox_to_anchor=(0, 1), frameon=False)

    # Add title and adjust layout
    # plt.title('Computation Time vs Binary Variables', pad=20, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{sco_dir}/{grid_name}_time_vs_no_binary.pdf', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()