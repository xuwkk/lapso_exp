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
BLUE = '#4444FF'
RED = '#FF4444'

GREEN = '#44FF44'
PURPLE = '#FF44FF'
YELLOW = '#FFFF44'

"""
Draw the areas
"""

# Create the plot
fig, ax = plt.subplots(figsize=(8, 4))

# Define line segments
x_vals = np.array([-4, 4])
true_boundary = -0.5 * x_vals  # Red line (True boundary)
data_boundary = 0.5 * x_vals   # Blue line (Data-driven boundary)

# Plot boundaries
ax.plot(x_vals, true_boundary, RED, label='True boundary', linewidth=2)
ax.plot(x_vals, data_boundary, BLUE, label='Data-driven boundary', linewidth=2)

# Add patterned regions
ax.fill_between(x_vals, true_boundary, -2, color=RED, alpha=0.1, hatch='/', linestyle='--')
ax.fill_between(x_vals, data_boundary, 2, color=BLUE, alpha=0.1, hatch='.', linestyle='--')

# Region labels
ax.text(-2, 0, r'$\mathcal{R}_1$', fontsize=30)
ax.text(0, 1.0, r'$\mathcal{R}_2$', fontsize=30) 
ax.text(2, 0, r'$\mathcal{R}_3$', fontsize=30)
ax.text(0, -1.2, r'$\mathcal{R}_4$', fontsize=30)

# Text annotations for boundary labels
ax.text(1.0, 0.65, 'Data-driven\n  boundary', fontsize=22, color=BLUE, rotation=20)
ax.text(1.0, -1.75, 'True boundary', fontsize=22, color=RED, rotation=-18)

# Clean up plot
ax.set_xlim(-3, 3)
ax.set_ylim(-2, 2)
ax.axis('off')
plt.tight_layout()
plt.savefig('paper_exp/sco_area.pdf', bbox_inches='tight', dpi=300, pad_inches=0)
plt.show()

"""
Evaluate the results
"""

# Load results from different models
sco_dir = "paper_exp/sco_result/"
models = {
    # model dir, 
    'linear_bce': ('bce_ori', 'bce_sco'),
    'linear_cbce': ('cbce_ori', 'cbce_sco'), 
    'nn_small': ('small_ori', 'small_sco'),
    'nn_medium': ('medium_ori', 'medium_sco'),
    'nn_large': ('large_ori', 'large_sco'),
    'nn_very_large': ('vlarge_ori', 'vlarge_sco')
}

# Load all results
results = {}
for model_dir, (ori_name, sco_name) in models.items():
    results[ori_name] = np.load(f"{sco_dir}/{model_dir}/ori_result.npy", allow_pickle=True).item()
    results[sco_name] = np.load(f"{sco_dir}/{model_dir}/sco_result.npy", allow_pickle=True).item()

print('ori keys: ', results['large_ori'].keys())
print('sco keys: ', results['large_sco'].keys())

def evaluate(ori, sco, start_day = 0, end_day = -1):
    """Evaluate and compare original and SCO results"""
    
    # Construct the reduced dataset
    ori_reduced = {}
    sco_reduced = {}
    for key, value in ori.items():
        if isinstance(value, np.ndarray):
            ori_reduced[key] = value[start_day:end_day]
        else:
                ori_reduced[key] = value
                
    for key, value in sco.items():
        if isinstance(value, np.ndarray):
            sco_reduced[key] = value[start_day:end_day]
        else:
            sco_reduced[key] = value
                
    ori = ori_reduced
    sco = sco_reduced
    
    no_day = len(ori['cost'])
    no_hour = np.prod(ori['gscr_cls'].shape)
    
    # print(f"Number of days: {no_day}, Total hours: {no_hour}") # 365, 8760
    # print(f"GSCR shape: {ori['gscr_cls'].shape}")  # 365*24
    
    # Cost comparison
    print("Average Cost:")
    print(f"Original: {np.round(np.mean(ori['cost']) * 100, 2)}, SCO: {np.round(np.mean(sco['cost']) * 100, 2)}")
    
    # stable hourly
    ori_cls_stable_idx_hour = np.where(ori['gscr_cls'].flatten() <= 0)[0]
    sco_cls_stable_idx_hour = np.where(sco['gscr_cls'].flatten() <= 0)[0]
    ori_gscr_stable_idx_hour = np.where(ori['gscr'].flatten() >= 2.5)[0]
    sco_gscr_stable_idx_hour = np.where(sco['gscr'].flatten() >= 2.5)[0]
    
    # stable daily
    ori_cls_stable_idx_day = np.where(np.sum(ori['gscr_cls'] <= 0, axis=1) == 24)[0]
    sco_cls_stable_idx_day = np.where(np.sum(sco['gscr_cls'] <= 0, axis=1) == 24)[0]
    ori_gscr_stable_idx_day = np.where(np.sum(ori['gscr'] >= 2.5, axis=1) == 24)[0]
    sco_gscr_stable_idx_day = np.where(np.sum(sco['gscr'] >= 2.5, axis=1) == 24)[0]
    
    # unstable hourly
    ori_cls_unstable_idx_hour = np.where(ori['gscr_cls'].flatten() > 0)[0]
    sco_cls_unstable_idx_hour = np.where(sco['gscr_cls'].flatten() > 0)[0]
    ori_gscr_unstable_idx_hour = np.where(ori['gscr'].flatten() < 2.5)[0]
    sco_gscr_unstable_idx_hour = np.where(sco['gscr'].flatten() < 2.5)[0]
    
    # unstable daily
    ori_cls_unstable_idx_day = np.where(np.sum(ori['gscr_cls'] > 0, axis=1) > 0)[0]
    sco_cls_unstable_idx_day = np.where(np.sum(sco['gscr_cls'] > 0, axis=1) > 0)[0]
    ori_gscr_unstable_idx_day = np.where(np.sum(ori['gscr'] < 2.5, axis=1) > 0)[0]
    sco_gscr_unstable_idx_day = np.where(np.sum(sco['gscr'] < 2.5, axis=1) > 0)[0]
    
    # print("UR-CLS-HOUR (%): ")
    # print(f"Original: {np.round(len(ori_cls_unstable_idx_hour)/no_hour * 100, 2)}, SCO: {np.round(len(sco_cls_unstable_idx_hour)/no_hour * 100, 2)}")
    
    # print("UR-GSCR-HOUR (%): ")
    # print(f"Original: {np.round(len(ori_gscr_unstable_idx_hour)/no_hour * 100, 2)}, SCO: {np.round(len(sco_gscr_unstable_idx_hour)/no_hour * 100, 2)}")
    
    # print("UR-CLS-DAY (%): ")
    # print(f"Original: {np.round(len(ori_cls_unstable_idx_day)/no_day * 100, 2)}, SCO: {np.round(len(sco_cls_unstable_idx_day)/no_day * 100, 2)}")
    
    print("UR-GSCR-DAY (%): ")
    print(f"Original: {np.round(len(ori_gscr_unstable_idx_day)/no_day * 100, 2)}, SCO: {np.round(len(sco_gscr_unstable_idx_day)/no_day * 100, 2)}")
    
    # Stablize rate: unstable -> stable
    SR_DAY = len(np.intersect1d(ori_gscr_unstable_idx_day, sco_gscr_stable_idx_day)) / len(ori_gscr_unstable_idx_day)
    # Stablize rate: stable -> unstable
    DR_DAY = len(np.intersect1d(ori_gscr_stable_idx_day, sco_gscr_unstable_idx_day)) / len(ori_gscr_stable_idx_day)
    
    # print(f"SR-DAY: {np.round(SR_DAY * 100, 4)}, DR-DAY: {np.round(DR_DAY * 100, 4)}")
    
    SR_HOUR = len(np.intersect1d(ori_gscr_unstable_idx_hour, sco_gscr_stable_idx_hour)) / len(ori_gscr_unstable_idx_hour)
    DR_HOUR = len(np.intersect1d(ori_gscr_stable_idx_hour, sco_gscr_unstable_idx_hour)) / len(ori_gscr_stable_idx_hour)
    
    print(f"SR-HOUR: {np.round(SR_HOUR * 100, 4)}, DR-HOUR: {np.round(DR_HOUR * 100, 4)}")
    
    # Overreaction, gscr stable -> cls unstable
    OR_DAY = len(np.intersect1d(ori_gscr_stable_idx_day, ori_cls_unstable_idx_day)) / len(ori_gscr_stable_idx_day)
    OR_HOUR = len(np.intersect1d(ori_gscr_stable_idx_hour, ori_cls_unstable_idx_hour)) / len(ori_gscr_stable_idx_hour)

    print(f"OR-DAY: {np.round(OR_DAY * 100, 5)}, OR-HOUR: {np.round(OR_HOUR * 100, 5)}")
    
    print(f"No. of Binary: ")
    print(f"Original: {ori['no_binary_var']}, SCO: {sco['no_binary_var']}")
    
    print(f"Computation Time: ")
    print(f"Original: {np.round(ori['time'], 3)}, SCO: {np.round(sco['time'], 3)}")
    
    print(f"No. of Trainable Parameters: {sco['no_trainable_param']}")
    
    return OR_HOUR

# Evaluate all models
start_day = 0
end_day = -1
for model_name, (ori_name, sco_name) in models.items():
    print(f'\n{model_name.upper().replace("_", " ")}')
    print('-' * 50)
    OR = evaluate(results[ori_name], results[sco_name], start_day = start_day, end_day = end_day)
    results[sco_name]['OR'] = OR

cost_list = []
OR = []
time_list = []
no_binary_list = []
for model_name, (ori_name, sco_name) in models.items():
    # print(results[sco_name].keys())
    cost_list.append(results[sco_name]['cost'][start_day:end_day].mean())
    time_list.append(results[sco_name]['time'])
    no_binary_list.append(results[sco_name]['no_binary_var'])
    OR.append(results[sco_name]['OR'] * 100)

"""
Plot cost vs OR
"""

# Cost vs OR
fig, ax1 = plt.subplots(figsize=(8, 4), dpi=300)

model_name_list = ["LR", "cLR", r"$NN_2^{12}$", r"$NN_2^{111}$", r"$NN_3^{161}$", r"$NN_3^{221}$"]

# Plot OR on left y-axis with markers and grid
ax1.plot(model_name_list, np.array(OR), color=BLUE, marker='o', linewidth=2, markersize=8, label='OR')
ax1.set_ylabel('Overreact Rate (%)', color=BLUE)
ax1.tick_params(axis='y', labelcolor=BLUE)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.set_ylim(0, 15)
# plt.xticks(rotation=15)

# Plot Cost on right y-axis
ax2 = ax1.twinx()
ax2.plot(model_name_list, np.array(cost_list) * 1e-2, color=RED, marker='s', linewidth=2, markersize=8, label='Cost')
ax2.set_ylabel(r'PSO Cost ($\times 10^4$£)', color=RED)
ax2.tick_params(axis='y', labelcolor=RED)
ax2.set_ylim(1.7, 2.1)
# Add legend with white background
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', 
           bbox_to_anchor=(1, 1), frameon=False)

# Add title and adjust layout
# plt.title('Operational Risk vs Operating Cost', pad=20, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{sco_dir}/cost_vs_or.pdf', dpi=300, bbox_inches='tight', pad_inches=0)
plt.show()


"""
Plot time vs no. of binary variables
"""

fig, ax1 = plt.subplots(figsize=(8, 4), dpi=300)

# Plot No. of Binary Variables on left y-axis with markers and grid
ax1.plot(model_name_list, np.array(no_binary_list) / 10, color=BLUE, marker='s', linewidth=2, markersize=8, label='No. of Binary')
ax1.set_ylabel(r'NO. Binary ($\times 10$)', color=BLUE)
ax1.set_ylim(25, 95)
ax1.tick_params(axis='y', labelcolor=BLUE)
ax1.grid(True, linestyle='--', alpha=0.7)
# plt.xticks(rotation=15)

# Plot Time on right y-axis
ax2 = ax1.twinx()
ax2.plot(model_name_list, time_list, color=RED, marker='o', linewidth=2, markersize=8, label='Time')
ax2.set_ylabel('Compute Time (s)', color=RED)
ax2.tick_params(axis='y', labelcolor=RED)
ax2.set_yscale('log')
# ax2.set_ylim(1e-1, 1e2)

# Add legend with white background
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left',
           bbox_to_anchor=(0, 1), frameon=False)

# Add title and adjust layout
# plt.title('Computation Time vs Binary Variables', pad=20, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{sco_dir}/time_vs_no_binary.pdf', dpi=300, bbox_inches='tight', pad_inches=0)
plt.show()

"""
Plot daily results
"""
model_name = 'large'
ori_ug_hourly_total = results[model_name + '_ori']['ug'].sum(axis=-1)
sco_ug_hourly_total = results[model_name + '_sco']['ug'].sum(axis=-1)
no_day = len(ori_ug_hourly_total)
for day_idx in range(no_day):
    
    if not np.any(sco_ug_hourly_total[day_idx] == 5):
        continue
    
    solar_hourly_total = results[model_name + '_ori']['solar'].sum(axis=-1)
    ori_solar_hourly_total = results[model_name + '_ori']['solar'].sum(axis=-1) - results[model_name + '_ori']['solarc'].sum(axis=-1)
    # ori_solar_hourly_total = results[model_name + '_ori']['solar'].sum(axis=-1)
    sco_solar_hourly_total = results[model_name + '_sco']['solar'].sum(axis=-1) - results[model_name + '_sco']['solarc'].sum(axis=-1)

    # Plot
    fig, ax1 = plt.subplots(figsize=(8, 4), dpi=300)
    x = np.arange(24)  # 24 hours in a day
    width = 0.35  # width of bars

    # Daily plot colors
    ax1.bar(x - width/2, ori_ug_hourly_total[day_idx], width, label=r'$\mathcal{P}_{basic}$', color=BLUE)
    ax1.bar(x + width/2, sco_ug_hourly_total[day_idx], width, label=r'$\mathcal{P}_{inf}^{sco}$', color=RED)
    ax1.set_xlabel('Hour of Day', fontweight='normal')
    ax1.set_ylabel('NO. Online Gen.', fontweight='normal', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_xticks(np.arange(0, 24, 4))  # Show every 4 hours
    ax1.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Plot solar curtailment on right y-axis
    ax2 = ax1.twinx()
    # ax2.plot(x, solar_hourly_total[day_idx], color=YELLOW, linestyle='--', linewidth=2, label=r'Solar Power')
    ax2.plot(x, ori_solar_hourly_total[day_idx], color=GREEN, linestyle='--', linewidth=2, label=r'$\mathcal{P}_{basic}$')
    ax2.plot(x, sco_solar_hourly_total[day_idx], color=PURPLE, linestyle='--', linewidth=2, label=r'$\mathcal{P}_{inf}^{sco}$')
    # Create shaded area with label
    ax2.fill_between(x, ori_solar_hourly_total[day_idx], sco_solar_hourly_total[day_idx], alpha=0.2, color=PURPLE, label='Curtail')
    ax2.set_ylabel(r'Solar ($\times 10^2 MW$)', fontweight='normal', color='black')
    ax2.tick_params(axis='y', labelcolor='black')

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1, labels1, loc='upper left', frameon=False, labelspacing=0.05, handlelength=1.5, handletextpad=0.4, borderaxespad=0.2)
    ax2.legend(lines2, labels2, loc='upper right', frameon=False, labelspacing=0.05, handlelength=1.5, handletextpad=0.4, borderaxespad=0.2)

    plt.tight_layout()
    plt.savefig(f'{sco_dir}/day_{day_idx}.pdf', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()