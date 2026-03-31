import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Set consistent style for all plots
rc = {"font.family" : "serif", 
      "mathtext.fontset" : "stix",
      }
plt.rcParams.update(rc)
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rc('font', size=20)

# Define consistent colors for all plots
BLUE = '#2C73D2'
RED = '#FF6666'
GREEN = '#2ac195'
PURPLE = '#8290bb'
YELLOW = '#cda23d'

# old color palette
# BLUE = '#4444FF'
# RED = '#FF4444'
# PURPLE = '#9B4DE0'
# YELLOW = '#FFFF44'
# GREEN = '#44FF44'

data_dir = 'paper_exp/obf_result'

def evaluate_obf_sco(grid_name):
    performance_dict = np.load(data_dir + f'/obf_sco/{grid_name}/performance_dict.npy', allow_pickle=True).item()
    
    print("====== Basic Information ======")
    print("All keys in the performance dict...")
    print(performance_dict.keys())
    print("Shape of performance_true_ori...")
    for key, value in performance_dict['performance_true_ori'].items():
        print(f'{key}: {value.shape}')
    
    total_number = performance_dict['solar'].shape[0]
    print(f'total number of data: {total_number}')

    def evaluate(key, start_idx, end_idx, verbose=False):
        """
        Forecast method list: [true, acc, obj,obj_sco] (key means the forecast method)
        Evaluate on both Pbasic (ori) and Pinf^sco (sco)
        start_idx: start index of the data (in our basis)
        end_idx: end index of the data (in hour basis) = -1 means all data
        """

        load_true = performance_dict[f'load'][start_idx:end_idx]
        solar_true = performance_dict[f'solar'][start_idx:end_idx]
        solar_forecast = performance_dict[f'{key}_forecast'][start_idx:end_idx]
        mape = np.mean(np.abs(solar_forecast - solar_true) / solar_true)
        nrmse = np.mean(
            np.sqrt(
                np.mean((solar_forecast - solar_true) ** 2, axis=0)
                ) / np.std(solar_true, axis=0)
            )
        
        ave_solar_load_ratio = np.mean(np.sum(solar_true, axis=-1) / np.sum(load_true, axis=-1))
        max_solar_load_ratio = np.max(np.sum(solar_true, axis=-1) / np.sum(load_true, axis=-1))
        min_solar_load_ratio = np.min(np.sum(solar_true, axis=-1) / np.sum(load_true, axis=-1))
        ave_solar = np.mean(np.sum(solar_true, axis=-1))
        ave_load = np.mean(np.sum(load_true, axis=-1))

        cost_ori = performance_dict[f"performance_{key}_ori"]["total_cost"][start_idx:end_idx] * 100
        cost_sco = performance_dict[f"performance_{key}_sco"]["total_cost"][start_idx:end_idx] * 100
        cost_ori_mean = np.mean(cost_ori)
        cost_sco_mean = np.mean(cost_sco)
        
        if verbose: 
            print(f'max solar/load: {max_solar_load_ratio}')
            print(f'nrmse: {nrmse}')
            print(f'mape: {mape}')
            print(f'No of samples: {solar_true.shape[0]}')
            print(f'ave cost (ori): {cost_ori_mean}')
            print(f'ave cost (sco): {cost_sco_mean}')
        
        # Hourly
        uc_gscr_ori = performance_dict[f"performance_{key}_ori"]["uc_gscr"][start_idx:end_idx]
        uc_gscr_sco = performance_dict[f"performance_{key}_sco"]["uc_gscr"][start_idx:end_idx]
        rd_gscr_ori = performance_dict[f"performance_{key}_ori"]["rd_gscr"][start_idx:end_idx]
        rd_gscr_sco = performance_dict[f"performance_{key}_sco"]["rd_gscr"][start_idx:end_idx]
        # Daily
        uc_gscr_ori_by_day = uc_gscr_ori.reshape(-1, 24)  # into daily (days, 24)
        uc_gscr_sco_by_day = uc_gscr_sco.reshape(-1, 24)
        rd_gscr_ori_by_day = rd_gscr_ori.reshape(-1, 24)
        rd_gscr_sco_by_day = rd_gscr_sco.reshape(-1, 24)
        
        no_hour, no_day = uc_gscr_ori.shape[0], uc_gscr_ori_by_day.shape[0]
        
        # Unstable rate
        uc_ur_ori_hourly = np.sum(uc_gscr_ori < 2.5) / no_hour * 100
        uc_ur_ori_daily = np.sum(np.sum(uc_gscr_ori_by_day < 2.5, axis=-1) > 0) / no_day * 100
        
        rd_ur_ori_hourly = np.sum(rd_gscr_ori < 2.5) / no_hour * 100
        rd_ur_ori_daily = np.sum(np.sum(rd_gscr_ori_by_day < 2.5, axis=-1) > 0) / no_day * 100
        
        uc_ur_sco_hourly = np.sum(uc_gscr_sco < 2.5) / no_hour * 100
        uc_ur_sco_daily = np.sum(np.sum(uc_gscr_sco_by_day < 2.5, axis=-1) > 0) / no_day * 100
        
        rd_ur_sco_hourly = np.sum(rd_gscr_sco < 2.5) / no_hour * 100
        rd_ur_sco_daily = np.sum(np.sum(rd_gscr_sco_by_day < 2.5, axis=-1) > 0) / no_day * 100
        
        if verbose: 
            print('ORI Optimization: ')
            print(f'uc gscr hourly: {uc_ur_ori_hourly}')
            print(f'rd gscr hourly: {rd_ur_ori_hourly}')
            print(f'uc gscr daily: {uc_ur_ori_daily}')
            print(f'rd gscr daily: {rd_ur_ori_daily}')
            
            print('SCO Optimization: ')
            print(f'uc gscr hourly: {uc_ur_sco_hourly}')
            print(f'rd gscr hourly: {rd_ur_sco_hourly}')
            print(f'uc gscr daily: {uc_ur_sco_daily}')
            print(f'rd gscr daily: {rd_ur_sco_daily}')

        # Summarize into a dictionary
        summary_dict = {
            'solar_true': solar_true, 'load_true': load_true,
            'solar_forecast': solar_forecast,
            'solar_ave': ave_solar, 'load_ave': ave_load,
            'cost_ori': cost_ori, 'cost_sco': cost_sco,
            'cost_ori_mean': cost_ori_mean, 'cost_sco_mean': cost_sco_mean,
            'solar_mape': mape, 'solar_nrmse': nrmse,
            'uc_ur_ori_hourly': uc_ur_ori_hourly, 'uc_ur_ori_daily': uc_ur_ori_daily,
            'rd_ur_ori_hourly': rd_ur_ori_hourly, 'rd_ur_ori_daily': rd_ur_ori_daily,
            'uc_ur_sco_hourly': uc_ur_sco_hourly, 'uc_ur_sco_daily': uc_ur_sco_daily,
            'rd_ur_sco_hourly': rd_ur_sco_hourly, 'rd_ur_sco_daily': rd_ur_sco_daily
        }
        
        return summary_dict

    # Total performance
    total_performance = {}
    for key in ['true', 'acc', 'obj', 'obj_sco']:
        print(f'====== Evaluating {key} ======')
        total_performance[key] = evaluate(key, 0, total_number, verbose=True)
    
    # Per sample cost performance
    print("====== Per Sample Cost Performance against True ======")
    abf_cost_ori = total_performance['acc']['cost_ori']
    abf_cost_sco = total_performance['acc']['cost_sco']
    obf_cost_ori = total_performance['obj']['cost_ori']
    obf_cost_sco = total_performance['obj']['cost_sco']
    obf_sco_cost_ori = total_performance['obj_sco']['cost_ori']
    obf_sco_cost_sco = total_performance['obj_sco']['cost_sco']
    true_cost_ori = total_performance['true']['cost_ori']
    true_cost_sco = total_performance['true']['cost_sco']

    abf_cost_ori_ratio = abf_cost_ori / true_cost_ori
    abf_cost_sco_ratio = abf_cost_sco / true_cost_sco
    obf_cost_ori_ratio = obf_cost_ori / true_cost_ori
    obf_cost_sco_ratio = obf_cost_sco / true_cost_sco
    obf_sco_cost_ori_ratio = obf_sco_cost_ori / true_cost_ori
    obf_sco_cost_sco_ratio = obf_sco_cost_sco / true_cost_sco

    # Print out statistics
    print('ABF Cost Ori Ratio: Mean {:.4f}, Std {:.4f}, Min {:.4f}, Max {:.4f}'.format(
        np.mean(abf_cost_ori_ratio), np.std(abf_cost_ori_ratio), np.min(abf_cost_ori_ratio), np.max(abf_cost_ori_ratio)))
    print('ABF Cost SCO Ratio: Mean {:.4f}, Std {:.4f}, Min {:.4f}, Max {:.4f}'.format(
        np.mean(abf_cost_sco_ratio), np.std(abf_cost_sco_ratio), np.min(abf_cost_sco_ratio), np.max(abf_cost_sco_ratio)))
    print('OBF Cost Ori Ratio: Mean {:.4f}, Std {:.4f}, Min {:.4f}, Max {:.4f}'.format(
        np.mean(obf_cost_ori_ratio), np.std(obf_cost_ori_ratio), np.min(obf_cost_ori_ratio), np.max(obf_cost_ori_ratio)))
    print('OBF Cost SCO Ratio: Mean {:.4f}, Std {:.4f}, Min {:.4f}, Max {:.4f}'.format(
        np.mean(obf_cost_sco_ratio), np.std(obf_cost_sco_ratio), np.min(obf_cost_sco_ratio), np.max(obf_cost_sco_ratio)))
    print('OBF/SCO Cost Ori Ratio: Mean {:.4f}, Std {:.4f}, Min {:.4f}, Max {:.4f}'.format(
        np.mean(obf_sco_cost_ori_ratio), np.std(obf_sco_cost_ori_ratio), np.min(obf_sco_cost_ori_ratio), np.max(obf_sco_cost_ori_ratio)))
    print('OBF/SCO Cost SCO Ratio: Mean {:.4f}, Std {:.4f}, Min {:.4f}, Max {:.4f}'.format(
        np.mean(obf_sco_cost_sco_ratio), np.std(obf_sco_cost_sco_ratio), np.min(obf_sco_cost_sco_ratio), np.max(obf_sco_cost_sco_ratio)))

    # Draw the scatter plots
    # On obf/basic: cost vs solar forecast error (obf tends to under forecast)
    # Calculate net load differences and costs
    abf_forecast_err = np.sum(total_performance['acc']['solar_true'], axis=-1) - np.sum(total_performance['acc']['solar_forecast'], axis=-1)
    abf_cost = total_performance['acc']['cost_ori']
    obf_forecast_err = np.sum(total_performance['obj']['solar_true'], axis=-1) - np.sum(total_performance['obj']['solar_forecast'], axis=-1)
    obf_cost = total_performance['obj']['cost_ori']
    
    # Create scatter plot
    fig, ax = plt.subplots(figsize=(8, 4))
    start_idx_ = 24*7*30  # for a single summer week
    end_idx_ = 24*7*31
    
    # Plot scatter points
    ax.scatter(abf_forecast_err[start_idx_:end_idx_], abf_cost[start_idx_:end_idx_]/100, 
              label=r'$\mathcal{M}_{abf}$', color=RED, alpha=0.6, s=50)
    ax.scatter(obf_forecast_err[start_idx_:end_idx_], obf_cost[start_idx_:end_idx_]/100, 
              label=r'$\mathcal{M}_{obf}$', color=BLUE, alpha=0.6, s=50)
    
    # Add mean points
    abf_mean_x = np.mean(abf_forecast_err[start_idx_:end_idx_])
    abf_mean_y = np.mean(abf_cost[start_idx_:end_idx_]/100)
    obf_mean_x = np.mean(obf_forecast_err[start_idx_:end_idx_])
    obf_mean_y = np.mean(obf_cost[start_idx_:end_idx_]/100)
    
    ax.scatter(abf_mean_x, abf_mean_y, color=RED, marker='*', s=400, 
              label=r'$\mathcal{M}_{abf}$ (mean)', edgecolor='black', linewidth=1)
    ax.scatter(obf_mean_x, obf_mean_y, color=BLUE, marker='*', s=400, 
              label=r'$\mathcal{M}_{obf}$ (mean)', edgecolor='black', linewidth=1)
    
    # Add arrow between means
    ax.annotate('', xy=(obf_mean_x, obf_mean_y), xytext=(abf_mean_x, abf_mean_y),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    # Customize plot
    ax.set_xlabel(r'Aggregated Forecast Error ($\times100MW$)')
    ax.set_xlim(-0.45, 0.45)
    ax.set_ylim(3, 13)
    ax.set_ylabel(r'PSO Cost ($\times10^2$£)')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(framealpha=0.0, edgecolor='black', loc='lower left', ncol=1, handletextpad=0.1, columnspacing=0.5, labelspacing=0.1)
    
    plt.tight_layout()
    plt.savefig(data_dir + f'/obf_sco/{grid_name}/scatter_plot.pdf', dpi=600, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    """
    Draw the seasonal performance comparison between the three methods
    """
    # Each method build a dictionary, Each entry is a list of seasonal performance
    true_performance = {
        'solar_ave': [], 'load_ave': [],
        'solar_mape': [], 'solar_nrmse': [],
        'cost_ori_mean': [], 'cost_sco_mean': [],
        'uc_ur_ori_hourly': [], 'uc_ur_ori_daily': [],
        'rd_ur_ori_hourly': [], 'rd_ur_ori_daily': [],
        'uc_ur_sco_hourly': [], 'uc_ur_sco_daily': [],
        'rd_ur_sco_hourly': [], 'rd_ur_sco_daily': []
    }
    acc_performance = deepcopy(true_performance)
    obj_performance = deepcopy(true_performance)
    obj_sco_performance = deepcopy(true_performance)

    all_performance = {
        'true': true_performance, 'acc': acc_performance,
        'obj': obj_performance, 'obj_sco': obj_sco_performance
    }
    
    # Group by three months
    step = 30*24*3 # group by three months

    for i in range(4): # 4 seasons (roughly)
        start_idx = i * step
        end_idx = start_idx + step
        for key in ['true', 'acc', 'obj', 'obj_sco']:
            performance = evaluate(key, start_idx, end_idx, verbose=False)
            for k, v in performance.items():
                if k in all_performance[key].keys():
                    all_performance[key][k].append(v)
                    
    # Convert all_performance to numpy arrays
    for key in all_performance:
        for k in all_performance[key]:
            all_performance[key][k] = np.array(all_performance[key][k])
            
    # Calculate relative cost differences against the true solar performance, with and without stability constraints
    # To assess if the performance is consistent across seasons (higher vs lower solar generation)
    acc_ori_diff = (all_performance['acc']['cost_ori_mean'] - all_performance['true']['cost_ori_mean'])/all_performance['true']['cost_ori_mean']
    acc_sco_diff = (all_performance['acc']['cost_sco_mean'] - all_performance['true']['cost_sco_mean'])/all_performance['true']['cost_sco_mean']
    obj_ori_diff = (all_performance['obj']['cost_ori_mean'] - all_performance['true']['cost_ori_mean'])/all_performance['true']['cost_ori_mean']
    obj_sco_diff = (all_performance['obj']['cost_sco_mean'] - all_performance['true']['cost_sco_mean'])/all_performance['true']['cost_sco_mean']
    obj_sco_ori_diff = (all_performance['obj_sco']['cost_ori_mean'] - all_performance['true']['cost_ori_mean'])/all_performance['true']['cost_ori_mean']
    obj_sco_sco_diff = (all_performance['obj_sco']['cost_sco_mean'] - all_performance['true']['cost_sco_mean'])/all_performance['true']['cost_sco_mean']
    solar = all_performance['true']['solar_ave']
    load = all_performance['true']['load_ave']

    # Create figure for evaluating original case
    fig, ax1 = plt.subplots(figsize=(5, 4))
    x = np.arange(4)
    width = 0.2

    # Plot bars
    ax1.bar(x - width, acc_ori_diff * 100, width, label=r'$\mathcal{M}_{abf}$', color=RED)
    ax1.bar(x, obj_ori_diff * 100, width, label=r'$\mathcal{M}_{obf}$', color=BLUE)
    ax1.bar(x + width, obj_sco_ori_diff * 100, width, label=r'$\mathcal{M}_{obf/sco}$', color=PURPLE)

    # Customize plot
    ax1.set_ylabel('Rel. Cost Diff (%)')
    ax1.set_ylim(0, 15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4'])
    ax1.grid(True, alpha=0.2, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(data_dir + f'/obf_sco/{grid_name}/seasonal_performance_ori.pdf', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Create figure for evaluating SCO case
    fig, ax1 = plt.subplots(figsize=(5, 4))
    x = np.arange(4)
    width = 0.2

    # Plot bars
    ax1.bar(x - width, acc_sco_diff * 100, width, label=r'$\mathcal{M}_{abf}$', color=RED)
    ax1.bar(x, obj_sco_diff * 100, width, label=r'$\mathcal{M}_{obf}$', color=BLUE)
    ax1.bar(x + width, obj_sco_sco_diff * 100, width, label=r'$\mathcal{M}_{obf/sco}$', color=PURPLE)

    # Customize plot
    ax1.set_ylabel('Rel. Cost Diff (%)')
    ax1.set_ylim(0, 15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4'])
    ax1.grid(True, alpha=0.2, linestyle='--')
    ax1.legend(framealpha=0.0, edgecolor='black', loc='upper right', ncol=1, handletextpad=0.1, columnspacing=0.5, labelspacing=0.1)
    
    plt.tight_layout()
    plt.savefig(data_dir + f'/obf_sco/{grid_name}/seasonal_performance_sco.pdf', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

def plot_uncertainty_performance_multi():
    # Plot the performance of the multi-uncertainty case
    
    data_dir = f'paper_exp/obf_result/obf_uncer_multi/bus14/'
    uncertainty_budget = '0.05_0.1_0.05'
    performance_dict = np.load(data_dir + f'{uncertainty_budget}.npy', allow_pickle=True).item()
    
    cost_abf = performance_dict['cost_acc'] * 100
    cost_abf_ml = performance_dict['worst_cost_acc_input'] * 100
    cost_abf_opt = performance_dict['worst_cost_acc'] * 100
    cost_abf_multi = performance_dict['worst_cost_acc_multi'] * 100
    
    cost_obf = performance_dict['cost_obj'] * 100
    cost_obf_ml = performance_dict['worst_cost_obj_input'] * 100
    cost_obf_opt = performance_dict['worst_cost_obj'] * 100
    cost_obf_multi = performance_dict['worst_cost_obj_multi'] * 100
    
    print('ABF Cost: ', np.mean(cost_abf), 'ML Cost: ', np.mean(cost_abf_ml), 'Opt Cost: ', np.mean(cost_abf_opt), 'Multi Cost: ', np.mean(cost_abf_multi))
    print('OBF Cost: ', np.mean(cost_obf), 'ML Cost: ', np.mean(cost_obf_ml), 'Opt Cost: ', np.mean(cost_obf_opt), 'Multi Cost: ', np.mean(cost_obf_multi))

    means_abf = [np.mean(cost_abf), np.mean(cost_abf_ml), np.mean(cost_abf_opt), np.mean(cost_abf_multi)]
    means_obf = [np.mean(cost_obf), np.mean(cost_obf_ml), np.mean(cost_obf_opt), np.mean(cost_obf_multi)]
    y_max = max(max(means_abf), max(means_obf)) / 100 * 1.15  # shared upper limit with margin for labels

    # Bar plot for ABF
    fig, ax = plt.subplots(figsize=(6, 4))
    labels_abf = ['Nominal', 'ML', 'Opt', 'Multi']
    x_abf = np.arange(len(labels_abf))
    bars_abf = ax.bar(x_abf, np.array(means_abf) / 100, color=[BLUE, RED, GREEN, PURPLE], width=0.4)
    ax.set_ylabel(r'PSO Cost ($\times10^2$£)')
    ax.set_ylim(0, y_max)
    ax.set_xticks(x_abf)
    ax.set_xticklabels(labels_abf)
    ax.grid(True, alpha=0.3)
    for rect in bars_abf:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=14)
    plt.tight_layout()
    plt.savefig(data_dir + f'bar_plot_abf_{uncertainty_budget}.pdf', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Bar plot for OBF
    fig, ax = plt.subplots(figsize=(6, 4))
    labels_obf = ['Nominal', 'ML', 'Opt', 'Multi']
    x_obf = np.arange(len(labels_obf))
    bars_obf = ax.bar(x_obf, np.array(means_obf) / 100, color=[BLUE, RED, GREEN, PURPLE], width=0.4)
    ax.set_ylabel(r'PSO Cost ($\times10^2$£)')
    ax.set_ylim(0, y_max)
    ax.set_xticks(x_obf)
    ax.set_xticklabels(labels_obf)
    ax.grid(True, alpha=0.3)
    for rect in bars_obf:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=14)
    plt.tight_layout()
    plt.savefig(data_dir + f'bar_plot_obf_{uncertainty_budget}.pdf', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()


def plot_uncertainty_performance(grid_name):
    # Load data
    data_dir = f'paper_exp/obf_result/obf_uncer/{grid_name}/' 
    # uncertainty_budget = [0.01, 0.03, 0.05, 0.07]
    # budget_percentage = ['1%', '3%', '5%', '7%']

    if grid_name == 'bus14':
        uncertainty_budget = [0.03, 0.05, 0.07]
        budget_percentage = ['3%', '5%', '7%']
        fig_size = (8, 4)
        rotation = 0
        with_percentage = True
    else:
        uncertainty_budget = [0.05]
        budget_percentage = ['5%']
        fig_size = (6, 4)
        rotation = 0
        with_percentage = False

    for idx, budget in enumerate(uncertainty_budget):
        # Load data for each budget
        performance_dict = np.load(data_dir + f'{budget}.npy', allow_pickle=True).item()
        
        cost_true = performance_dict['cost_true'] * 100
        cost_abf = performance_dict['cost_acc'] * 100 
        cost_obf = performance_dict['cost_obj'] * 100
        cost_obf_robust = performance_dict['cost_robust'] * 100

        worst_cost_true = performance_dict['worst_cost_true'] * 100
        worst_cost_abf = performance_dict['worst_cost_acc'] * 100
        worst_cost_obf = performance_dict['worst_cost_obj'] * 100
        worst_cost_obf_robust = performance_dict['worst_cost_robust'] * 100

        # Create figure
        fig, ax = plt.subplots(figsize=fig_size)

        # Set up data
        if with_percentage:
            labels = ['True', r'$\mathcal{M}_{abf}$', r'$\mathcal{M}_{obf}$', r'$\mathcal{M}_{obf/uncer}$('+budget_percentage[idx]+')']
        else:
            labels = ['True', r'$\mathcal{M}_{abf}$', r'$\mathcal{M}_{obf}$', r'$\mathcal{M}_{obf/uncer}$']
        
        original_costs = [np.mean(cost_true), np.mean(cost_abf), np.mean(cost_obf), np.mean(cost_obf_robust)]
        worst_costs = [np.mean(worst_cost_true), np.mean(worst_cost_abf), np.mean(worst_cost_obf), np.mean(worst_cost_obf_robust)]

        print(f'=== Uncertainty Budget: {budget} ===')
        print('Original Costs: ', original_costs)
        print('Worst-case Costs: ', worst_costs)

        obf_time = performance_dict['obj_time']
        ccg_time = performance_dict['robust_time']

        print('OBF Time: ', np.mean(obf_time), 'CCG Time: ', np.mean(ccg_time))

        x = np.arange(len(labels))
        width = 0.35

        # Create bars
        rects1 = ax.bar(x, np.array(worst_costs) / 100, width, label='Worst', color=RED)
        rects2 = ax.bar(x, np.array(original_costs) / 100, width, label='Nominal', color=BLUE)

        # Set up ylim
        if grid_name == 'bus14':
            ylim = 12
        else:
            ylim = np.max(worst_costs) / 100 * 1.6

        # Customize plot
        ax.set_ylabel(r'PSO Cost ($\times10^2$£)')
        # ax.set_title(f'Uncertainty Budget = {budget_percentage[idx]}')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=rotation)
        ax.set_ylim(0, ylim)
        ax.grid(True, alpha=0.3)
        
        if idx == 0:
            ax.legend(framealpha=0.0, edgecolor='black', loc='upper right', ncol=1, handletextpad=0.1, columnspacing=0.5, labelspacing=0.1)

        # Add value labels
        def autolabel(rects, offset=0):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, offset),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=14)

        autolabel(rects1, offset=5)
        autolabel(rects2, offset=-1)

        plt.tight_layout()
        plt.savefig(data_dir + f'/{grid_name}_uncertainty_performance_{idx}.pdf', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()

def draw_sensitivity(grid_name):
    """
    Sensitivity analysis for ABF, OBF/Basic, OBF/SCO
    """
    data_dir = 'paper_exp/obf_result/'
    performance = np.load(data_dir + f'obf_sco/obf_sco_grad/{grid_name}/performance.npy', allow_pickle=True).item()
    
    # Extract data
    # cost_acc = performance['cost_acc']
    cost_obj = performance['cost_obj']
    cost_obj_sco = performance['cost_obj_sco']
    
    grad_acc = np.concatenate([performance['grad_W_acc'], performance['grad_b_acc']], axis=-1)
    grad_obj = np.concatenate([performance['grad_W_obj'], performance['grad_b_obj']], axis=-1)
    grad_obj_sco = np.concatenate([performance['grad_W_obj_sco'], performance['grad_b_obj_sco']], axis=-1)
    
    grad_acc_nonzero_idx = np.where(np.linalg.norm(grad_acc, axis=-1) != 0)[0]
    grad_obj_nonzero_idx = np.where(np.linalg.norm(grad_obj, axis=-1) != 0)[0]
    grad_obj_sco_nonzero_idx = np.where(np.linalg.norm(grad_obj_sco, axis=-1) != 0)[0]
    
    cos_sim_acc_obj = np.sum(grad_acc * grad_obj, axis=-1) / (np.linalg.norm(grad_acc, axis=-1) * np.linalg.norm(grad_obj, axis=-1))
    cos_sim_obj_obj_sco = np.sum(grad_obj * grad_obj_sco, axis=-1) / (np.linalg.norm(grad_obj, axis=-1) * np.linalg.norm(grad_obj_sco, axis=-1))
    cos_sim_acc_obj_sco = np.sum(grad_acc * grad_obj_sco, axis=-1) / (np.linalg.norm(grad_acc, axis=-1) * np.linalg.norm(grad_obj_sco, axis=-1))
    
    print("Overall Performance:")
    # print('ABF Cost: ', np.mean(cost_acc), 'OBF/Basic Cost: ', np.mean(cost_obj), 'OBF/SCO Cost: ', np.mean(cost_obj_sco))
    print('OBF/Basic Cost: ', np.mean(cost_obj), 'OBF/SCO Cost: ', np.mean(cost_obj_sco))
    print('ABF-OBF/Basic Cosine Similarity: ', np.mean(cos_sim_acc_obj[grad_acc_nonzero_idx]))
    print('OBF/Basic-OBF/SCO Cosine Similarity: ', np.mean(cos_sim_obj_obj_sco[grad_acc_nonzero_idx]))
    print('ABF-OBF/SCO Cosine Similarity: ', np.mean(cos_sim_acc_obj_sco[grad_acc_nonzero_idx]))
    
    print("Seasonal Performance:")
    step_size = 30*24*3
    ABF_OBF_BASIC_COS_SIM = []
    OBF_BASIC_OBF_SCO_COS_SIM = []
    ABF_OBF_SCO_COS_SIM = []
    
    for i in range(4):
        start_idx = i * step_size
        end_idx = np.min([start_idx + step_size, len(cost_obj)])
        # print(f'Season {i+1}:')
        # # print('ABF Cost: ', np.mean(cost_acc[start_idx:end_idx]), 'OBF/Basic Cost: ', np.mean(cost_obj[start_idx:end_idx]), 'OBF/SCO Cost: ', np.mean(cost_obj_sco[start_idx:end_idx]))
        # print('ABF-OBF/Basic Cosine Similarity: ', np.mean(cos_sim_acc_obj[start_idx:end_idx]))
        # print('OBF/Basic-OBF/SCO Cosine Similarity: ', np.mean(cos_sim_obj_obj_sco[start_idx:end_idx]))
        # print('ABF-OBF/SCO Cosine Similarity: ', np.mean(cos_sim_acc_obj_sco[start_idx:end_idx]))
        cos_sim_acc_obj_nonnan_idx = np.where(~np.isnan(cos_sim_acc_obj[start_idx:end_idx]))[0]
        ABF_OBF_BASIC_COS_SIM.append(np.mean(cos_sim_acc_obj[start_idx:end_idx][cos_sim_acc_obj_nonnan_idx]))
        OBF_BASIC_OBF_SCO_COS_SIM.append(np.mean(cos_sim_obj_obj_sco[start_idx:end_idx][cos_sim_acc_obj_nonnan_idx]))
        ABF_OBF_SCO_COS_SIM.append(np.mean(cos_sim_acc_obj_sco[start_idx:end_idx][cos_sim_acc_obj_nonnan_idx]))
    
    ABF_OBF_BASIC_Pearsonr = []
    OBF_BASIC_OBF_SCO_Pearsonr = []
    ABF_OBF_SCO_Pearsonr = []
    for i in range(4):
        start_idx = i * step_size
        end_idx = np.min([start_idx + step_size, len(cost_obj)])
        ABF_OBF_BASIC_Pearsonr_ = []
        OBF_BASIC_OBF_SCO_Pearsonr_ = []
        ABF_OBF_SCO_Pearsonr_ = []
        for j in range(start_idx, end_idx):
            if np.linalg.norm(grad_acc[j]) != 0:
                ABF_OBF_BASIC_Pearsonr_.append(pearsonr(grad_acc[j], grad_obj[j])[0])
                OBF_BASIC_OBF_SCO_Pearsonr_.append(pearsonr(grad_obj[j], grad_obj_sco[j])[0])
                ABF_OBF_SCO_Pearsonr_.append(pearsonr(grad_acc[j], grad_obj_sco[j])[0])
        ABF_OBF_BASIC_Pearsonr.append(np.mean(ABF_OBF_BASIC_Pearsonr_))
        OBF_BASIC_OBF_SCO_Pearsonr.append(np.mean(OBF_BASIC_OBF_SCO_Pearsonr_))
        ABF_OBF_SCO_Pearsonr.append(np.mean(ABF_OBF_SCO_Pearsonr_))
    
    print("Average Cosine Similarity:")
    print('ABF-OBF/Basic: ', ABF_OBF_BASIC_COS_SIM)
    print('OBF/Basic-OBF/SCO: ', OBF_BASIC_OBF_SCO_COS_SIM)
    print('ABF-OBF/SCO: ', ABF_OBF_SCO_COS_SIM)
    
    print("Average Pearsonr:")
    print('ABF-OBF/Basic: ', ABF_OBF_BASIC_Pearsonr)
    print('OBF/Basic-OBF/SCO: ', OBF_BASIC_OBF_SCO_Pearsonr)
    print('ABF-OBF/SCO: ', ABF_OBF_SCO_Pearsonr)
    
    # Plot bar plots
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(4)
    width = 0.25
    
    # Create bars
    rects1 = ax.bar(x - width, ABF_OBF_BASIC_COS_SIM, width, label=r'$\mathcal{M}_{abf}$ vs $\mathcal{M}_{obf}$', color=RED)
    rects2 = ax.bar(x, OBF_BASIC_OBF_SCO_COS_SIM, width, label=r'$\mathcal{M}_{obf}$ vs $\mathcal{M}_{obf/sco}$', color=BLUE)
    rects3 = ax.bar(x + width, ABF_OBF_SCO_COS_SIM, width, label=r'$\mathcal{M}_{abf}$ vs $\mathcal{M}_{obf/sco}$', color='#E6B3FF')
    
    # Customize plot
    ax.set_ylabel('Cosine Similarity')
    ax.set_xticks(x)
    ax.set_ylim(0, 1)
    ax.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4'])
    ax.legend(framealpha=0.0, edgecolor='black', bbox_to_anchor=(1.02, 1), loc='upper left', ncol=1, handletextpad=0.1, columnspacing=0.5, labelspacing=0.1, 
              handlelength=0.5)
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(data_dir + '/cosine_similarity.pdf', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(4)
    width = 0.25
    
    # Create bars
    rects1 = ax.bar(x - width, ABF_OBF_BASIC_Pearsonr, width, label=r'$\mathcal{M}_{abf}$ vs $\mathcal{M}_{obf}$', color=RED)
    rects2 = ax.bar(x, OBF_BASIC_OBF_SCO_Pearsonr, width, label=r'$\mathcal{M}_{obf}$ vs $\mathcal{M}_{obf/sco}$', color=BLUE)
    rects3 = ax.bar(x + width, ABF_OBF_SCO_Pearsonr, width, label=r'$\mathcal{M}_{abf}$ vs $\mathcal{M}_{obf/sco}$', color='#E6B3FF')
    
    # Customize plot    
    ax.set_ylabel("Pearson r")
    ax.set_xticks(x)
    ax.set_ylim(0, 1)
    ax.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4'])
    ax.legend(framealpha=0.0, edgecolor='black', bbox_to_anchor=(1.02, 1), loc='upper left', ncol=1, handletextpad=0.1, columnspacing=0.5, labelspacing=0.1, 
              handlelength=0.5)
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(data_dir + '/pearsonr.pdf', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    
if __name__ == '__main__':
    # import argparse
    # parser = argparse.ArgumentParser()
    # # parser.add_argument('--grid', type=str, default='bus14', help='Grid name: bus14')
    # args = parser.parse_args()
    # grid_name = args.grid

    # evaluate_obf_sco(grid_name=grid_name)
    # plot_uncertainty_performance(grid_name=grid_name)
    # draw_sensitivity(grid_name=grid_name)
    # plot_uncertainty_performance_multi()
    
    # evaluate_obf_sco(grid_name="bus14")
    for grid_name_ in ['bus14', 'bus39', 'bus57']:
        plot_uncertainty_performance(grid_name=grid_name_)
    # draw_sensitivity(grid_name="bus14")
    # plot_uncertainty_performance_multi()