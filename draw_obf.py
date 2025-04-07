import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

# Set consistent style for all plots
rc = {"font.family" : "serif", 
      "mathtext.fontset" : "stix",
      }
plt.rcParams.update(rc)
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rc('font', size=20)

# Define consistent colors for all plots
BLUE = '#4444FF'
RED = '#FF4444'
GREEN = '#44FF44'
PURPLE = '#FF44FF'
YELLOW = '#FFFF44'

data_dir = 'paper_exp/obf_result_1'

def evaluate_obf_sco():
    performance_dict = np.load(data_dir + '/obf_sco/performance_dict.npy', allow_pickle=True).item()
    
    print("All keys in the performance dict...")
    print(performance_dict.keys())
    print("Shape of performance_true_ori...")
    for key, value in performance_dict['performance_true_ori'].items():
        print(f'{key}: {value.shape}')
    
    total_number = performance_dict['solar'].shape[0]
    print(f'total number of data: {total_number}')

    """
    Forecast method list: [true, acc, obj,obj_sco]
    start_idx: start index of the data (in our basis)
    end_idx: end index of the data (in hour basis) = -1 means all data
    """
    def evaluate(key, start_idx, end_idx, verbose=False):
        # key means the forecast method
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
        uc_gscr_ori_by_day = uc_gscr_ori.reshape(-1, 24)  # into daily
        uc_gscr_sco_by_day = uc_gscr_sco.reshape(-1, 24)
        rd_gscr_ori_by_day = rd_gscr_ori.reshape(-1, 24)
        rd_gscr_sco_by_day = rd_gscr_sco.reshape(-1, 24)
        
        no_hour, no_day = uc_gscr_ori.shape[0], uc_gscr_ori_by_day.shape[0]
        
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

    """
    Total performance
    """
    print(f'===============>Evaluating total performance...')
    total_performance = {}
    for key in ['true', 'acc', 'obj', 'obj_sco']:
        print(f'===============>Evaluating {key}...')
        total_performance[key] = evaluate(key, 0, total_number, verbose=True)
    
    """
    Draw the scatter plots
    """
    # Calculate net load differences and costs
    abf_net_load = np.sum(total_performance['acc']['solar_true'], axis=-1) - np.sum(total_performance['acc']['solar_forecast'], axis=-1)
    abf_cost = total_performance['acc']['cost_ori']
    obf_net_load = np.sum(total_performance['obj']['solar_true'], axis=-1) - np.sum(total_performance['obj']['solar_forecast'], axis=-1)
    obf_cost = total_performance['obj']['cost_ori']
    
    # Create scatter plot
    fig, ax = plt.subplots(figsize=(8, 4))
    start_idx_ = 24*7*30
    end_idx_ = 24*7*31
    
    # Plot scatter points
    ax.scatter(abf_net_load[start_idx_:end_idx_], abf_cost[start_idx_:end_idx_]/100, 
              label=r'$\mathcal{P}_{train}^{abf}$', color=RED, alpha=0.6, s=50)
    ax.scatter(obf_net_load[start_idx_:end_idx_], obf_cost[start_idx_:end_idx_]/100, 
              label=r'$\mathcal{P}_{train}^{obf}$', color=BLUE, alpha=0.6, s=50)
    
    # Add mean points
    abf_mean_x = np.mean(abf_net_load[start_idx_:end_idx_])
    abf_mean_y = np.mean(abf_cost[start_idx_:end_idx_]/100)
    obf_mean_x = np.mean(obf_net_load[start_idx_:end_idx_])
    obf_mean_y = np.mean(obf_cost[start_idx_:end_idx_]/100)
    
    ax.scatter(abf_mean_x, abf_mean_y, color=RED, marker='*', s=400, 
              label=r'$\mathcal{P}_{train}^{abf}$', edgecolor='black', linewidth=1)
    ax.scatter(obf_mean_x, obf_mean_y, color=BLUE, marker='*', s=400, 
              label=r'$\mathcal{P}_{train}^{obf}$', edgecolor='black', linewidth=1)
    
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
    plt.savefig(data_dir + '/scatter_plot.pdf', dpi=600, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    """
    Plot seasonal performance
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
            
    # Calculate relative cost differences
    acc_ori_diff = (all_performance['acc']['cost_ori_mean'] - all_performance['true']['cost_ori_mean'])/all_performance['true']['cost_ori_mean']
    acc_sco_diff = (all_performance['acc']['cost_sco_mean'] - all_performance['true']['cost_sco_mean'])/all_performance['true']['cost_sco_mean']
    obj_ori_diff = (all_performance['obj']['cost_ori_mean'] - all_performance['true']['cost_ori_mean'])/all_performance['true']['cost_ori_mean']
    obj_sco_diff = (all_performance['obj']['cost_sco_mean'] - all_performance['true']['cost_sco_mean'])/all_performance['true']['cost_sco_mean']
    obj_sco_ori_diff = (all_performance['obj_sco']['cost_ori_mean'] - all_performance['true']['cost_ori_mean'])/all_performance['true']['cost_ori_mean']
    obj_sco_sco_diff = (all_performance['obj_sco']['cost_sco_mean'] - all_performance['true']['cost_sco_mean'])/all_performance['true']['cost_sco_mean']
    solar = all_performance['true']['solar_ave']
    load = all_performance['true']['load_ave']

    # Create figure for original case
    fig, ax1 = plt.subplots(figsize=(8, 4))
    x = np.arange(4)
    width = 0.2

    # Plot bars
    ax1.bar(x - width, acc_ori_diff * 100, width, label=r'$\mathcal{P}_{train}^{abf}$', color=RED)
    ax1.bar(x, obj_ori_diff * 100, width, label=r'$\mathcal{P}_{train}^{obf}$', color=BLUE)
    ax1.bar(x + width, obj_sco_ori_diff * 100, width, label=r'$\mathcal{P}_{train}^{obf/sco}$', color=PURPLE)

    # Customize plot
    ax1.set_ylabel('Rel. Cost Diff (%)')
    ax1.set_ylim(0, 15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4'])
    ax1.grid(True, alpha=0.2, linestyle='--')
    ax1.legend(framealpha=0.0, edgecolor='black', loc='upper right', ncol=1, handletextpad=0.1, columnspacing=0.5, labelspacing=0.1)
    
    plt.tight_layout()
    plt.savefig(data_dir + '/seasonal_performance_ori.pdf', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Create figure for SCO case
    fig, ax1 = plt.subplots(figsize=(8, 4))
    x = np.arange(4)
    width = 0.2

    # Plot bars
    ax1.bar(x - width, acc_sco_diff * 100, width, label=r'$\mathcal{P}_{train}^{abf}$', color=RED)
    ax1.bar(x, obj_sco_diff * 100, width, label=r'$\mathcal{P}_{train}^{obf}$', color=BLUE)
    ax1.bar(x + width, obj_sco_sco_diff * 100, width, label=r'$\mathcal{P}_{train}^{obf/sco}$', color=PURPLE)

    # Customize plot
    ax1.set_ylabel('Rel. Cost Diff (%)')
    ax1.set_ylim(0, 10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4'])
    ax1.grid(True, alpha=0.2, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(data_dir + '/seasonal_performance_sco.pdf', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    
def plot_uncertainty_performance():
    # Load data
    data_dir = 'paper_exp/obf_result_1/'
    # uncertainty_budget = [0.01, 0.03, 0.05, 0.07]
    # budget_percentage = ['1%', '3%', '5%', '7%']

    uncertainty_budget = [0.03, 0.05, 0.07]
    budget_percentage = ['3%', '5%', '7%']

    for idx, budget in enumerate(uncertainty_budget):
        # Load data for each budget
        performance_dict = np.load(data_dir + f'obf_uncer/{budget}.npy', allow_pickle=True).item()
        
        cost_true = performance_dict['cost_true'] * 100
        cost_abf = performance_dict['cost_acc'] * 100 
        cost_obf = performance_dict['cost_obj'] * 100
        cost_obf_robust = performance_dict['cost_robust'] * 100

        worst_cost_true = performance_dict['worst_cost_true'] * 100
        worst_cost_abf = performance_dict['worst_cost_acc'] * 100
        worst_cost_obf = performance_dict['worst_cost_obj'] * 100
        worst_cost_obf_robust = performance_dict['worst_cost_robust'] * 100

        # Create figure
        fig, ax = plt.subplots(figsize=(8, 4))

        # Set up data
        labels = ['True', r'$\mathcal{P}_{train}^{abf}$', r'$\mathcal{P}_{train}^{obf}$', r'$\mathcal{P}_{train}^{obf/uncer}$('+budget_percentage[idx]+')']
        original_costs = [np.mean(cost_true), np.mean(cost_abf), np.mean(cost_obf), np.mean(cost_obf_robust)]
        worst_costs = [np.mean(worst_cost_true), np.mean(worst_cost_abf), np.mean(worst_cost_obf), np.mean(worst_cost_obf_robust)]

        x = np.arange(len(labels))
        width = 0.35

        # Create bars
        rects1 = ax.bar(x, np.array(worst_costs) / 100, width, label='Worst-case Cost', color=RED)
        rects2 = ax.bar(x, np.array(original_costs) / 100, width, label='Original Cost', color=BLUE)

        # Customize plot
        ax.set_ylabel(r'PSO Cost ($\times10^2$£)')
        # ax.set_title(f'Uncertainty Budget = {budget_percentage[idx]}')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 12)
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
                        ha='center', va='bottom', fontsize=12)

        autolabel(rects1, offset=5)
        autolabel(rects2, offset=-1)

        plt.tight_layout()
        plt.savefig(data_dir + f'/uncertainty_performance_{idx}.pdf', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()

def draw_sensitivity():
    """
    Sensitivity analysis for ABF, OBF/Basic, OBF/SCO
    """
    data_dir = 'paper_exp/obf_result_1/'
    performance = np.load(data_dir + 'obf_sco/obf_sco_grad/performance.npy', allow_pickle=True).item()
    
    # Extract data
    cost_acc = performance['cost_acc']
    cost_obj = performance['cost_obj']
    cost_obj_sco = performance['cost_obj_sco']
    
    grad_acc = np.concatenate([performance['grad_W_acc'], performance['grad_b_acc']], axis=-1)
    grad_obj = np.concatenate([performance['grad_W_obj'], performance['grad_b_obj']], axis=-1)
    grad_obj_sco = np.concatenate([performance['grad_W_obj_sco'], performance['grad_b_obj_sco']], axis=-1)
    
    cos_sim_acc_obj = np.sum(grad_acc * grad_obj, axis=-1) / (np.linalg.norm(grad_acc, axis=-1) * np.linalg.norm(grad_obj, axis=-1))
    cos_sim_obj_obj_sco = np.sum(grad_obj * grad_obj_sco, axis=-1) / (np.linalg.norm(grad_obj, axis=-1) * np.linalg.norm(grad_obj_sco, axis=-1))
    cos_sim_acc_obj_sco = np.sum(grad_acc * grad_obj_sco, axis=-1) / (np.linalg.norm(grad_acc, axis=-1) * np.linalg.norm(grad_obj_sco, axis=-1))
    
    print("Overall Performance:")
    print('ABF Cost: ', np.mean(cost_acc), 'OBF/Basic Cost: ', np.mean(cost_obj), 'OBF/SCO Cost: ', np.mean(cost_obj_sco))
    print('ABF-OBF/Basic Cosine Similarity: ', np.mean(cos_sim_acc_obj))
    print('OBF/Basic-OBF/SCO Cosine Similarity: ', np.mean(cos_sim_obj_obj_sco))
    print('ABF-OBF/SCO Cosine Similarity: ', np.mean(cos_sim_acc_obj_sco))
    
    print("Seasonal Performance:")
    step_size = 30*24*3
    ABF_OBF_BASIC_COS_SIM = []
    OBF_BASIC_OBF_SCO_COS_SIM = []
    ABF_OBF_SCO_COS_SIM = []
    for i in range(4):
        start_idx = i * step_size
        end_idx = np.min([start_idx + step_size, len(cost_acc)])
        # print(f'Season {i+1}:')
        # # print('ABF Cost: ', np.mean(cost_acc[start_idx:end_idx]), 'OBF/Basic Cost: ', np.mean(cost_obj[start_idx:end_idx]), 'OBF/SCO Cost: ', np.mean(cost_obj_sco[start_idx:end_idx]))
        # print('ABF-OBF/Basic Cosine Similarity: ', np.mean(cos_sim_acc_obj[start_idx:end_idx]))
        # print('OBF/Basic-OBF/SCO Cosine Similarity: ', np.mean(cos_sim_obj_obj_sco[start_idx:end_idx]))
        # print('ABF-OBF/SCO Cosine Similarity: ', np.mean(cos_sim_acc_obj_sco[start_idx:end_idx]))
        ABF_OBF_BASIC_COS_SIM.append(np.mean(cos_sim_acc_obj[start_idx:end_idx]))
        OBF_BASIC_OBF_SCO_COS_SIM.append(np.mean(cos_sim_obj_obj_sco[start_idx:end_idx]))
        ABF_OBF_SCO_COS_SIM.append(np.mean(cos_sim_acc_obj_sco[start_idx:end_idx]))
    
    print("Average Cosine Similarity:")
    print('ABF-OBF/Basic: ', ABF_OBF_BASIC_COS_SIM)
    print('OBF/Basic-OBF/SCO: ', OBF_BASIC_OBF_SCO_COS_SIM)
    print('ABF-OBF/SCO: ', ABF_OBF_SCO_COS_SIM)
    
    # Plot bar plots
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(4)
    width = 0.25
    
    # Create bars
    rects1 = ax.bar(x - width, ABF_OBF_BASIC_COS_SIM, width, label=r'$\mathcal{P}_{train}^{abf}/\mathcal{P}_{train}^{obf}$', color=RED)
    rects2 = ax.bar(x, OBF_BASIC_OBF_SCO_COS_SIM, width, label=r'$\mathcal{P}_{train}^{obf}/\mathcal{P}_{train}^{obf/sco}$', color=BLUE)
    rects3 = ax.bar(x + width, ABF_OBF_SCO_COS_SIM, width, label=r'$\mathcal{P}_{train}^{abf}/\mathcal{P}_{train}^{obf/sco}$', color='#E6B3FF')
    
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
    
if __name__ == '__main__':
    evaluate_obf_sco()
    plot_uncertainty_performance()
    draw_sensitivity()