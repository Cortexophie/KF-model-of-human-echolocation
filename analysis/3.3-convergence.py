# -*- coding: utf-8 -*-
"""
@author: sofia krasovskaya
Created on Tue Feb  3 12:01:35 2026

Show how error decreases over time
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import ttest_ind
import pandas as pd

#create output folder
output_dir = './analysis_results'
os.makedirs(output_dir, exist_ok=True)
    
###############################################################################
#try plot a single trial first
# Load Big Target data
with open('../sim_test_big/test_experiment_results.json', 'r') as f:
    data = json.load(f)

# Get one trial
trial = data['subjects'][0]['trials'][0]

head_positions = np.array(trial['head_positions'])
target_az = trial['target_az']

# Calculate absolute error at each step
absolute_error = np.abs(head_positions - target_az)

# Time array
time = np.arange(-len(absolute_error),0) * 0.1

# Plot it
plt.figure(figsize=(12, 6))
plt.plot(time, absolute_error, 'b-', linewidth=2)
plt.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Perfect (0° error)')
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Absolute Error (°)', fontsize=14)
plt.title('Single Trial: Error Convergence', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.ylim(0, 180)
plt.tight_layout()
plt.show()

print(f"Initial error: {absolute_error[0]:.1f}°")
print(f"Final error: {absolute_error[-1]:.1f}°")
print(f"Error reduction: {absolute_error[0] - absolute_error[-1]:.1f}°")

###############################################################################
# Show average error convergence for all three conditions
conditions = {
    'Control': '../sim_control_condition/control_experiment_results.json',
    'Big Target': '../sim_test_big/test_experiment_results.json', 
    'Small Target': '../sim_test_small/test_experiment_results.json'
}
colors = {'Control': '#c06c84', 'Big Target': '#2e00ff', 'Small Target': '#0091ff'}

plt.figure(figsize=(12, 7))

for name, path in conditions.items():
    with open(path, 'r') as f:
        data = json.load(f)
    
    # Collect all trials
    all_errors = []
    
    for subject in data['subjects']:
        for trial in subject['trials']:
            if trial.get('status') == 'FAILED':
                continue
                
            head_positions = np.array(trial['head_positions'])
            target_az = trial['target_az']
            
            # Calculate absolute error at each step
            absolute_error = np.abs(head_positions - target_az)
            all_errors.append(absolute_error)
    
    # Find max length
    max_length = max(len(err) for err in all_errors)
    
    # Pad shorter trials by CARRYING FORWARD the final error value
    padded_errors = []
    for err in all_errors:
        padded = np.full(max_length, np.nan)
        padded[:len(err)] = err
        # Carry forward final error for all timepoints after trial end
        if len(err) < max_length:
            padded[len(err):] = err[-1]
        padded_errors.append(padded)
    
    # Convert to array and calculate mean
    padded_errors = np.array(padded_errors)
    mean_error = np.nanmean(padded_errors, axis=0)
    std_error = np.nanstd(padded_errors, axis=0)
    
    # Time array
    time = np.arange(max_length) * 0.1
    
    # Plot mean with shaded error band
    plt.plot(time, mean_error, color=colors[name], linewidth=5, label=name)
    plt.fill_between(time, mean_error - std_error, mean_error + std_error, 
                     color=colors[name], alpha=0.2)
    
    print(f"{name}:")
    print(f"  Initial error: {mean_error[0]:.1f}°")
    print(f"  Final error: {mean_error[-1]:.1f}°")
    print(f"  Max trial length: {max_length} steps = {max_length * 0.1:.1f} sec")

plt.axhline(y=5, color='red', linestyle='--', linewidth=2, label='target localization threshold')
plt.xlabel('Time (s)', fontsize=27)
plt.ylabel('MAE (°)', fontsize=27)
# plt.title('Convergence (Error Reduction Over Time)', fontsize=18)
plt.tick_params(axis='both', labelsize=22)
plt.legend(fontsize=20)
plt.grid(True, alpha=0.3)
plt.ylim(0, None)
plt.tight_layout()
plt.savefig(os.path.join(output_dir,'convergence_analysis_start_lock.png'), dpi=300, bbox_inches='tight')
plt.show()

###############################################################################
#####  #####    ##   #####  #####
#        #     #  #    #    #
#####    #     ####    #    #####
    #    #     #  #    #        #
#####    #     #  #    #    #####
###############################################################################
"""
Convergence Statistics Analyses

Calculate four key convergence metrics:
1. Final error - error at trial end
2. Error reduction - how much error decreased (initial - final)
3. Time to convergence - when error first drops below 5° threshold
4. Convergence rate - how fast error drops (deg/s) 

"""
# Convergence threshold (from model criterion)
converge_threshold = 5.0  # degrees

print("\n" + "="*60)
print("CONVERGENCE STATISTICS")
print("="*60)
print(f"Using threshold: {converge_threshold }°\n")

#dictionary for stats
converge_summary_stats = {}

for name, path in conditions.items():
    with open(path, 'r') as f:
        data = json.load(f)
    
    #storage arrays for this condition
    start_errors =[]
    final_errors = []
    error_reductions = []
    times_to_converge = []
    converge_rate = []
    
    #now go through each trial
    for sub in data['subjects']:
        for trial in sub['trials']:
            head_positions = np.array(trial['head_positions'])
            target_az = trial['target_az']
            
            #calculate absolute error at each timestep
            errors = np.abs(head_positions - target_az)
            
            # 0. get starting error
            start_errors.append(errors[0])
            
            # 1. get final error
            final_errors.append(errors[-1])
            
            #2. get error reduction
            error_reductions.append(errors[0] - errors[-1])
            
            # 3. time to convergence
            #find index of first error that fell below threshold:
            converged_idx = np.where(errors < converge_threshold)[0]
            #now if there was a convergence, pull the value and translate into seconds:
            if len(converged_idx) > 0:
                times_to_converge.append(converged_idx[0] * 0.1) #1 step = 0.1s
            
            # 4. get rate of convergence 
            trial_dur = len(errors) * 0.1
            converge_rate.append((errors[0] - errors[-1]) / trial_dur)
    
    #save results
    converge_summary_stats[name] = {
        'start_errors': np.array(start_errors),
        'final_errors': np.array(final_errors),
        'error_reductions': np.array(error_reductions),
        'times_to_converge': np.array(times_to_converge) if times_to_converge else np.array([]),
        'converge_rate': np.array(converge_rate)
    }

# Create summary DataFrame
summary_rows = []
for condition in ['Control', 'Big Target', 'Small Target']:
    for metric in ['start_errors','final_errors', 'error_reductions', 'times_to_converge', 'converge_rate']:
        data = converge_summary_stats[condition][metric]
        
        if len(data) == 0:
            continue
            
        summary_rows.append({
            'Metric': metric,
            'Condition': condition,
            'Mean': data.mean(),
            'Std': data.std(),
            'N': len(data)
        })

df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(os.path.join(output_dir, 'convergence_summary_stats.csv'), index=False)
print("Saved: convergence_summary_stats.csv")
print(df_summary)

###############################################################################
# Now stats to compare conditions
###############################################################################

# Create comparison statistics DataFrame
comparisons = []

###############################################################################
# 1 FINAL ERROR STATS
print("-" * 60)
print("1: FINAL ERROR (error at trial end)")
print("-" * 60)

metric = 'final_error'

control_final = converge_summary_stats['Control']['final_errors']
big_final = converge_summary_stats['Big Target']['final_errors']
small_final = converge_summary_stats['Small Target']['final_errors']

#big vs small
t_test_big_vs_small, p_val_big_vs_small = ttest_ind(big_final, small_final)
cohens_d_big_vs_small = (big_final.mean() - small_final.mean()) / np.sqrt((big_final.std()**2 + small_final.std()**2) / 2)

# save big vs small comaprison stats to df
comparisons.append({
    'Metric': metric,
    'Comparison': 'big_vs_small',
    'Mean_Diff': big_final.mean() - small_final.mean(),
    't_stat': t_test_big_vs_small,
    'p_value': p_val_big_vs_small,
    'cohens_d': cohens_d_big_vs_small
})

# control vs big
t_test_ctrl_vs_big, p_val_ctrl_vs_big = ttest_ind(control_final, big_final)
cohens_d_ctrl_vs_big = (control_final.mean() - big_final.mean()) / np.sqrt((control_final.std()**2 + big_final.std()**2) / 2)

# save ctrl vs big comaprison stats to df
comparisons.append({
    'Metric': metric,
    'Comparison': 'ctrl_vs_big',
    'Mean_Diff': control_final.mean() - big_final.mean(),
    't_stat': t_test_ctrl_vs_big,
    'p_value': p_val_ctrl_vs_big,
    'cohens_d': cohens_d_ctrl_vs_big
})

# ctrl vs small
t_test_ctrl_vs_small, p_val_ctrl_vs_small = ttest_ind(control_final, small_final)
cohens_d_ctrl_vs_small = (control_final.mean() - small_final.mean()) / np.sqrt((control_final.std()**2 + small_final.std()**2) / 2)

# save ctrl vs big comaprison stats to df
comparisons.append({
    'Metric': metric,
    'Comparison': 'ctrl_vs_small',
    'Mean_Diff': control_final.mean() - small_final.mean(),
    't_stat': t_test_ctrl_vs_small,
    'p_value': p_val_ctrl_vs_small,
    'cohens_d': cohens_d_ctrl_vs_small
})


# Levene test for SD
from scipy.stats import levene

# Test for equal variances (Levene's test)
levene_stat, levene_p = levene(big_final, small_final)

# Variance ratio (useful effect size for precision)
variance_ratio = np.var(small_final) / np.var(big_final)

print(f"\nPrecision comparison (variance test):")
print(f"  Big SD: {np.std(big_final):.2f}°")
print(f"  Small SD: {np.std(small_final):.2f}°")
print(f"  Levene's W = {levene_stat:.3f}, p = {levene_p:.6f}")
print(f"  Variance ratio (Small/Big): {variance_ratio:.2f}")
print(f"  Significant: {'***' if levene_p < 0.001 else '**' if levene_p < 0.01 else '*' if levene_p < 0.05 else 'ns'}")


###############################################################################
# 2 ERROR REDUCTION STATS
print("-" * 60)
print("2: ERROR REDUCTIN (initial - final)")
print("-" * 60)

metric = 'error_reduction'

control_reduct = converge_summary_stats['Control']['error_reductions']
big_reduct = converge_summary_stats['Big Target']['error_reductions']
small_reduct = converge_summary_stats['Small Target']['error_reductions']

#big vs small
t_test_big_vs_small, p_val_big_vs_small = ttest_ind(big_reduct, small_reduct)
cohens_d_big_vs_small = (big_reduct.mean() - small_reduct.mean()) / np.sqrt((big_reduct.std()**2 + small_reduct.std()**2) / 2)

comparisons.append({
    'Metric': metric,
    'Comparison': 'big_vs_small',
    'Mean_Diff': big_reduct.mean() - small_reduct.mean(),
    't_stat': t_test_big_vs_small,
    'p_value': p_val_big_vs_small,
    'cohens_d': cohens_d_big_vs_small
})

# control vs big
t_test_ctrl_vs_big, p_val_ctrl_vs_big = ttest_ind(control_reduct, big_reduct)
cohens_d_ctrl_vs_big = (control_reduct.mean() - big_reduct.mean()) / np.sqrt((control_reduct.std()**2 + big_reduct.std()**2) / 2)

comparisons.append({
    'Metric': metric,
    'Comparison': 'ctrl_vs_big',
    'Mean_Diff': control_reduct.mean() - big_reduct.mean(),
    't_stat': t_test_ctrl_vs_big,
    'p_value': p_val_ctrl_vs_big,
    'cohens_d': cohens_d_ctrl_vs_big
})

# ctrl vs small
t_test_ctrl_vs_small, p_val_ctrl_vs_small = ttest_ind(control_reduct, small_reduct)
cohens_d_ctrl_vs_small = (control_reduct.mean() - small_reduct.mean()) / np.sqrt((control_reduct.std()**2 + small_reduct.std()**2) / 2)

comparisons.append({
    'Metric': metric,
    'Comparison': 'ctrl_vs_small',
    'Mean_Diff': control_reduct.mean() - small_reduct.mean(),
    't_stat': t_test_ctrl_vs_small,
    'p_value': p_val_ctrl_vs_small,
    'cohens_d': cohens_d_ctrl_vs_small
})

###############################################################################
# 3 TIME TO CONVERGE
print("-" * 60)
print(f"3. TIME TO CONVERGE ((first time error < {converge_threshold}°))")
print("-" * 60)

metric = 'time_to_converge'

#omit control since no convergence data there
big_conv_time = converge_summary_stats['Big Target']['times_to_converge']
small_conv_time = converge_summary_stats['Small Target']['times_to_converge']

#big vs small
t_test_big_vs_small, p_val_big_vs_small = ttest_ind(big_conv_time, small_conv_time)
cohens_d_big_vs_small = (big_conv_time.mean() - small_conv_time.mean()) / np.sqrt((big_conv_time.std()**2 + small_conv_time.std()**2) / 2)

comparisons.append({
    'Metric': metric,
    'Comparison': 'big_vs_small',
    'Mean_Diff': big_conv_time.mean() - small_conv_time.mean(),
    't_stat': t_test_big_vs_small,
    'p_value': p_val_big_vs_small,
    'cohens_d': cohens_d_big_vs_small
})

###############################################################################
# 4 CONVERGENCE RATES
print("-" * 60)
print("4. CONVERGENCE RATE (deg/s)")
print("-" * 60)

metric = 'converge_rate'

#omit control since no convergence data there
big_conv_rate = converge_summary_stats['Big Target']['converge_rate']
small_conv_rate = converge_summary_stats['Small Target']['converge_rate']

#big vs small
t_test_big_vs_small, p_val_big_vs_small = ttest_ind(big_conv_rate, small_conv_rate)
cohens_d_big_vs_small = (big_conv_rate.mean() - small_conv_rate.mean()) / np.sqrt((big_conv_rate.std()**2 + small_conv_rate.std()**2) / 2)

comparisons.append({
    'Metric': metric,
    'Comparison': 'big_vs_small',
    'Mean_Diff': big_conv_rate.mean() - small_conv_rate.mean(),
    't_stat': t_test_big_vs_small,
    'p_value': p_val_big_vs_small,
    'cohens_d': cohens_d_big_vs_small
})

#save comparisons stats df as csv
df_comparisons = pd.DataFrame(comparisons)
df_comparisons.to_csv(os.path.join(output_dir, 'convergence_comparisons_stats.csv'), index=False)
print(f"Saved: {output_dir}/convergence_comparisons_stats.csv")