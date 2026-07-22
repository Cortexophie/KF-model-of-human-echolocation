# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 10:51:51 2026

@author: sofia krasovskaya

Look at rates of target localization & RT's
"""

import json
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, ttest_ind, sem

#create output folder
output_dir = './analysis_results'
os.makedirs(output_dir, exist_ok=True)

# Load all three conditions
conditions = {
    'Control': '../sim_control_condition/control_experiment_results.json',
    'Big Target': '../sim_test_big/test_experiment_results.json',
    'Small Target': '../sim_test_small/test_experiment_results.json'
}

colors = {'Control': '#c06c84', 'Big Target': '#2e00ff', 'Small Target': '#0091ff'}

###############################################################################
# CHECK IF WE HAVE DATA & LOOK AT STRUCTURE
# Load one condition

# with open('../sim_control_condition/control_experiment_results.json', 'r') as f:
#     data = json.load(f)

# # check what data contain
# print("Keys:", data.keys())
# print("\nNumber of subjects:", data['experiment_params']['num_subs'])
# print("Trials per subject:", data['experiment_params']['num_trials'])

# # Look at one trial
# first_trial = data['subjects'][0]['trials'][0]
# print("\nFirst trial keys:", first_trial.keys())
# print("Target location:", first_trial['target_az'])
# print("Number of steps:", first_trial['num_steps_used'])
# print("Target found?:", first_trial.get('target_found', 'NOT IN DATA'))



###############################################################################
# CALCULATE BOTH SUBJECTIVE AND OBJECTIVE SUCCESS
print("="*60)
print("SUCCESS ANALYSIS: SUBJECTIVE vs OBJECTIVE")
print("="*60)

OBJECTIVE_THRESHOLD = 5.0  # degrees - target within 5° counts as found

success_data = {}  # Store all success metrics

for name, path in conditions.items():
    with open(path, 'r') as f:
        data = json.load(f)
    
    total_trials = 0
    subjective_success = 0
    objective_success = 0
    both_success = 0  # Correctly detected AND actually near target
    false_positive = 0  # Thinks found but actually far
    missed_detection = 0  # Actually near but didn't detect
    
    for subject in data['subjects']:
        for trial in subject['trials']:
            if trial.get('status') == 'FAILED':
                continue
                
            total_trials += 1
            
            # Subjective: Model thinks it found target
            subjective = trial.get('target_found', False)
            
            # Objective: Actually near target
            final_head = trial['head_positions'][-1]
            target = trial['target_az']
            final_error = abs(final_head - target)
            objective = final_error <= OBJECTIVE_THRESHOLD
            
            # Count each category
            if subjective:
                subjective_success += 1
            if objective:
                objective_success += 1
            if subjective and objective:
                both_success += 1
            elif subjective and not objective:
                false_positive += 1
            elif not subjective and objective:
                missed_detection += 1
    
    # Calculate rates
    subj_rate = (subjective_success / total_trials * 100) if total_trials > 0 else 0
    obj_rate = (objective_success / total_trials * 100) if total_trials > 0 else 0
    both_rate = (both_success / total_trials * 100) if total_trials > 0 else 0
    fp_rate = (false_positive / total_trials * 100) if total_trials > 0 else 0
    miss_rate = (missed_detection / total_trials * 100) if total_trials > 0 else 0
    
    # Store results
    success_data[name] = {
        'total': total_trials,
        'subjective_success': subjective_success,
        'objective_success': objective_success,
        'both_success': both_success,
        'false_positive': false_positive,
        'missed_detection': missed_detection,
        'subjective_rate': subj_rate,
        'objective_rate': obj_rate,
        'both_rate': both_rate,
        'fp_rate': fp_rate,
        'miss_rate': miss_rate
    }
    
    print(f"\n{name} (n={total_trials}):")
    print(f"  Subjective success (model thinks found): {subjective_success} ({subj_rate:.1f}%)")
    print(f"  Objective success (actually <{OBJECTIVE_THRESHOLD}°): {objective_success} ({obj_rate:.1f}%)")
    print(f"  Both (correct detection): {both_success} ({both_rate:.1f}%)")
    print(f"  False positives (thinks found but >{OBJECTIVE_THRESHOLD}): {false_positive} ({fp_rate:.1f}%)")
    print(f"  Missed detections (near target but didn't detect): {missed_detection}" )
    
    
# ###############################################################################
# # CHECK DETECTION RATES
# #Look at success rates
# print("="*60)
# print("COMPARISON - OBJECTIVE SUCCESS (Chi-square test)")
# print("="*60)

# success_conditions = list(success_data.keys())
# success_stats = {}

# for name, path in conditions.items():
#     with open(path, 'r') as f:
#         data = json.load(f)
    
#     total_trials = 0
#     successful_trials = 0
    
#     for subject in data['subjects']:
#         for trial in subject['trials']:
#             total_trials += 1
#             if trial['target_found']:
#                 successful_trials += 1
                
#     success_rate = (successful_trials / total_trials * 100) if total_trials > 0 else 0
    
#     # Store in dictionary
#     success_data[name] = {
#         'total': total_trials,
#         'successes': successful_trials,
#         'rate': success_rate
#     }
    
#     print(f"{name}:")
#     print(f"  Successes: {successful_trials}/{total_trials}")
#     print(f"  Success rate: {success_rate:.1f}%")
    
    
###############################################################################
# STATS FOR SUCCESS 
# Statistical comparison of success rates using chi-square test

print("\n" + "="*70)
print("SUCCESS RATE COMPARISON (Chi-square test)")
print("="*70)

success_conditions = list(success_data.keys())
success_stats = {}


# Pairwise comparisons for objective localizations within 5° of target
for i, cond1 in enumerate(success_conditions):
    for cond2 in success_conditions[i+1:]:
        m1 = success_data[cond1]
        m2 = success_data[cond2]
        
        # Create contingency table
        contingency = [
            [m1['objective_success'], m1['total'] - m1['objective_success']],
            [m2['objective_success'], m2['total'] - m2['objective_success']]
        ]
        
        chi2, p_val, dof, expected = chi2_contingency(contingency)
        
        print(f"\n{cond1} vs {cond2}:")
        print(f"  {cond1}: {m1['objective_success']}/{m1['total']} ({m1['objective_rate']:.1f}%)")
        print(f"  {cond2}: {m2['objective_success']}/{m2['total']} ({m2['objective_rate']:.1f}%)")
        print(f"  χ² = {chi2:.3f}, p = {p_val:.4f}")
        print(f"  Significant: {'***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'}")
        
        success_stats[f"{cond1}_vs_{cond2}"] = {
            'chi2': chi2,
            'p_value': p_val,
            'significant': p_val < 0.05,
            'dof': dof
        }

# save stats to csv
df_success_stats = pd.DataFrame(success_data).T
df_success_stats.to_csv(os.path.join(output_dir, 'success_rates_detailed.csv'))

df_stats = pd.DataFrame(success_stats).T
df_stats.to_csv(os.path.join(output_dir, 'success_rates_stats.csv'))
    

###############################################################################
# CHECK NUM TRIALS FOR EACH SUB    
#How many trials per condition for each sub?    
print("\n" + "="*60)
print("TRIALS PER SUBJECT")
print("="*60)

for name, path in conditions.items():
    with open(path, 'r') as f:
        data = json.load(f)
    
    print(f"\n{name}:")
    for subject in data['subjects']:
        n_trials = len(subject['trials'])
        print(f"  Subject {subject['id']}: {n_trials} trials")
        

        
###############################################################################
# LOOK AT RESPONSE TIMES
# calculate and save RT's for successful trials
print("\n" + "="*60)
print("RESPONSE TIME ANALYSIS")
print("="*60)


# Dictionary to store all response times
all_response_times = {}

for name, path in conditions.items():
    with open(path, 'r') as f:
        data = json.load(f)
    
    response_times = []
    
    for subject in data['subjects']:
        for trial in subject['trials']:
            if trial['target_found']:
                time_seconds = trial['num_steps_used'] * 0.1
                response_times.append(time_seconds)
    
    
    # Store in dictionary
    all_response_times[name] = response_times
    
    
    # Print summary
    if response_times:
        mean_time = np.mean(response_times)
        median_time = np.median(response_times)  
        std_time = np.std(response_times)
        min_time = np.min(response_times) 
        max_time = np.max(response_times)
        print(f"\n{name}:")
        print(f"  Mean: {mean_time:.2f} ± {std_time:.2f} seconds")
        print(f"  Median: {median_time:.2f} seconds") 
        print(f"  Range: {min_time:.1f} - {max_time:.1f} seconds") 
        print(f"  N trials: {len(response_times)}")
    else:
        print(f"\n{name}: No successful trials")
        
       
###############################################################################
# RESPONSE TIMES STATS
# Statistical comparison of response times using t-tests
print("\n" + "="*60)
print("STATISTICAL COMPARISON: RESPONSE TIMES")
print("="*60)

rt_stats = {}


for i, cond1 in enumerate(success_conditions):
        for cond2 in success_conditions[i+1:]:
            
            rt1 =  all_response_times[cond1]
            rt2 = all_response_times[cond2]
            
            # Skip if either condition has no RTs
            if len(rt1) == 0 or len(rt2) == 0:
                continue
            
            # t-test
            t_stat, p_val = ttest_ind(rt1, rt2)
            
            # Effect size (Cohen's d)
            mean_diff = np.mean(rt1) - np.mean(rt2)
            pooled_std = np.sqrt((np.std(rt1)**2 + np.std(rt2)**2) / 2)
            cohens_d = mean_diff / pooled_std
            
            # means and SEM's
            m1 = np.mean(rt1)
            sem1 = sem(rt1)
            med1 = np.median(rt1)
            min1 = np.min(rt1)
            max1 = np.max(rt1)
            m2 = np.mean(rt2)
            sem2 = sem(rt2)
            med2 = np.median(rt2)
            min2 = np.min(rt2)
            max2 = np.max(rt2)
            
            
            print(f"\n{cond1} vs {cond2}:")
            print(f"  {cond1}: {m1:.2f} ± {sem1:.2f} s (n={len(rt1)})")
            print(f"  {cond2}: {m2:.2f} ± {sem2:.2f} s (n={len(rt2)})")
            print(f" Difference: {abs(mean_diff):.2f} s")
            print(f"  t = {t_stat:.3f}, p = {p_val:.4f}")
            print(f"  Cohen's d = {cohens_d:.3f}")
            print(f"  Significant: {'***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'}")
            
            # store results
            rt_stats[f"{cond1}_vs_{cond2}"] = {
                f"{cond1}_mean": m1,
                f"{cond1}_sem": sem1,
                f"{cond1}_median": med1,
                f"{cond1}_min": min1,
                f"{cond1}_max": max1,
                f"{cond1}_n": len(rt1),
                f"{cond2}_mean": m2,
                f"{cond2}_sem": sem2,
                f"{cond2}_median": med2,
                f"{cond2}_min": min2,
                f"{cond2}_max": max2,
                f"{cond2}_n": len(rt2),
                't_stat': t_stat,
                'p_value': p_val,
                'cohens_d': cohens_d,
                'significant': p_val < 0.05
            }

# save rt stats to csv
df_rt_stats = pd.DataFrame(rt_stats).T #transpose so each comparison is a row
df_rt_stats.to_csv(os.path.join(output_dir,'rt_stats.csv'))

###############################################################################
# VISUALISE
# 
print("\n" + "="*60)
print("CREATING PLOTS")
print("="*60)

plot_colors = [colors[c] for c in success_conditions]

############################
# Plot 1: Success rates
fig1, ax1 = plt.subplots(figsize=(8, 6))
fig1.patch.set_facecolor('white')

rates = [success_data[c]['both_rate'] for c in success_conditions]
ax1.bar(success_conditions, rates, color=plot_colors, alpha=0.7)
ax1.set_ylabel('Localization rate (%)', fontsize=20)
ax1.set_title('Target localization rates to within 5° of target', fontsize=20)
ax1.set_ylim(0, 105)
ax1.grid(axis='y', alpha=0.3)
ax1.tick_params(axis='both', labelsize=20)
for i, v in enumerate(rates):
    ax1.text(i, v + 2, f'{v:.1f}%', ha='center', fontsize=20)

# plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'objective_success_rates.png'), dpi=300, bbox_inches='tight')
plt.show()
print("Success rate plot saved as 'success_rates.png'")

############################
 # Plot 2: RT boxplot or violinplot
rt_conditions = ['Big Target', 'Small Target']
fig2, ax2 = plt.subplots(figsize=(8, 6))
fig2.patch.set_facecolor('white')

response_data = [all_response_times[c] for c in rt_conditions]
# bp = ax2.boxplot(response_data, labels=rt_conditions, patch_artist=True)

# # Colour the boxes
# for patch, color in zip(bp['boxes'], plot_colors[1:]):
#     patch.set_facecolor(color)
#     patch.set_alpha(0.7)

vp = ax2.violinplot(response_data, positions=[1, 2], showmeans=False, showmedians=True)
# Colour the violins
for i, (pc, color) in enumerate(zip(vp['bodies'], [colors['Big Target'], colors['Small Target']])):
    pc.set_facecolor(color)
    pc.set_alpha(0.7)
    pc.set_edgecolor('black')
    pc.set_linewidth(1)

# Style the other violin components (median, mean, etc.)
for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians', 'cmeans'):
    if partname in vp:
        vp[partname].set_edgecolor('black')
        vp[partname].set_linewidth(1.5)

# Set x-axis labels (violins)
ax2.set_xticks([1, 2])
ax2.set_xticklabels(rt_conditions)
ax2.set_yticks(np.arange(4, 18, 2))

#works for both vp or bp
ax2.set_ylabel('Response time (s)', fontsize=20)
ax2.set_title('Time to response', fontsize=20)
ax2.grid(axis='y', alpha=0.3)
ax2.tick_params(axis='both', labelsize=20)

# plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'response_times.png'), dpi=300, bbox_inches='tight')
plt.show()
print("Response time plot saved as 'response_times.png'")
