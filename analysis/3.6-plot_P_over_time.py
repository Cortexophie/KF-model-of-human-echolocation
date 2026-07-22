# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 22:45:47 2026

@author: sophie_chan
"""

"""
Plot Kalman Filter variance (P) over time for Big vs Small target conditions
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem

# Load results from both conditions
with open('sim_test_big/test_experiment_results.json', 'r') as f:
    big_results = json.load(f)

with open('sim_test_small/test_experiment_results.json', 'r') as f:
    small_results = json.load(f)

def extract_variances_aligned(results):
    """Extract P values aligned to trial start"""
    # Find max trial length
    max_length = 0
    for subject in results['subjects']:
        for trial in subject['trials']:
            if trial.get('status') != 'FAILED':
                length = len(trial['kf_variances'])
                max_length = max(max_length, length)
    
    # Collect P values at each timestep
    all_variances = []
    for subject in results['subjects']:
        for trial in subject['trials']:
            if trial.get('status') != 'FAILED':
                variances = trial['kf_variances']
                all_variances.append(variances)
    
    # Calculate mean and SEM at each timestep
    mean_P = []
    sem_P = []
    
    for step_idx in range(max_length):
        values_at_step = [trial[step_idx] for trial in all_variances 
                         if step_idx < len(trial)]
        if values_at_step:
            mean_P.append(np.mean(values_at_step))
            sem_P.append(sem(values_at_step))
        else:
            mean_P.append(np.nan)
            sem_P.append(np.nan)
    
    return np.array(mean_P), np.array(sem_P)

# Extract data
big_mean_P, big_sem_P = extract_variances_aligned(big_results)
small_mean_P, small_sem_P = extract_variances_aligned(small_results)

# Create time axis in seconds
max_steps = max(len(big_mean_P), len(small_mean_P))
time_axis = np.arange(max_steps) * 0.1  # 0.1s per step

# Plot
fig, ax = plt.subplots(figsize=(12, 6))

# Big target
valid_big = ~np.isnan(big_mean_P)
ax.plot(time_axis[:len(big_mean_P)][valid_big], big_mean_P[valid_big], 
        'b-', linewidth=2, label='Big Target')
ax.fill_between(time_axis[:len(big_mean_P)][valid_big],
                big_mean_P[valid_big] - big_sem_P[valid_big],
                big_mean_P[valid_big] + big_sem_P[valid_big],
                color='b', alpha=0.3)

# Small target
valid_small = ~np.isnan(small_mean_P)
ax.plot(time_axis[:len(small_mean_P)][valid_small], small_mean_P[valid_small], 
        'r-', linewidth=2, label='Small Target')
ax.fill_between(time_axis[:len(small_mean_P)][valid_small],
                small_mean_P[valid_small] - small_sem_P[valid_small],
                small_mean_P[valid_small] + small_sem_P[valid_small],
                color='r', alpha=0.3)


ax.set_xlabel('Time (s)', fontsize=12)
ax.set_ylabel('KF Uncertainty (P) [deg²]', fontsize=12)
ax.set_title('KF uncertainty (P) over time: Big vs Small target', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig('kf_variance_P_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

