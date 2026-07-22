# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 11:55:51 2026

@author: sofia krasovskaya

Plot distribution of final head azimuths: Big vs Small target
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import os


# Define colours
colors = {'Control': '#c06c84', 'Big Target': '#2e00ff', 'Small Target': '#0091ff'}

# Create output directory
output_dir = './analysis_results'
os.makedirs(output_dir, exist_ok=True)

conditions = {
    'Control': '../sim_control_condition/control_experiment_results.json',
    'Big Target': '../sim_test_big/test_experiment_results.json', 
    'Small Target': '../sim_test_small/test_experiment_results.json'
}

# Extract final positions and errors for all conditions
final_data = {}

for name, path in conditions.items():
    with open(path, 'r') as f:
        data = json.load(f)
    
    final_positions = []
    target_positions = []
    final_errors = []
    
    for subject in data['subjects']:
        for trial in subject['trials']:
            if trial.get('status') != 'FAILED':
                final_head = trial['head_positions'][-1]
                target = trial['target_az']
                error = abs(final_head - target)
                
                final_positions.append(final_head)
                target_positions.append(target)
                final_errors.append(error)
    
    final_data[name] = {
        'final_positions': final_positions,
        'target_positions': target_positions,
        'errors': final_errors
    }
    
    print(f"\n{name}:")
    print(f"  Mean error: {np.mean(final_errors):.2f}°")
    print(f"  Median error: {np.median(final_errors):.2f}°")
    print(f"  Std: {np.std(final_errors):.2f}°")


###############################################################################
# PLOT 1: Final head position vs target (3 panels)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.patch.set_facecolor('white')

for idx, (name, ax) in enumerate(zip(conditions.keys(), axes)):
    data = final_data[name]
    color = colors[name]
    
    # Scatter plot
    ax.scatter(data['target_positions'], data['final_positions'], 
               alpha=0.5, s=30, color=color, edgecolors='black', linewidth=0.5)
    
    # Perfect localization line
    ax.plot([-90, 90], [-90, 90], 'r--', linewidth=3, label='Target')
    
    # ax.set_xlabel('Target azimuth (°)', fontsize=14)
    # ax.set_ylabel('Final head position (°)', fontsize=14)
   
    # Add shared axis labels
    fig.supxlabel('Target azimuth (°)', fontsize=27)
    fig.supylabel('Final head position (°)', fontsize=27)
    ax.set_title(f'{name}', fontsize=30, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-90, 90)
    ax.set_ylim(-90, 90)
    ax.set_aspect('equal')
    ax.legend(fontsize=20)
    ax.tick_params(axis='both', labelsize=22)
    ax.set_xticks(np.arange(-80, 81, 20))

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'final_head_vs_target.png'), dpi=300, bbox_inches='tight')
plt.show()
print("\nFinal head position plot saved")


###############################################################################
# PLOT 2: Distribution of final errors
fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
fig.patch.set_facecolor('white')

bins = np.arange(0, 91, 5)

for idx, (name, ax) in enumerate(zip(conditions.keys(), axes)):
    data = final_data[name]
    color = colors[name]
    errors = data['errors']
    
    # Histogram
    ax.hist(errors, bins=bins, alpha=0.7, color=color, edgecolor='black', linewidth=1)
    
    # Add mean line
    mean_error = np.mean(errors)
    ax.axvline(mean_error, color='black', linestyle='--', linewidth=2,
               label=f'Mean: {mean_error:.1f}°\n Std: {np.std(errors):.1f}')
    
    # ax.set_xlabel('Final error (°)', fontsize=14)
    # ax.set_ylabel('Frequency', fontsize=14)
    fig.supxlabel('Final error (°)', fontsize=27)
    fig.supylabel('Frequency', fontsize=26)
    ax.set_title(f'{name}', fontsize=28, fontweight = 'bold')
    ax.legend(fontsize=20)
    ax.grid(axis='y', alpha=0.3)
    ax.set_xlim(0, 90)
    ax.set_ylim(0, 470)
    ax.tick_params(axis='both', labelsize=22)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'final_error_distribution.png'), dpi=300, bbox_inches='tight')
plt.show()
print("Final error distribution plot saved")

###############################################################################
# FINAL ERROR ANALYSIS & STATS
print("\n" + "="*60)
print("FINAL ERROR ANALYSIS")
print("="*60)
for name in conditions.keys():
    errors = final_data[name]['errors']
    
    print(f"\n{name}:")
    print(f"  Mean: {np.mean(errors):.2f}°")
    print(f"  Median: {np.median(errors):.2f}°")
    print(f"  Std: {np.std(errors):.2f}°")
    print(f"  Range: {np.min(errors):.1f} - {np.max(errors):.1f}°")
    print(f"  N: {len(errors)}")

###############################################################################
# FINAL ERROR STATS - Pairwise comparisons
print("\n" + "="*60)
print("STATISTICAL COMPARISON: FINAL ERRORS")
print("="*60)

from scipy.stats import ttest_ind
from scipy.stats import sem
import pandas as pd

final_error_stats = {}
success_conditions = list(conditions.keys())

for i, cond1 in enumerate(success_conditions):
    for cond2 in success_conditions[i+1:]:
        
        err1 = final_data[cond1]['errors']
        err2 = final_data[cond2]['errors']
        
        # t-test
        t_stat, p_val = ttest_ind(err1, err2)
        
        # Effect size (Cohen's d)
        mean_diff = np.mean(err1) - np.mean(err2)
        pooled_std = np.sqrt((np.std(err1)**2 + np.std(err2)**2) / 2)
        cohens_d = mean_diff / pooled_std
        
        # Summary stats
        m1 = np.mean(err1)
        sem1 = sem(err1)
        med1 = np.median(err1)
        std1 = np.std(err1)
        min1 = np.min(err1)
        max1 = np.max(err1)
        
        m2 = np.mean(err2)
        sem2 = sem(err2)
        med2 = np.median(err2)
        std2 = np.std(err2)
        min2 = np.min(err2)
        max2 = np.max(err2)
        
        print(f"\n{cond1} vs {cond2}:")
        print(f"  {cond1}: {m1:.2f} ± {sem1:.2f}° (median={med1:.2f}°, n={len(err1)})")
        print(f"  {cond2}: {m2:.2f} ± {sem2:.2f}° (median={med2:.2f}°, n={len(err2)})")
        print(f"  Difference: {abs(mean_diff):.2f}°")
        print(f"  t = {t_stat:.3f}, p = {p_val:.6f}")
        print(f"  Cohen's d = {cohens_d:.3f}")
        print(f"  Significant: {'***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'}")
        
        # Store results
        final_error_stats[f"{cond1}_vs_{cond2}"] = {
            f"{cond1}_mean": m1,
            f"{cond1}_sem": sem1,
            f"{cond1}_median": med1,
            f"{cond1}_std": std1,
            f"{cond1}_min": min1,
            f"{cond1}_max": max1,
            f"{cond1}_n": len(err1),
            f"{cond2}_mean": m2,
            f"{cond2}_sem": sem2,
            f"{cond2}_median": med2,
            f"{cond2}_std": std2,
            f"{cond2}_min": min2,
            f"{cond2}_max": max2,
            f"{cond2}_n": len(err2),
            't_stat': t_stat,
            'p_value': p_val,
            'cohens_d': cohens_d,
            'significant': p_val < 0.05
        }

# Save to CSV
df_final_error_stats = pd.DataFrame(final_error_stats).T
df_final_error_stats.to_csv(os.path.join(output_dir, 'final_error_stats.csv'))
print("\n" + "="*60)
print("Final error stats saved to final_error_stats.csv")
print("="*60)
