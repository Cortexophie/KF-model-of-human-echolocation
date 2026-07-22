# -*- coding: utf-8 -*-
"""
@author: sofia krasovskaya
Created on Tue Feb  3 11:39:59 2026

trajectory_analysis

Shows head orientation relative to target over time
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import os

#create output folder
output_dir = './analysis_results'
os.makedirs(output_dir, exist_ok=True)

###############################################################################
# Start with one trial to understand the data
# Load Big Target data
with open('../sim_test_big/test_experiment_results.json', 'r') as f:
    data = json.load(f)

# Get single trial (sub 1, trial 1)
trial = data['subjects'][0]['trials'][0]

# Extract the key info
head_positions = np.array(trial['head_positions'])
target_az = trial['target_az']
num_steps = trial['num_steps_used']

# Calculate relative orientation (head - target)
# When it is 0, head is pointing at target
relative_orientation = head_positions - target_az

# Create time array (0.1 seconds per step)
time = np.arange((-num_steps),0) * 0.1

# Print info
print(f"Trial info:")
print(f"  Target at: {target_az:.1f}°")
print(f"  Trial duration: {num_steps} steps = {num_steps * 0.1:.1f} seconds")
print(f"  Target found: {trial['target_found']}")
print(f"  Final head position: {head_positions[-1]:.1f}°")
print(f"  Final error: {abs(relative_orientation[-1]):.1f}°")

# Plot it
plt.figure(figsize=(12, 6))
plt.plot(time, relative_orientation, 'b-', linewidth=2, label='Head orientation')
plt.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Target (perfect alignment)')
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Head Orientation Relative to Target (°)', fontsize=14)
plt.title('Single Trial: Head Movement Over Time', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()

###############################################################################
# Now try 50 trials
import random

# Load Big Target data
with open('../sim_test_big/test_experiment_results.json', 'r') as f:
    big_data = json.load(f)

# Collect all trials
all_trials = []
for subject in big_data['subjects']:
    for trial in subject['trials']:
        all_trials.append(trial)

# Pick 50 random trials
random.seed(42)  # for reproducibility
selected_trials = random.sample(all_trials, min(50, len(all_trials)))

print(f"Plotting {len(selected_trials)} trials...")

# Plot them all
plt.figure(figsize=(14, 8))

for trial in selected_trials:
    head_positions = np.array(trial['head_positions'])
    target_az = trial['target_az']
    relative_orientation = head_positions - target_az
    time = np.arange(-len(head_positions),0) * 0.1
    
    # Plot with transparency so we can see overlaps
    plt.plot(time, relative_orientation, 'b-', alpha=0.3, linewidth=1)

# Add target line
plt.axhline(y=0, color='r', linestyle='--', linewidth=3, label='Target')

plt.xlabel('Time (s)', fontsize=16)
plt.ylabel('Head Orientation Relative to Target (°)', fontsize=16)
plt.title('Big Target: 50 Random Trials Overlaid', fontsize=18)
plt.ylim(-180, 180)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=14)
plt.tight_layout()
plt.show()

###############################################################################
# Now compare all three conditions side by side, 50 random samples each

fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.patch.set_facecolor('white')

conditions = {
    'Control': '../sim_control_condition/control_experiment_results.json',
    'Big Target': '../sim_test_big/test_experiment_results.json', 
    'Small Target': '../sim_test_small/test_experiment_results.json'
}

colors = {'Control': '#c06c84', 'Big Target': '#2e00ff', 'Small Target': '#0091ff'}

for idx, (name, path) in enumerate(conditions.items()):
    ax = axes[idx]
    
    with open(path, 'r') as f:
        data = json.load(f)
    
    # Collect all trials
    all_trials = []
    for subject in data['subjects']:
        all_trials.extend(subject['trials'])
    
    # Sample 50 random trials
    selected = random.sample(all_trials, min(50, len(all_trials)))
    
    # Plot each trial
    for trial in selected:
        head_positions = np.array(trial['head_positions'])
        target_az = trial['target_az']
        relative_orientation = head_positions - target_az
        time = np.arange(-len(head_positions),0) * 0.1
        
        ax.plot(time, relative_orientation, color=colors[name], 
                alpha=0.3, linewidth=1)
    
    # Styling
    ax.axhline(y=0, color='red', linestyle='--', linewidth=3, label='Target')
    # ax.set_xlabel('Time (s)', fontsize=20)
    # ax.set_ylabel('Head Orientation Relative to Target (°)', fontsize=20)
    ax.tick_params(axis='both', labelsize=22)
    ax.set_title(f'{name}', fontsize=30, fontweight='bold')
    ax.set_ylim(-180, 180)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=20)
    fig.supxlabel('Time (s)', fontsize=27)
    fig.supylabel('Head relative to target (°)', fontsize=27)

plt.tight_layout()
plt.savefig(os.path.join(output_dir,'trajectory_comparison_50.png'), dpi=300, bbox_inches='tight')
plt.show()

print("Comparison plot saved!")

###############################################################################
# Now plot all three with ALL samples

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.patch.set_facecolor('white')

for idx, (name, path) in enumerate(conditions.items()):
    ax = axes[idx]
    
    with open(path, 'r') as f:
        data = json.load(f)
    
    # Collect all trials
    all_trials = []
    for subject in data['subjects']:
        all_trials.extend(subject['trials'])
    
    print(f"{name}: plotting {len(all_trials)} trials")

    for trial in all_trials:
        head_positions = np.array(trial['head_positions'])
        target_az = trial['target_az']
        relative_orientation = head_positions - target_az
        time = np.arange(-len(head_positions),0) * 0.1 #change to (len(head_positions))*0.1 if want to plot locked to trial start 
        
        ax.plot(time, relative_orientation, color=colors[name], 
                alpha=0.2, linewidth=0.5)  # Lower alpha, thinner lines
    
    # Styling
    ax.axhline(y=0, color='red', linestyle='--', linewidth=3, label='Target')
    # ax.set_xlabel('Time (s)', fontsize=14)
    ax.set_ylabel('Head Orientation Relative to Target (°)', fontsize=14)
    # ax.set_title(f'{name} (n={len(all_trials)})', fontsize=16, fontweight='bold')
    ax.set_title(f'{name}', fontsize=16, fontweight='bold')
    ax.set_ylim(-180, 180)
    ax.set_xlim(-30, 0) #change to 0, 30 if want to plot locked to trial start
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    fig.supxlabel('Time (s)', fontsize=20)

plt.tight_layout()
plt.savefig(os.path.join(output_dir,'trajectory_comparison_ALL_end_lock.png'), dpi=300, bbox_inches='tight')#rename to 'trajectory_comparison_ALL_start_lock.png' if want to plot locked to trial start
plt.show()

print("All trials plotted!")