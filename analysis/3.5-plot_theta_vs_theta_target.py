# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 17:38:42 2026

get a plot of head locations (theta) against 
desired head locations(theta_target) based on the KF estimate (x)
@author: sophie_chan
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os

#load json
json_path = "C:/Users/sophie_chan/ski.org Dropbox/Sofia Krasovskaya/echolocalization/KF v.Jan26.1 - remove sinusoid/sim_test_small"
json_file = os.path.join(json_path, "test_experiment_results.json")

with open(json_file, 'r') as f:
    results = json.load(f)
    
    
#select subject & trial to plot
sub_idx = 0
trial_idx = 50

#extract data from json file
trial = results['subjects'][sub_idx]['trials'][trial_idx]
theta = np.array(trial['head_positions'])
theta_target = np.array(trial['theta_target_positions'])
target_az = trial['target_az']
num_steps = trial['num_steps_used']

#create time axis in seconds
time = np.arange(num_steps)*0.1


#plot
plt.figure(figsize=(14,8))

#plot head positions theta 
plt.plot(time, theta, 'b-o',  label = "theta", markersize =3 )

#plot target head positions theta_target
plt.plot(time, theta_target, 'g-s',  label = "theta_target", markersize = 3)

#plot true target location (ground truth)
plt.axhline(y = target_az, color = 'r',linestyle = '--', linewidth = 2, label = f'True target location ({target_az:.1f}°)')

# Add KF estimate to see belief evolution
if 'kf_estimates' in trial:
    kf_estimates = np.array(trial['kf_estimates'])
    plt.plot(time, kf_estimates, 'c-^', markersize = 3, 
             label='x (KF belief estimate)', alpha=0.6)

#format plot

plt.xlabel('Time (s)', fontsize = 16)
plt.ylabel('Azimuth (°)', fontsize = 16)
plt.legend()
plt.grid(True, alpha = 0.3)
plt.ylim(-100,100)
plt.tight_layout()
plt.show()

plt.savefig(os.path.join(json_path, 'theta_vs_theta_target_without_sin.png'),
            dpi=100, bbox_inches='tight')
plt.close()
