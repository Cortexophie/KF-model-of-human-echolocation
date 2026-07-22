# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 14:28:03 2026
@author: sofia krasovskaya
"""

    ##############################################################################
##############################################################################

    ########    #      #    ########
    #       #   #      #    #      #
    ########    #      #    #      #
    #     #     #      #    #      #
    #     ##    ########    #      #

##############################################################################
    ##############################################################################
'''
                            === SETUP INSTRUCTIONS ===

        1. Put your .wav files in a directory with angle-based names:
           Examples: 'az_3_.wav','0deg.wav', '5deg.wav', '-10deg.wav', 'angle_15.wav'
        2. Update the AUDIO_DIR path below to point to your files
        3. Run with RUN_AUDIO_TEST=True first to verify file detection
        4. Then RUN_SINGLE_TRIAL=True to test the full pipeline on one trial
        5. Finally RUN_FULL_EXPERIMENT=True for multi-subject analysis
        
        '''
import os
import numpy as np
import matplotlib.pyplot as plt


from model_engine import (
    simulate_test_condition_with_audio,
    simulate_control_condition,
    run_multi_subject_experiment_test,
    run_multi_subject_experiment_control
    )
from model_core import (
    test_audio_library_setup, 
    validate_audio_files, 
    DEFAULT_MAX_STEPS
    )

# ============================================================================
# EDIT BELOW THIS LINE
# ============================================================================
# ------------------------------------------------------------
# 1. CONDITION
# 'control' : no audio feedback
# 'big'     : audio feedback, large target
# 'small'   : audio feedback, small target
# ------------------------------------------------------------
CONDITION = 'small'             # SET TO: 'control', 'big', or 'small'


# ------------------------------------------------------------
# 2. PATH
# set path to audio directory here
# ------------------------------------------------------------

AUDIO_DIR = "C://add//path//to//your//stereo//files//wavs_1deg"

SAVE_PATH = None                # None = auto (./sim_control, ./sim_test_big, ./sim_test_small)



# ------------------------------------------------------------
# 3. WHAT TO RUN
# Control flag config - change these to control what to run:
# ------------------------------------------------------------
RUN_AUDIO_TEST   = False        # True = test if audio files are being found and read
RUN_SINGLE_TRIAL = False        # True = test the model with just a single trial
RUN_EXPERIMENT   = True         # Set to True when ready for full experiment


SINGLE_TRIAL_TARGET_AZ = 70     # some azimuth between +/-90, for single trial test only

# ============================================================================
# DO NOT EDIT BELOW THIS LINE
# ============================================================================

# Resolve save path from condition if not set
if SAVE_PATH is None:
    SAVE_PATH = {
        'control': './sim_control_condition',
        'big':     './sim_test_big',
        'small':   './sim_test_small'
    }[CONDITION]



# test if audio is set up correctly
if RUN_AUDIO_TEST:
    print("=== TESTING AUDIO LIBRARY SETUP ===")
    if CONDITION == 'control':
       print("Audio test not applicable for control condition")
    elif os.path.exists(AUDIO_DIR):
       test_audio_library_setup(AUDIO_DIR)
       validate_audio_files(AUDIO_DIR)
    else:
       print(f"Audio directory not found: {AUDIO_DIR}")
       print(
           "Please update the AUDIO_DIR path and ensure it contains .wav files")
       print("Expected naming: '0deg.wav', '5deg.wav', '-10deg.wav', etc.")

        

#  SINGLE TRIAL
if RUN_SINGLE_TRIAL:
    print(f"\n=== TESTING SINGLE TRIAL  — {CONDITION.upper()}===")
    
    
    if CONDITION == 'control':
        (head_pos, theta_target, measurements, kf_est,
         kf_var, n_steps) = simulate_control_condition(
            target_az=SINGLE_TRIAL_TARGET_AZ,
            max_steps=DEFAULT_MAX_STEPS
        )
        target_found = False
        click_events = []
        metadata = []
    
    else:
        if not os.path.exists(AUDIO_DIR):
            raise FileNotFoundError(f"Audio directory not found: {AUDIO_DIR}")

        head_pos, theta_target, measurements, kf_est, kf_var,\
         n_steps, click_events, metadata, target_found = simulate_test_condition_with_audio(
            audio_directory=AUDIO_DIR,
            target_az=SINGLE_TRIAL_TARGET_AZ,
            target_size=CONDITION,
            max_steps=DEFAULT_MAX_STEPS,
            angle_tolerance=0.5
        )



    print(f"Trial completed in {n_steps} steps")
    print(f"Final head position: {head_pos[-1]:.1f}°")
    print(f"Target was at: {SINGLE_TRIAL_TARGET_AZ}°")
    print(f"Target {'FOUND' if target_found else 'NOT FOUND'}")
    if CONDITION != 'control':
        print(f"Made {len(click_events)} clicks, \
              {len(metadata)} successful measurements")

    # CREATE MAIN PLOT
    fig, axes = plt.subplots(1, 1, figsize=(20, 12))
    steps = np.arange(1, n_steps + 1)

    # Plot head positions and KF estimates
    axes.plot(steps, head_pos, 'b-',
              label='Head Position', linewidth=2)
    axes.plot(steps, kf_est, 'cyan',
              label='KF estimate', linewidth=2)
    axes.axhline(SINGLE_TRIAL_TARGET_AZ, color='g',
                 linestyle='--', label='Target', linewidth=2)

    # Mark measurements
    valid_meas = ~np.isnan(measurements)
    if np.any(valid_meas):
        axes.plot(steps[valid_meas], measurements[valid_meas],
                  'go', markersize=8, label='Measurements')
        
        
    axes.set_xlabel('Step')
    axes.set_ylabel('Azimuth (°)')
    axes.set_title(f'Single trial — {CONDITION} condition, target at {SINGLE_TRIAL_TARGET_AZ}°')
    axes.legend()
    axes.grid(True)
    plt.tight_layout()
    plt.show()


# FULL EXPERIMENT
if RUN_EXPERIMENT:
    print(f"\n=== RUNNING FULL EXPERIMENT — {CONDITION.upper()}===")
        
    if CONDITION == 'control':
        run_multi_subject_experiment_control(
            save_path=SAVE_PATH,
            visualize=True,
            save_data=True
        )
        print("Control experiment complete.")

    else:
        if not os.path.exists(AUDIO_DIR):
            raise FileNotFoundError(f"Audio directory not found: {AUDIO_DIR}")

        run_multi_subject_experiment_test(
            audio_directory=AUDIO_DIR,
            target_size=CONDITION,
            save_path=SAVE_PATH,
            visualize=True,
            save_data=True,
            angle_tolerance=0.5
        )
        print(f"{CONDITION.capitalize()} target experiment complete.")
        print(f"Results saved to: {SAVE_PATH}")