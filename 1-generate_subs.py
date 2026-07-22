# -*- coding: utf-8 -*-
"""

Generate consistent subject parameters and target positions for echolocation experiments.
Run this script ONCE before running any condition scripts.

This ensures all three conditions (control, test_big, test_small) use:
- Same subjects (same individual parameters)
- Same target positions for each trial

Created on Mon Jul  7 21:48:06 2025
@author: sophie_chan
"""

import numpy as np
import json


def generate_subject_parameters(num_subs, random_seed=42):
    """Generate consistent subject parameters for use across all conditions"""
    np.random.seed(random_seed)

    subject_params = []
    for i in range(num_subs):
        params = {
            'subject_id': i + 1,
            # Control condition parameters
            'drift_std': 1.0,
            'Q': 0.1,
            # Test condition parameters (unused in control)
            # 'R_base': np.random.uniform(3.0, 8.0),
            'measurement_noise_std': 5.0, ## σ_base in paper
            'base_click_frequency': 1.3,
            'angle_tolerance': 0.5
            # 'subject_id': i + 1,
            # # Control condition parameters
            # 'drift_std': np.random.uniform(0.5, 2.0),
            # 'Q': np.random.uniform(0.05, 0.2),
            # # Test condition parameters (unused in control)
            # # 'R_base': np.random.uniform(3.0, 8.0),
            # 'measurement_noise_std': np.random.uniform(1.0, 3.0), ## σ_base in paper
            # 'base_click_frequency': np.random.uniform(1.0, 2.0),
            # 'angle_tolerance': 0.5
        }
        subject_params.append(params)

    return subject_params


def generate_target_positions(num_trials, target_range=(-90, 90), random_seed=42):
    """Generate consistent target positions for use across all conditions"""
    np.random.seed(random_seed)
    return np.random.uniform(target_range[0], target_range[1], num_trials)


if __name__ == "__main__":
    # Experimental parameters
    NUM_SUBS = 1
    NUM_TRIALS = 500
    RANDOM_SEED = 42
    TARGET_RANGE = (-90, 90)

    print("=== GENERATING EXPERIMENTAL PARAMETERS ===")
    print(f"Number of subjects: {NUM_SUBS}")
    print(f"Number of trials per subject: {NUM_TRIALS}")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Target range: {TARGET_RANGE}")

    # Generate parameters
    subject_params = generate_subject_parameters(NUM_SUBS, RANDOM_SEED)
    target_positions = generate_target_positions(
        NUM_TRIALS, TARGET_RANGE, RANDOM_SEED)

    # Create experimental setup
    experimental_setup = {
        'num_subs': NUM_SUBS,
        'num_trials': NUM_TRIALS,
        'random_seed': RANDOM_SEED,
        'target_range': TARGET_RANGE,
        'subject_params': subject_params,
        'target_positions': target_positions.tolist()
    }

    # Save to file
    with open('experimental_setup.json', 'w') as f:
        json.dump(experimental_setup, f, indent=2)

    print(f"\n=== PARAMETERS GENERATED ===")
    print(f"Subject parameters saved to: experimental_setup.json")
    print(
        f"Example - Subject 1: drift_std={subject_params[0]['drift_std']:.2f}, Q={subject_params[0]['Q']:.3f}")
    print(f"Example - Trial 1: target at {target_positions[0]:.1f}°")
    print(f"Example - Trial 2: target at {target_positions[1]:.1f}°")

    print(f"\nNow you can run:")
    print(f"1. python control_condition.py")
    print(f"2. python test_big_condition.py")
    print(f"3. python test_small_condition.py")
    print(f"4. analysis")
