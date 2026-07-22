# -*- coding: utf-8 -*-
"""
Edited on Jan 30th 2025
@author: sofia krasovskaya


This script contains simulation functions for the echolocation model.

Two simulate & experiment runner functions for three experimental conditions:

    simulate_control_condition()
        No acoustic feedback. The KF runs in predict-only mode (no
        measurement updates). The head explores via momentum and boundary
        reversals. Trials always run to max_steps (no early stopping). To run, 
        set CONDITION = 'control' in model_launch.py.

    simulate_test_condition_with_audio()
        Full closed-loop system with acoustic feedback. At each click the
        model loads the appropriate .wav file, extracts ITD via
        ITDILDProcessor, and feeds the resulting angle measurement into
        the KF update step. The KF estimate in turn steers the head.
        Trials end early when P converges and the head is near the target.

Which condition is simulated depends only on what you pass in — there is
no global TARGET_SIZE constant. set CONDITION = 'big' or 'small'
in model_launch.py to pass target_size to simulate_test_condition(); it propagates 
through reflectivity() to scale the measurement noise covariance R_k. 

Two corresponding experiment-runner functions loop over subjects and
trials, collect results, save JSON, and optionally produce plots:

    run_multi_subject_experiment_control()
    run_multi_subject_experiment_test()

Both load subject parameters from experimental_setup.json (produced by
1-generate_subs.py) so that the same subjects and target positions are
used across all conditions.
"""
#general imports
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import sem
import json
import gc

#model-specific imports
from KF import kf_predict, kf_update
from model_core import (
    DEFAULT_DT,
    DEFAULT_MAX_STEPS,
    SweepController,
    EchoAudioLibrary,
    ITDILDProcessor,
    load_experimental_setup,
    reflectivity,
    analyze_timing_results,
    convert_numpy_types,
    plot_fixed_view_rotating_head,    
    )


############################################################################
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# TEST CONDITION
# ('big' or 'small' target)
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
############################################################################

def simulate_test_condition_with_audio(
    audio_directory: str,
    target_az: float,
    target_size:str=None,
    max_steps:int=DEFAULT_MAX_STEPS,
    initial_head_az:float=0.0,
    drift_std:float=1.0,
    Q:float=0.1,
    measurement_noise_std:float=0.5, # σ_base in paper
    base_click_frequency:float=1.3,# Hz - EB's mean rate for big target trials(~.77s between clicks)
    angle_tolerance:float=0.5,
    max_amp:float=90.0,
    target_threshold:float=5.0,
    detection_threshold:int=3,
    rng=np.random.default_rng()
)-> tuple:
    """
    Simulate one trial of the test condition with acoustic feedback.

    Closed-loop sensorimotor system: at each click the model loads the
    .wav file for the current relative head-target angle, extracts ITD,
    converts it to an absolute angle measurement z, and feeds z into the
    KF update step. The posterior estimate x guides the head on the next
    step.
    
    Detection criterion
    -------------------
    The trial ends (target found) when BOTH:
      - P has plateaued for detection_threshold consecutive measurements
        (convergence_ratio = P_k / P_{k-1} >= 0.95), AND
      - The KF estimate x is within target_threshold degrees of head_az (i.e., 
         subjectively within threshold)
    
    Parameters
    ----------
    audio_directory : str
        Path to directory of angle-indexed stereo .wav files.
    target_az : float
        ground truth, i.e. the true target azimuth in degrees.
    target_size : str
        'big' or 'small'. Controls reflectivity and thus R_k scaling.
    max_steps : int
        Maximum trial length in timesteps. The default is DEFAULT_MAX_STEPS.
    initial_head_az : float
        Starting head position and initial KF estimate. The default is 0.0.
    drift_std : float
        Per-step motor noise std (passed to SweepController). The default is 1.0.
    Q : float
        KF process noise covariance (degrees^2) The default is 1.0.
    measurement_noise_std : float
        Base measurement noise σ_base (degrees). Scaled by reflectivity. The default is 5.0.
    base_click_frequency : float
        Mean click rate in Hz.The default is 1.3.
    angle_tolerance : float
        Maximum angular difference (in degrees) between requested and
        available audio file angle. The default is 0.5.
    max_amp : float
        Head azimuth limit (in degrees). The default is 90.0.
    target_threshold : float
        Head-target distance threshold for detection criterion (degrees).The default is 5.0.
    detection_threshold : int
        Number of consecutive converged measurements needed to stop. The default is 3.
    rng : numpy.random.Generator 


    Returns
    -------
    tuple
        (head_positions, theta_target_positions, measurements,
         kf_estimates, kf_variances, num_steps_used,
         click_events, measurement_metadata, target_found)
        Arrays are trimmed to num_steps_used.

    """
   
    # Get reflectivity factor for this target size
    reflect_factor = reflectivity(target_size)
    print(f"Target size: {target_size}, Reflectivity: {reflect_factor}")

    # Initialize audio system
    audio_library = EchoAudioLibrary(audio_directory)
    itd_processor = ITDILDProcessor()

    if len(audio_library.available_angles) == 0:
        raise ValueError(f"No audio files found in {audio_directory}")

    # Initialize arrays (same as control)
    head_positions = np.zeros(max_steps)
    measurements = np.full(max_steps, np.nan)
    kf_estimates = np.zeros(max_steps)
    kf_variances = np.zeros(max_steps)
    theta_target_positions = np.zeros(max_steps) 
    

    # Initialize sweep controller (same as control)
    sweep_controller = SweepController(
        max_amp=max_amp,
        min_vel=0.0,
        max_vel=50.0,
        noise_std=drift_std * 0.5,
        rng=rng
    )
    
    
    #reset phase so each trial has a different phase offset
    # sweep_controller.phase = rng.uniform(0, 2*np.pi)


    # Initialize click timing system
    click_interval = 1.0 / base_click_frequency  # seconds between clicks
   
    # Randomize first click timing - occurs between 0.1s and full interval
    time_since_last_click = rng.uniform(0.0, click_interval * 0.9)
    next_click_interval = click_interval  

    # Kalman filter initialization
    # x = target location estimate
    # Initial guess: target is at 0 degrees (unknown location)
    x = initial_head_az
    P = 500.0  # High initial uncertainty about target location

    # Track click events for analysis
    click_events = []
    measurement_metadata = []

    # Add target detection
    # Track for detection criterion
    # Track for convergence-based detection
    P_prev = None
    convergence_count = 0  # Count consecutive converged measurements
    convergence_threshold = 0.95  # P_k+1/P_k >= 0.95 means plateaued
    target_found = False
    
    #store initial consitions at k=0
    head_positions[0] = sweep_controller.position #should be 0 deg
    measurements[0] = np.nan #no measurement yet
    kf_estimates[0] = x # initial_head_az (should be 0)
    kf_variances [0]= P #500
    theta_target_positions[0] = sweep_controller._calculate_target_pos()

    # Main simulation loop
    num_steps_used = 1 # start at 1 instead of 0 to preserve state 0  and avoid overwriting it

    for k in range(1, max_steps):
        # Prediction step
        x_pred, P_pred = kf_predict(x, P, Q)
        
        # Get head position from controller
        current_az = sweep_controller.step()
        theta_target = sweep_controller._calculate_target_pos()

        # CLICK DECISION-MAKER
        # Update click timing
        time_since_last_click += DEFAULT_DT

        # Determine if we should click based on frequency
        # make sure clicks not too close together
        should_click = time_since_last_click >= next_click_interval

        if should_click:
            # set click rate to 1.3 Hz - the mean DK had in big target trials
            next_click_interval = 1.0/base_click_frequency
            time_since_last_click = 0.0  # reset
            make_click = True
        else:
            make_click = False

        measurement_available = False
        if make_click:
            # Calculate relative angle between head and target
            # The audio files represent target position relative to a head pointing at 0°
            # So we need to find which audio file represents the current head-target relationship
            relative_angle_to_target = target_az - current_az

            # Wrap angle to [-180, 180] range to match audio file naming
            while relative_angle_to_target > 180:
                relative_angle_to_target -= 360
            while relative_angle_to_target < -180:
                relative_angle_to_target += 360

            # Get audio file for relative angle given current head position
            audio_info, used_angle = audio_library.get_audio_for_head_angle(
                relative_angle_to_target)

            if audio_info is not None:
                angle_diff = abs(relative_angle_to_target - used_angle)

                if angle_diff <= angle_tolerance:
                    # Process audio to get ITD/ILD measurement
                    result = itd_processor.extract_itd_ild_from_audio(
                        audio_info['audio_data'], audio_info['fs']
                    )

                    if result is not None:
                        # We have a measurement!
                        measurement_available = True

                        # Quality-based measurement noise
                        # high quality (1.0) = excellent echo, low noise -> trust measurement
                        # low qual (0.1) = poor echo, high noise -> KF puts less weight on this measurement
                        base_quality = result['quality']

                        # Apply reflectivity effects to quality
                        # Use sqrt scaling (signal strength scales with area)
                        quality_factor = np.sqrt(reflect_factor)
                        enhanced_quality = base_quality * quality_factor

                        # Get target angle measurement and add noise
                        # We have a measurement of target angle relative to current head position - result['angle_degrees']
                        relative_target_angle = result['angle_degrees']
                        
                        # Convert it to absolute coordinates for KF
                        absolute_target_angle = current_az + relative_target_angle
               
                        actual_measurement_noise = measurement_noise_std / np.sqrt(reflect_factor) # σ_measurement in paper, eq.4
                        

                       
                        # Use reflectivity-based enhanced quality for R_k calculation
                        # Higher R_k = noisier measurement → Lower Kalman gain → Less influence on estimate
                        # Lower R_k = less noisy → Higher Kalman gain → More influence on estimate
                        # Inverse relationship - lower quality = much higher R_k
                        # Prevent division by zero
                        R_k = (actual_measurement_noise**2) / max(enhanced_quality, 0.05) # eq.6
                        
                        # get noisy measurement (eq. 5 in paper)
                        z = absolute_target_angle + \
                            np.random.normal(0, R_k)
                        

                        # Measurement update
                        x, P = kf_update(x_pred, P_pred, z, R_k)
                        
                        
                        # Check for convergence (whether P has plateaued or not)
                        if P_prev is not None:
                            P_ratio = P / P_prev
                            
                            if P_ratio >= convergence_threshold:
                                # P is no longer decreasing significantly
                                convergence_count += 1
                                print(f"  Convergence check: P_ratio={P_ratio:.3f}, count={convergence_count}/{detection_threshold}")
                            else:
                                # P still decreasing, reset counter
                                convergence_count = 0
                        else:
                            # First measurement - initialize
                            convergence_count = 0
                        
                        # Update P_prev for next iteration
                        P_prev = P

                            
                        # Update controller with current target estimate
                        sweep_controller.set_kf_target_estimate(float(x))
                        # Update controller with current uncertainty level
                        sweep_controller.set_current_uncertainty(float(P))
                        
                        # set guidance with adjusted bias influence
                        sweep_controller.set_echo_guidance(    
                            enabled=True
                        )
                        
                        measurements[k] = z
                        print(
                            f"STORE MEASUREMENT: step={k}, head_pos={current_az:.1f}°, relative={relative_target_angle:.1f}°, absolute_calc={absolute_target_angle:.1f}°, final_z={z:.1f}°")

                   
                        measurement_accurate = abs(x - current_az) <= target_threshold
                        
                        # Detect when P has plateaued AND model thinks head is pointing at target (subjective)
                        if convergence_count >= detection_threshold and measurement_accurate:
                            target_found = True
                            print(f"Target found! P converged ({convergence_count} consecutive measurements with ratio >= {convergence_threshold}), P={P:.2f}°²")
                            break
                        
                        # Record metadata
                        click_events.append({
                            'step': k,
                            'head_angle': current_az,
                            'audio_angle': used_angle,
                            'success': True
                        })

                        measurement_metadata.append({
                            'step': k,
                            'head_angle': current_az,
                            'audio_angle': used_angle,
                            'measurement': z,
                            'raw_angle': result['angle_degrees'],
                            'ITD': result['ITD_seconds'],
                            'ILD': result['ILD_db'],
                            'base_quality': base_quality,
                            'enhanced_quality': enhanced_quality,
                            'reflectivity': reflect_factor,
                            'target_size': target_size
                        })

                # Update controller with current uncertainty level
                sweep_controller.set_current_uncertainty(float(P))

                # Record click attempt even if failed
                if not measurement_available:
                    click_events.append({
                        'step': k,
                        'head_angle': current_az,
                        'audio_angle': used_angle,
                        'success': False,
                        'reason': 'angle_too_far' if angle_diff > angle_tolerance else 'processing_failed'
                    })

        # No measurement case - then just use prediction
        if not measurement_available:
            x = x_pred
            P = P_pred

            # Disable echo guidance if no recent measurements
            if k > 10 and not np.any(~np.isnan(measurements[max(0, k-10):k])):
                sweep_controller.set_echo_guidance(enabled=False)

        # Store results
        head_positions[k] = current_az
        theta_target_positions[k] = theta_target
        kf_estimates[k] = x.item() if hasattr(  # .item() extracts the scalar value from np array
            x, 'item') else x  # This is target location estimate
        kf_variances[k] = P.item() if hasattr(P, 'item') else P

        num_steps_used = k + 1

    # Trim arrays
    head_positions = head_positions[:num_steps_used]
    theta_target_positions = theta_target_positions[:num_steps_used] 
    measurements = measurements[:num_steps_used]
    kf_estimates = kf_estimates[:num_steps_used]
    kf_variances = kf_variances[:num_steps_used]

    return (head_positions, theta_target_positions, measurements, kf_estimates, kf_variances,
            num_steps_used, click_events, measurement_metadata, target_found)


# ----------------------------------------------------------------------------
# Experiment Function 
# ----------------------------------------------------------------------------

def run_multi_subject_experiment_test(
    audio_directory,
    target_size=None,
    save_path=None,
    visualize=True,
    num_trials_to_visualize=3,
    show_every_n_clicks=1,
    save_data=True,
    condition='test',
    angle_tolerance=0.5
):
    """
    Run multi-subject experiment with audio feedback using consistent parameters.
    """

    # Load consistent experimental setup
    setup = load_experimental_setup()
    if setup is None:
        return None

    # Extract parameters from setup
    num_subs = setup['num_subs']
    num_trials = setup['num_trials']
    subject_params = setup['subject_params']
    target_positions = setup['target_positions']
    max_steps = DEFAULT_MAX_STEPS

    print(f"=== RUNNING {target_size.upper()} TARGET EXPERIMENT ===")
    print(f"Loaded setup: {num_subs} subjects, {num_trials} trials")
    print(
        f"Target size: {target_size}, Reflectivity: {reflectivity(target_size)}")

    import matplotlib
    matplotlib.use('Agg')

    # Setup directories
    if save_path and (visualize or save_data):
        os.makedirs(save_path, exist_ok=True)

    # Pick trials to visualize
    if num_trials_to_visualize > num_trials:
        trials_to_visualize = list(range(num_trials))
    else:
        trials_to_visualize = sorted(np.random.choice(
            range(num_trials), size=num_trials_to_visualize, replace=False
        ).tolist())

    # Initialize results
    results = {
        'experiment_params': {
            'num_subs': num_subs,
            'num_trials': num_trials,
            'max_steps': max_steps,
            'trial_duration_seconds': max_steps * DEFAULT_DT,
            'dt': DEFAULT_DT,
            'target_range': setup['target_range'],
            'condition': f'test_{target_size}',
            'target_size': target_size,
            'reflectivity': reflectivity(target_size),
            'audio_directory': audio_directory,
            'angle_tolerance': angle_tolerance,
            'random_seed': setup['random_seed']
        },
        'target_positions': target_positions,
        'subject_params': subject_params,
        'subjects': []
    }

    # Run simulations for each subject (same loop structure as control)
    for sub_idx in range(num_subs):
        sub_params = subject_params[sub_idx]

        # make an RNG per subject
        rng = np.random.default_rng(setup['random_seed'] + sub_idx)

        sub_dir = None
        if save_path and (visualize or save_data):
            sub_dir = os.path.join(save_path, f'subject_{sub_idx+1}')
            os.makedirs(sub_dir, exist_ok=True)

        print(f"Running simulations for Subject {sub_idx+1}/{num_subs}...")

        subject_data = {
            'id': sub_idx + 1,
            'params': sub_params,
            'trials': []
        }

        # Run trials for this subject
        for trial_idx in range(num_trials):
            target_az = target_positions[trial_idx]

            trial_dir = None
            if sub_dir and visualize and (trial_idx in trials_to_visualize):
                trial_dir = os.path.join(sub_dir, f'trial_{trial_idx+1}')
                os.makedirs(trial_dir, exist_ok=True)

            print(
                f"  - Subject {sub_idx+1}, Trial {trial_idx+1}: target {target_az:.1f}°")

            # Run simulation
            try:
                (head_positions, theta_target_positions, measurements, kf_estimates, kf_variances,
                 num_steps_used, click_events, measurement_metadata, target_found) = simulate_test_condition_with_audio(
                    audio_directory=audio_directory,
                    target_az=target_az,
                    target_size=target_size,
                    max_steps=max_steps,
                    # filter out the subject_id from parameters (sim function does not expect that param)
                    **{k: v for k, v in sub_params.items() if k != 'subject_id'},
                    rng=rng
                )

                # Store trial data
                trial_data = {
                    'id': trial_idx + 1,
                    'target_az': target_az,
                    'target_size': target_size,
                    'reflectivity': reflectivity(target_size),
                    'head_positions': head_positions.tolist(),
                    'theta_target_positions': theta_target_positions.tolist(),
                    'measurements': [x if not np.isnan(x) else None for x in measurements],
                    'kf_estimates': kf_estimates.tolist(),
                    'kf_variances': kf_variances.tolist(),
                    'num_steps_used': num_steps_used,
                    'num_clicks_made': len(click_events),
                    'successful_measurements': len(measurement_metadata),
                    'click_events': click_events,
                    'measurement_metadata': measurement_metadata,
                    'target_found': target_found  # true if target_found = T
                }
                subject_data['trials'].append(trial_data)

                # Visualization (same as control but adapted for measurements)
                if trial_dir:
                    # Create summary plot
                    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

                    steps = np.arange(1, num_steps_used + 1)

                    # Plot 1: Head positions and estimates
                    axes[0].plot(steps, head_positions, 'b-o',
                                 linewidth=2, label='Head Position', markersize=3)
                    axes[0].plot(steps, kf_estimates, 'r-s',
                                 linewidth=2, label='KF Estimate', markersize=3)
                    axes[0].axhline(
                        y=target_az, color='g', linestyle='--', linewidth=2, label='True Target')

                    # Add confidence intervals
                    upper_bound = kf_estimates + 2 * np.sqrt(kf_variances)
                    lower_bound = kf_estimates - 2 * np.sqrt(kf_variances)
                    axes[0].fill_between(
                        steps, lower_bound, upper_bound, color='r', alpha=0.2, label='95% Confidence')

                    # Mark successful clicks
                    successful_clicks = [
                        e for e in click_events if e['success']]
                    if successful_clicks:
                        click_steps = [e['step'] +
                                       1 for e in successful_clicks]
                        click_positions = [head_positions[e['step']]
                                           for e in successful_clicks]
                        axes[0].plot(click_steps, click_positions, 'ko',
                                     markersize=6, label='Successful Clicks')

                    axes[0].set_ylabel('Azimuth (degrees)')

                    target_status = "found" if target_found else "not found"
                    axes[0].set_title(
                        f'Test condition with audio feedback. Subject {sub_idx+1}, Trial {trial_idx+1}: Target at {target_az:.1f}° ({target_status})')

                    axes[0].legend()
                    axes[0].grid(True)

                    # Plot 2: Measurements
                    valid_measurements = ~np.isnan(measurements)
                    if np.any(valid_measurements):
                        meas_steps = steps[valid_measurements]
                        meas_values = measurements[valid_measurements]
                        axes[1].plot(meas_steps, meas_values, 'go',
                                     markersize=6, label='Audio Measurements')
                        axes[1].axhline(y=target_az, color='g',
                                        linestyle='--', alpha=0.5)

                    axes[1].set_ylabel('Measured Angle (degrees)')
                    axes[1].set_title('Echo-based Angle Measurements')
                    axes[1].legend()
                    axes[1].grid(True)

                    # Plot 3: Error over time
                    error = np.abs(head_positions - target_az)
                    axes[2].plot(steps, error, 'k-', linewidth=2)
                    axes[2].set_xlabel('Step')
                    axes[2].set_ylabel('Absolute Error (degrees)')
                    axes[2].set_title('Localization Error Over Time')
                    axes[2].grid(True)

                    plt.tight_layout()
                    fig.savefig(os.path.join(trial_dir, 'summary.png'),
                                dpi=100, bbox_inches='tight')
                    plt.close(fig)

                    # Polar visualization
                    steps_to_visualize = range(
                        0, num_steps_used, show_every_n_clicks)
                    for step_idx in steps_to_visualize:
                        fig = plot_fixed_view_rotating_head(
                            head_positions, target_az, condition, alpha=2.0, beta=2.4, step_idx=step_idx
                        )
                        fig_path = os.path.join(
                            trial_dir, f'Click_{step_idx+1:02d}.png')
                        fig.savefig(fig_path, dpi=100, bbox_inches='tight')
                        plt.close(fig)

                    plt.close('all')
                    gc.collect()

            except Exception as e:
                print(f"    Error in trial {trial_idx+1}: {e}")
                continue

        results['subjects'].append(subject_data)

        # Subject summary
        if visualize and sub_dir:
            fig, ax = plt.subplots(figsize=(12, 8))

            # Collect data on number of steps used for each trial
            steps_used = [t['num_steps_used'] for t in subject_data['trials']]
            max_steps_used = max(steps_used)
            
            # Plot individual trials aligned to trial end
            for trial_idx, trial_data in enumerate(subject_data['trials']):
                target_az = trial_data['target_az']
                head_positions = np.array(trial_data['head_positions'])
                trial_length = len(head_positions)
                
                # Compute error at each step
                errors = np.abs(head_positions - target_az)
                
                # Create time axis: from [- to 0]
                time_before_end = np.arange(-trial_length, 0)
                
                # Plot with light transparency for individual trials
                ax.plot(time_before_end, errors, 'o-', linewidth=1, alpha=0.3,
                        label=f'Trial {trial_idx+1}' if trial_idx < 5 else None)
                
            # Compute average error across trials aligned to trial end
            mean_errors = np.zeros(max_steps_used)
            std_errors = np.zeros(max_steps_used)

            for relative_idx in range(max_steps_used):
                errors_at_step = []
                
                for t in subject_data['trials']:
                    trial_length = len(t['head_positions'])
                    # Calculate absolute position in this trial
                    absolute_idx = trial_length - 1 - relative_idx
                    
                    if absolute_idx >= 0: # If trial was long enough
                        error = np.abs(t['head_positions'][absolute_idx] - t['target_az'])
                        errors_at_step.append(error)
                
                if errors_at_step:
                    mean_errors[relative_idx] = np.mean(errors_at_step)
                    std_errors[relative_idx] = np.std(errors_at_step)
                else:
                    mean_errors[relative_idx] = np.nan
                    std_errors[relative_idx] = np.nan

            # Reverse arrays so index 0 = earliest, index -1 = latest
            mean_errors = mean_errors[::-1]
            std_errors = std_errors[::-1]
            
            # Plot with x-axis showing steps before trial end
            steps_range = np.arange(-max_steps_used + 1, 1)
            valid_indices = ~np.isnan(mean_errors)
            
            #plot mean error
            ax.plot(steps_range[valid_indices], mean_errors[valid_indices],
                    'k-', linewidth=3, label='Mean Error')
            
            # Plot ±1 SD
            ax.fill_between(
                steps_range[valid_indices],
                mean_errors[valid_indices] - std_errors[valid_indices],
                mean_errors[valid_indices] + std_errors[valid_indices],
                color='k', alpha=0.2, label='±1 SD'
            )

            ax.set_xlabel('Step Number before trial end')
            ax.set_ylabel('Absolute Error (°)')
            ax.set_title(
                f'{target_size} condition. Subject {sub_idx+1}: Error across {num_trials} Trials')

            # Add parameters to figure
            param_text = '\n'.join([
                f"Drift: {sub_params['drift_std']:.2f}",
                f"Q: {sub_params['Q']:.2f}",
                f"Meas noise: {sub_params['measurement_noise_std']:.2f}",
                f"Angle tol: {sub_params['angle_tolerance']:.2f}",
                f"Avg steps: {np.mean(steps_used):.1f}"
            ])
            props = dict(boxstyle='round', facecolor='white', alpha=0.7)
            ax.text(0.95, 0.95, param_text, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', horizontalalignment='right', bbox=props)

            # plot legend
            if num_trials <= 10:
                ax.legend()
            else:
                # Only show first 5 trials plus mean in legend
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles[:5] + handles[-2:], labels[:5] + labels[-2:])

            ax.grid(True)
            ax.set_ylim(bottom=0)  # Error can't be negative

            fig.savefig(os.path.join(sub_dir, 'all_trials_summary.png'),
                        dpi=100, bbox_inches='tight')
            plt.close(fig)
            
            # Clear memory after subject is complete
            plt.close('all')
            gc.collect()

    # Create overall experiment summary plot
    if visualize and save_path:
        fig, ax = plt.subplots(figsize=(12, 8))

        # Find the maximum number of steps used across all subjects/trials
        max_steps_overall = max([
            max([len(t['head_positions']) for t in subject_data['trials']])
            for subject_data in results['subjects']
        ])

        # Calculate mean error for each subject
        for sub_idx, subject_data in enumerate(results['subjects']):
            # Calculate mean error at each step position
            mean_errors = np.zeros(max_steps_overall)
            sem_errors = np.zeros(max_steps_overall)

            for relative_idx in range(max_steps_overall):
                
                # Collect errors for this step across all trials that lasted this long
                errors_at_step = []
                for t in subject_data['trials']:
                    trial_length = len(t['head_positions'])
                    absolute_idx = trial_length - 1 - relative_idx

                    if absolute_idx >= 0:
                        error = np.abs(t['head_positions']
                                       [absolute_idx] - t['target_az'])
                        errors_at_step.append(error)

                if errors_at_step:  # If we have data for this step
                    mean_errors[relative_idx] = np.mean(errors_at_step)
                    sem_errors[relative_idx] = sem(errors_at_step)
                else:
                    # No data for this step (all trials ended before this point)
                    mean_errors[relative_idx] = np.nan
                    sem_errors[relative_idx] = np.nan

            # Reverse arrays
            mean_errors = mean_errors[::-1]
            sem_errors = sem_errors[::-1]

            # Plot using masked array to handle NaN values
            valid_indices = ~np.isnan(mean_errors)
            steps_range = np.arange(-max_steps_overall + 1, 1)

            ax.plot(steps_range[valid_indices], mean_errors[valid_indices],
                    'o-', linewidth=2, label=f'Subject {sub_idx+1}')

            # Add error bars for standard error (less cluttered than fill_between)
            if num_subs <= 5:  # Only add error bars if we have few subjects
                ax.errorbar(
                    steps_range[valid_indices],
                    mean_errors[valid_indices],
                    yerr=sem_errors[valid_indices],
                    fmt='none', alpha=0.3
                )

        # Calculate grand average across all subjects and trials
        grand_mean = np.zeros(max_steps_overall)
        grand_sem = np.zeros(max_steps_overall)

        for relative_idx in range(max_steps_overall):
            # Collect all errors for this step index across all subjects and trials
            all_errors = []
            for subject_data in results['subjects']:
                for trial_data in subject_data['trials']:
                    trial_length = len(trial_data['head_positions'])
                    # Calculate absolute position in this trial
                    absolute_idx = trial_length - 1 - relative_idx

                    if absolute_idx >= 0:  # If this trial was long enough to have this relative position
                        error = abs(
                            trial_data['head_positions'][absolute_idx] - trial_data['target_az'])
                        all_errors.append(error)

            if all_errors:  # If we have data for this step
                grand_mean[relative_idx] = np.mean(all_errors)
                grand_sem[relative_idx] = sem(all_errors)
            else:
                grand_mean[relative_idx] = np.nan
                grand_sem[relative_idx] = np.nan

        # Reverse the arrays so index 0 is the earliest point
        grand_mean = grand_mean[::-1]
        grand_sem = grand_sem[::-1]

        # Plot grand average using masked array
        valid_indices = ~np.isnan(grand_mean)
        
        # X-axis: negative numbers counting down to 0
        # e.g., -299, -298, ..., -1, 0
        steps_range = np.arange(-max_steps_overall + 1, 1)

        ax.plot(steps_range[valid_indices], grand_mean[valid_indices],
                'k-', linewidth=3, label='Grand Mean')

        ax.fill_between(
            steps_range[valid_indices],
            grand_mean[valid_indices] - grand_sem[valid_indices],
            grand_mean[valid_indices] + grand_sem[valid_indices],
            color='k', alpha=0.2
        )

        ax.set_xlabel('Step Number before trial end')
        ax.set_ylabel('Absolute Error (degrees)')
        ax.set_title(
            f'{target_size} condition with audio. Error across {num_subs} Subjects ({num_trials} Trials Each)')

        if num_subs <= 10:
            ax.legend()
        else:
            # Only show a subset of subjects in the legend
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles[:5] + handles[-1:], labels[:5] + labels[-1:])

        ax.grid(True)
        ax.set_ylim(bottom=0)  # Error can't be negative

        fig_path = os.path.join(save_path, 'experiment_summary.png')
        fig.savefig(fig_path, dpi=100, bbox_inches='tight')
        plt.close(fig)

        # Add histogram of number of steps used
        fig, ax = plt.subplots(figsize=(10, 6))
        all_steps_used = []
        for subject_data in results['subjects']:
            all_steps_used.extend([t['num_steps_used']
                                  for t in subject_data['trials']])

        ax.hist(all_steps_used, bins=range(1, max_steps+2), alpha=0.7)
        ax.axvline(x=np.mean(all_steps_used), color='r', linestyle='--',
                   label=f'Mean: {np.mean(all_steps_used):.1f} steps')

        ax.set_xlabel('Number of Steps Used')
        ax.set_ylabel('Frequency')
        ax.set_title(
            'Distribution of Steps Used Across All Subjects and Trials')
        ax.legend()
        ax.grid(True)

        fig_path = os.path.join(save_path, 'steps_histogram.png')
        fig.savefig(fig_path, dpi=100, bbox_inches='tight')
        plt.close(fig)

    # Save results
    if save_data and save_path:

        results_json = {
            'experiment_params': convert_numpy_types(results['experiment_params']),
            'target_positions': results['target_positions'],
            'subject_params': convert_numpy_types(results['subject_params']),
            'subjects': convert_numpy_types(results['subjects']),
            'condition': results['experiment_params']['condition']
        }
        results_path = os.path.join(
            save_path, 'test_experiment_results.json')
        with open(results_path, 'w') as f:
            json.dump(results_json, f, indent=2)

        plt.close('all')
        gc.collect()

    # Timing analysis
    analyze_timing_results(results)

    return results



############################################################################
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Control
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
############################################################################

def simulate_control_condition(
    target_az,
    max_steps=DEFAULT_MAX_STEPS,
    initial_head_az=0.0,
    drift_std=1.0,
    Q=1.0,
    # threshold params are for symmetry with test condition function, 
    # but are not used in control since there's no measurement:
    target_threshold=5.0,
    detection_threshold=3,
    max_amp=90.0,
    rng=np.random.default_rng()
):
    """
    Control condition: The subject does NOT make clicks, so there is no measurement
    to feed into the Kalman Filter. We keep the same stepping loop, but skip
    the measurement update entirely.
    Parameters:
    -----------
    target_az : float
        True target location in degrees (unused except for reference)
    max_steps : int
        Maximum number of 'steps' to take
    initial_head_az : float
        Initial head azimuth
    drift_std : float
        Random drift standard deviation each step
    Q : float
        Process noise for the Kalman Filter
  
    Returns:
    --------
    tuple
        (head_positions, measurements, kf_estimates, kf_variances, num_steps_used)
    """
    rng = rng = rng or np.random.default_rng()
    
    trial_duration = max_steps * DEFAULT_DT
    print(f"Trial duration: {trial_duration} seconds")

    # Initialize arrays
    head_positions = np.zeros(max_steps)
    # No real measurement in ctrl, so fill with NaN to signify "no measurement"
    measurements = np.full(max_steps, np.nan)
    kf_estimates = np.zeros(max_steps)
    kf_variances = np.zeros(max_steps)
    theta_target_positions = np.zeros(max_steps)

    # Initialize sweep controller
    sweep_controller = SweepController(
        max_amp=max_amp,
        min_vel=1.0,
        max_vel=50.0,
        noise_std=drift_std * 0.5,  # Use drift_std to influence noise
        rng=rng
    )

    #               ***********

    # Kalman filter state initialization
    x = initial_head_az  # Initial estimate starts at head position
    P = 500.0            # Initial uncertainty (covariance)

    #               ***********

    #store initial consitions at k=0
    head_positions[0] = sweep_controller.position #should be 0 deg
    measurements[0] = np.nan #no measurement yet
    kf_estimates[0] = x # initial_head_az (should be 0)
    kf_variances [0]= P #500
    theta_target_positions[0] = sweep_controller._calculate_target_pos()
    

    # Main simulation loop
    num_steps_used = 1 # start at 1 instead of 0 to preserve state 0  and avoid overwriting it

    for k in range(1, max_steps):
        # Prediction step (formally here, but no real update without measurements)
        x_pred, P_pred = kf_predict(x, P, Q)

        # Get head position from controller
        current_az = sweep_controller.step()
        theta_target = sweep_controller._calculate_target_pos()

        # Formal measurement update (no actual measurement)
        x, P = x_pred, P_pred


        # Store results
        head_positions[k] = current_az
        theta_target_positions[k] = theta_target
        kf_estimates[k] = x
        kf_variances[k] = P

        #in ctrl cond always continue to max_steps (no early stopping)
        num_steps_used = k + 1  # Record step count


    # Trim arrays
    head_positions = head_positions[:num_steps_used]
    theta_target_positions = theta_target_positions[:num_steps_used]
    measurements = measurements[:num_steps_used]
    kf_estimates = kf_estimates[:num_steps_used]
    kf_variances = kf_variances[:num_steps_used]

    return head_positions, theta_target_positions, measurements, kf_estimates, kf_variances, num_steps_used



# ----------------------------------------------------------------------------
# Experiment Function - Control Condition
# ----------------------------------------------------------------------------
def run_multi_subject_experiment_control(
    save_path=None,
    visualize=True,
    num_trials_to_visualize=3,
    show_every_n_clicks=1,
    save_data=True,
    condition='control',
    # the following params are for symmetry with test condition function, 
    # but are not used in control since there's no measurement:
    target_threshold=5.0,
    detection_threshold=3,
    alpha=2.0,
    beta=2.4
):
    """
    Run a multi-subject experiment using pregenerated experimental setup.
    FIXED VERSION: Now uses experimental_setup.json for consistency.
    """

    # Load consistent experimental setup
    setup = load_experimental_setup()
    if setup is None:
        return None

    # Extract parameters from setup
    num_subs = setup['num_subs']
    num_trials = setup['num_trials']
    subject_params = setup['subject_params']
    target_positions = setup['target_positions']
    max_steps = DEFAULT_MAX_STEPS

    # Ensure target_positions is a numpy array for consistency
    if isinstance(target_positions, list):
        target_positions = np.array(target_positions)

    # initial_head_az = 0.0

    print("=== RUNNING CONTROL CONDITION EXPERIMENT ===")
    print(f"Loaded setup: {num_subs} subjects, {num_trials} trials")

    import matplotlib
    matplotlib.use('Agg')

    # Create base save directory if needed
    if save_path and (visualize or save_data):
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    # Pick which trials to visualize
    if num_trials_to_visualize > num_trials:
        trials_to_visualize = list(range(num_trials))
    else:
        trials_to_visualize = sorted(np.random.choice(
            range(num_trials),
            size=num_trials_to_visualize,
            replace=False
        ).tolist())

    # Initialize results dictionary
    results = {
        'experiment_params': {
            'num_subs': num_subs,
            'num_trials': num_trials,
            'max_steps': max_steps,
            'trial_duration_seconds': max_steps*DEFAULT_DT,
            'dt': DEFAULT_DT,
            # 'initial_head_az': initial_head_az,
            'target_range': setup['target_range'],
            'condition': condition,
            'target_threshold': target_threshold,
            'detection_threshold': detection_threshold,
            'random_seed': setup['random_seed']
        },
        'target_positions': target_positions,
        'subject_params': subject_params,
        'subjects': []
    }

    # Run simulations for each subject
    for sub_idx in range(num_subs):
        sub_params = subject_params[sub_idx]

        # make an RNG per subject
        rng = np.random.default_rng(setup['random_seed'] + sub_idx)

        # Create subject directory if visualizing
        sub_dir = None
        if save_path and (visualize or save_data):
            sub_dir = os.path.join(save_path, f'subject_{sub_idx+1}')
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)

        print(f"Running simulations for Subject {sub_idx+1}/{num_subs}...")

        # Initialize subject data
        subject_data = {
            'id': sub_idx + 1,
            'params': sub_params,
            'trials': []
        }

        # Run trials for this subject
        for trial_idx in range(num_trials):
            target_az = target_positions[trial_idx]

            # Create trial directory if visualizing
            trial_dir = None
            if sub_dir and visualize and (trial_idx in trials_to_visualize):
                trial_dir = os.path.join(sub_dir, f'trial_{trial_idx+1}')
                if not os.path.exists(trial_dir):
                    os.makedirs(trial_dir)

            print(
                f"  - Subject {sub_idx+1}, Trial {trial_idx+1}: target {target_az:.1f}°")

            # Run simulation for this trial with the new stopping criteria
            head_positions, theta_target_positions, measurements, kf_estimates, kf_variances, num_steps_used = simulate_control_condition(
                target_az=target_az,
                max_steps=max_steps,
                # initial_head_az=initial_head_az,
                drift_std=sub_params['drift_std'],
                Q=sub_params['Q'],
                target_threshold=target_threshold,
                detection_threshold=detection_threshold,
                rng=rng
            )

            # Store trial data
            trial_data = {
                'id': trial_idx + 1,
                'target_az': target_az,
                'head_positions': head_positions.tolist(),
                'theta_target_positions': theta_target_positions.tolist(),
                'measurements': measurements.tolist(),   # Should be all NaN
                'kf_estimates': kf_estimates.tolist(),
                'kf_variances': kf_variances.tolist(),
                'num_steps_used': num_steps_used,
                'target_found': False  # True if stopped before max steps
            }
            subject_data['trials'].append(trial_data)

            # Visualize if requested & trial is in visualization list
            if trial_dir:
                steps_to_visualize = range(
                    0, num_steps_used, show_every_n_clicks)
                for step_idx in steps_to_visualize:
                    fig = plot_fixed_view_rotating_head(
                        head_positions, target_az, condition, alpha, beta, step_idx=step_idx
                    )
                    fig_path = os.path.join(
                        trial_dir, f'Step_{step_idx+1:02d}.png')
                    fig.savefig(fig_path, dpi=100, bbox_inches='tight')
                    plt.close(fig)

                # Create a summary plot of head positions vs steps
                fig, ax = plt.subplots(figsize=(10, 6))
                steps = np.arange(1, num_steps_used + 1)
                ax.plot(steps, head_positions, 'b-o',
                        linewidth=2, label='Head Position')
                ax.plot(steps, kf_estimates, 'r-s',
                        linewidth=2, label='KF Estimate')
                ax.axhline(y=target_az, color='g', linestyle='--',
                           linewidth=2, label='True Target')

                # Add confidence intervals
                upper_bound = kf_estimates + 2 * np.sqrt(kf_variances)
                lower_bound = kf_estimates - 2 * np.sqrt(kf_variances)
                ax.fill_between(steps, lower_bound, upper_bound,
                                color='r', alpha=0.2, label='95% Confidence')

                ax.set_xlabel('Step Number')
                ax.set_ylabel('Azimuth (degrees)')
                target_status = "found" if num_steps_used < max_steps else "not found"
                ax.set_title(
                    f'Control condition. Subject {sub_idx+1}, Trial {trial_idx+1}: Target at {target_az:.1f}° ({target_status})')
                ax.legend()
                ax.grid(True)

                fig_path = os.path.join(trial_dir, 'summary.png')
                fig.savefig(fig_path, dpi=100, bbox_inches='tight')
                plt.close(fig)

                # Clear memory after trial is complete
                plt.close('all')
                gc.collect()

        # Add subject data to results
        results['subjects'].append(subject_data)

        # Create a summary plot for all trials of this subject
        if visualize and sub_dir:
            fig, ax = plt.subplots(figsize=(12, 8))

            # Collect data on number of steps used for each trial
            steps_used = [t['num_steps_used'] for t in subject_data['trials']]
            max_steps_used = max(steps_used)

            # Plot individual trials aligned to trial end
            for trial_idx, trial_data in enumerate(subject_data['trials']):
                target_az = trial_data['target_az']
                head_positions = np.array(trial_data['head_positions'])
                trial_length = len(head_positions)
                
                # Compute error at each step
                errors = np.abs(head_positions - target_az)
                
                # Create time axis: from [- to 0]
                time_before_end = np.arange(-trial_length, 0)
                
                # Plot with light transparency for individual trials
                ax.plot(time_before_end, errors, 'o-', linewidth=1, alpha=0.3,
                        label=f'Trial {trial_idx+1}' if trial_idx < 5 else None)
                
            # Compute average error across trials aligned to trial end
            mean_errors = np.zeros(max_steps_used)
            std_errors = np.zeros(max_steps_used)

            for relative_idx in range(max_steps_used):
                errors_at_step = []
                
                for t in subject_data['trials']:
                    trial_length = len(t['head_positions'])
                    # Calculate absolute position in this trial
                    absolute_idx = trial_length - 1 - relative_idx
                    
                    if absolute_idx >= 0: # If trial was long enough
                        error = np.abs(t['head_positions'][absolute_idx] - t['target_az'])
                        errors_at_step.append(error)
                
                if errors_at_step:
                    mean_errors[relative_idx] = np.mean(errors_at_step)
                    std_errors[relative_idx] = np.std(errors_at_step)
                else:
                    mean_errors[relative_idx] = np.nan
                    std_errors[relative_idx] = np.nan
            
            # Reverse arrays so index 0 = earliest, index -1 = latest
            mean_errors = mean_errors[::-1]
            std_errors = std_errors[::-1]
            
            # Plot with x-axis showing steps before trial end
            steps_range = np.arange(-max_steps_used + 1, 1)
            valid_indices = ~np.isnan(mean_errors)
            
            #plot mean error
            ax.plot(steps_range[valid_indices], mean_errors[valid_indices],
                    'k-', linewidth=3, label='Mean Error')
            
            # Plot ±1 SD
            ax.fill_between(
                steps_range[valid_indices],
                mean_errors[valid_indices] - std_errors[valid_indices],
                mean_errors[valid_indices] + std_errors[valid_indices],
                color='k', alpha=0.2, label='±1 SD'
            )

            ax.set_xlabel('Step Number before trial end')
            ax.set_ylabel('Absolute Error (°)')
            ax.set_title(
                f'Control condition. Subject {sub_idx+1}: Error across {num_trials} Trials')

            # Add parameters to figure
            param_text = '\n'.join([
                f"Drift: {sub_params['drift_std']:.2f}",
                f"Q: {sub_params['Q']:.2f}",
                f"Target threshold: {target_threshold:.1f}°",
                f"Avg steps: {np.mean(steps_used):.1f}"
            ])
            props = dict(boxstyle='round', facecolor='white', alpha=0.7)
            ax.text(0.05, 0.95, param_text, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)
            
            # plot legend
            if num_trials <= 10:
                ax.legend()
            else:
                # Only show first 5 trials plus mean in legend
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles[:5] + handles[-2:], labels[:5] + labels[-2:])

            ax.grid(True)
            ax.set_ylim(bottom=0)  # Error can't be negative

            fig_path = os.path.join(sub_dir, 'all_trials_summary.png')
            fig.savefig(fig_path, dpi=100, bbox_inches='tight')
            plt.close(fig)

            # Clear memory after subject is complete
            plt.close('all')
            gc.collect()

    # Create overall experiment summary plot
    if visualize and save_path:
        fig, ax = plt.subplots(figsize=(12, 8))

        # Find the maximum number of steps used across all subjects/trials
        max_steps_overall = max([
            max([len(t['head_positions']) for t in subject_data['trials']])
            for subject_data in results['subjects']
        ])

        # Calculate mean error for each subject
        for sub_idx, subject_data in enumerate(results['subjects']):
            # Calculate mean error at each step position
            mean_errors = np.zeros(max_steps_overall)
            sem_errors = np.zeros(max_steps_overall)

            for relative_idx in range(max_steps_overall):
                errors_at_step = []
                for t in subject_data['trials']:
                    trial_length = len(t['head_positions'])
                    absolute_idx = trial_length - 1 - relative_idx

                    if absolute_idx >= 0:
                        error = np.abs(t['head_positions']
                                       [absolute_idx] - t['target_az'])
                        errors_at_step.append(error)

                if errors_at_step:  # If we have data for this step
                    mean_errors[relative_idx] = np.mean(errors_at_step)
                    sem_errors[relative_idx] = sem(errors_at_step)
                else:
                    # No data for this step (all trials ended before this point)
                    mean_errors[relative_idx] = np.nan
                    sem_errors[relative_idx] = np.nan

            # Reverse arrays
            mean_errors = mean_errors[::-1]
            sem_errors = sem_errors[::-1]

            # Plot using masked array to handle NaN values
            valid_indices = ~np.isnan(mean_errors)
            steps_range = np.arange(-max_steps_overall + 1, 1)

            ax.plot(steps_range[valid_indices], mean_errors[valid_indices],
                    'o-', linewidth=2, label=f'Subject {sub_idx+1}')

            # Add error bars for standard error (less cluttered than fill_between)
            if num_subs <= 5:  # Only add error bars if we have few subjects
                ax.errorbar(
                    steps_range[valid_indices],
                    mean_errors[valid_indices],
                    yerr=sem_errors[valid_indices],
                    fmt='none', alpha=0.3
                )

        # Calculate grand average across all subjects and trials
        grand_mean = np.zeros(max_steps_overall)
        grand_sem = np.zeros(max_steps_overall)

        for relative_idx in range(max_steps_overall):
            # Collect all errors for this step index across all subjects and trials
            all_errors = []
            for subject_data in results['subjects']:
                for trial_data in subject_data['trials']:
                    trial_length = len(trial_data['head_positions'])
                    # Calculate absolute position in this trial
                    absolute_idx = trial_length - 1 - relative_idx

                    if absolute_idx >= 0:  # If this trial was long enough to have this relative position
                        error = abs(
                            trial_data['head_positions'][absolute_idx] - trial_data['target_az'])
                        all_errors.append(error)

            if all_errors:  # If we have data for this step
                grand_mean[relative_idx] = np.mean(all_errors)
                grand_sem[relative_idx] = sem(all_errors)
            else:
                grand_mean[relative_idx] = np.nan
                grand_sem[relative_idx] = np.nan

        # Reverse the arrays so index 0 is the earliest point
        grand_mean = grand_mean[::-1]
        grand_sem = grand_sem[::-1]

        # Plot grand average using masked array
        valid_indices = ~np.isnan(grand_mean)
        # X-axis: negative numbers counting down to 0
        # e.g., -299, -298, ..., -1, 0
        steps_range = np.arange(-max_steps_overall + 1, 1)

        ax.plot(steps_range[valid_indices], grand_mean[valid_indices],
                'k-', linewidth=3, label='Grand Mean')

        ax.fill_between(
            steps_range[valid_indices],
            grand_mean[valid_indices] - grand_sem[valid_indices],
            grand_mean[valid_indices] + grand_sem[valid_indices],
            color='k', alpha=0.2
        )

        ax.set_xlabel('Step Number before trial end')
        ax.set_ylabel('Absolute Error (degrees)')
        ax.set_title(
            f'Control condition w/o audio. Error across {num_subs} Subjects ({num_trials} Trials Each)')

        if num_subs <= 10:
            ax.legend()
        else:
            # Only show a subset of subjects in the legend
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles[:5] + handles[-1:], labels[:5] + labels[-1:])

        ax.grid(True)
        ax.set_ylim(bottom=0)  # Error can't be negative

        fig_path = os.path.join(save_path, 'experiment_summary.png')
        fig.savefig(fig_path, dpi=100, bbox_inches='tight')
        plt.close(fig)

        # Add histogram of number of steps used
        fig, ax = plt.subplots(figsize=(10, 6))
        all_steps_used = []
        for subject_data in results['subjects']:
            all_steps_used.extend([t['num_steps_used']
                                  for t in subject_data['trials']])

        ax.hist(all_steps_used, bins=range(1, max_steps+2), alpha=0.7)
        ax.axvline(x=np.mean(all_steps_used), color='r', linestyle='--',
                   label=f'Mean: {np.mean(all_steps_used):.1f} steps')

        ax.set_xlabel('Number of Steps Used')
        ax.set_ylabel('Frequency')
        ax.set_title(
            'Distribution of Steps Used Across All Subjects and Trials')
        ax.legend()
        ax.grid(True)

        fig_path = os.path.join(save_path, 'steps_histogram.png')
        fig.savefig(fig_path, dpi=100, bbox_inches='tight')
        plt.close(fig)

    # Save results if requested
    if save_data and save_path:
        results_path = os.path.join(
            save_path, 'control_experiment_results.json')
        with open(results_path, 'w') as f:
            json.dump(convert_numpy_types(results), f, indent=2)

        # Final cleanup before returning
        plt.close('all')
        gc.collect()

    # Add timing analysis
    analyze_timing_results(results)

    return results