# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 12:02:40 2026
@author: sofia krasovskaya

===============================================================================
Echolocation Simulation with Kalman Filter: core components of the model
===============================================================================

-------------------------------------------------------------------------------
Contents:
-------------------------------------------------------------------------------

Constants:
    DEFAULT_DT, DEFAULT_TRIAL_DURATION, DEFAULT_MAX_STEPS

Classes:
    SweepController     : Head movement controller
    EchoAudioLibrary    : Manages angle-indexed .wav file library
    ITDILDProcessor     : Extracts ITD from stereo audio

Shared utility functions:
    load_experimental_setup()
    reflectivity()
    compute_directivity_pattern()
    analyze_timing_results()
    convert_numpy_types()

Visualization:
    plot_fixed_view_rotating_head()
===============================================================================
"""

#Try on big and small target first, then see if you can add the control or just leave it as as separate script

import numpy as np
import matplotlib.pyplot as plt
# import os
import scipy.io.wavfile as wav
# from scipy.stats import sem
import json
# import gc
from pathlib import Path
import re



# ----------------------------------------------------------------------------
# CONSTANTS
# ----------------------------------------------------------------------------

DEFAULT_DT = 0.1                                              # sec per step (10 Hz)
DEFAULT_TRIAL_DURATION = 30.0                                 # sec
DEFAULT_MAX_STEPS = int(DEFAULT_TRIAL_DURATION / DEFAULT_DT)  # 300 steps



# ----------------------------------------------------------------------------
# LOAD EXPERIMENTAL SETUP
# ----------------------------------------------------------------------------

def load_experimental_setup(path: str = 'experimental_setup.json') -> dict | None:
    
    """Load the pregenerated experimental setup fron JSON
    
    The JSON file was pregenerated using 1-generate_subs.py.
    It contains important data likesubject parameters (Q, perceptual noise, etc.), 
    target positions, & random seed to ensure that the parameters and target 
    locations are shared across all model conditions.
        
    Parameters
    ----------
    path: str
        Path to the JSON file (default: 'experimental_setup.json').

    Returns
    -------
    dict or None
        Setup dictionary, or None if file is not found.
    """
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"ERROR: {path} not found!")
        print("Run '1-generate_subjects.py' first")
        return None


# ---------------------------------------------------------------------------
# TARGET REFLECTIVITY FACTOR
# ---------------------------------------------------------------------------

def reflectivity(target_size: str) -> float:
    """
    Return the reflectivity scaling factor for a given target size.

    Reflectivity scales echo quality, which in turn scales the measurement
    noise covariance R_k in the KF. Larger targets produce
    stronger echoes (higher quality → lower R_k → more trust in measurement).
    
    The sizes replicate the target sizes used in Teng et al. (2026)

    Parameters
    ----------
    target_size: str
        'big': 29 cm × 36 cm board  (reflect_factor = 1.0)
        'small': 2.5 cm × 17 cm board (reflect_factor = 0.04)

    Returns
    -------
    float
        Reflectivity factor between [0, 1].
    """
    if target_size == 'big':
        return 1.0   # 29 cm × 36 cm
    elif target_size == 'small':
        return 0.04  # 2.5 cm × 17 cm
    else:
        return 0.0   # can potentially another condition




# ----------------------------------------------------------------------------
# SWEEP CONTROLLER
# ----------------------------------------------------------------------------
class SweepController:
    """
    Controls head azimuth movement.
    
    Implements a proportional gain controller with velocity-based damping. The
    head sweeps toward a desired position (theta_target), which equals the
    KF's current target estimate (x). When echo guidance is active, gain 
    increases and damping adapts to the KF uncertainty, producing progressively
    tighter tracking as the target estimate converges.
    
    Params:
    -------
    max_amp : float
        Maximum head azimuth in degrees (head stays within +-max_amp).
    min_vel : float
        Minimum speed 
    max_vel : float
        Velocity clipping limit (deg per timestep).
    noise_std : float
        Standard deviation of motor noise (deg per timestep).
    rng : numpy.random.Generator or None
        Random number generator. Created internally if None.        
    
    """
    def __init__(self, max_amp: float, min_vel:float =0.0, max_vel:float =50.0, noise_std:float =0.0, rng=None):
        #Params
        self.max_amp = max_amp
        self.min_vel = min_vel
        self.max_vel = max_vel
        self.noise_std = noise_std
        self.rng = rng or np.random.default_rng()
        
        #State
        self.position = 0.0
        self.direction = self.rng.choice([-1, 1])  # start moving in a random direction
        self.velocity = self.direction * self.rng.uniform(0, max_vel) # Randomize initial speed
        
        # Internal tracking
        self.target_pos_history=[] #to store theta_target history for later plotting
        self.time_step = 0
        self.echo_guided = False  # Flag for when we have echo feedback
     
        # KF interface
        # updated by the simulation loop after each measurement 
        self.kf_target_estimate = 0.0 # theta_target = x
        self.has_target_estimate = False
        self.current_uncertainty = 500.0 # initialised to P_0
        
# ------------------------------------------------------------------
# PUBLIC
# ------------------------------------------------------------------

    def set_kf_target_estimate(self, target_angle:float) -> None:
        """Update desired head position from KF estimate"""
        
        self.kf_target_estimate = target_angle
        self.has_target_estimate = True

    def step(self) -> float:
        """
      Advance the controller by one timestep and return the new head position.

      Returns
      -------
      float
          Head azimuth in degrees, clipped to +-max_amp.
      """
        self.time_step += 1
        
        # calculate assumed target position at each timestep
        target_pos = self._calculate_target_pos()
        self.target_pos_history.append(target_pos) #append to plot theta_target later

        # Move toward assumed target position
        self._move_toward_target(target_pos)

        # Handle boundary conditions
        self._handle_boundaries()

        # Add movement noise and return final position
        final_position = self.position + \
            self.rng.normal(0, self.noise_std)
            
        return np.clip(final_position, -self.max_amp, self.max_amp)

    def set_current_uncertainty(self, uncertainty):
        """
        Update the current uncertainty P from KF
        """
        self.current_uncertainty = uncertainty


    def set_echo_guidance(self, enabled:bool = True) -> None:
        """
        Set whether echo guidance is active.
        True when a valid measurement was received this timestep.
        """
        self.echo_guided = enabled

# ------------------------------------------------------------------
# PRIVATE
# ------------------------------------------------------------------
    def _calculate_target_pos(self) -> float:
        """       
        Calculate theta_target: desired head position based on KF estimate.
           
        Represents motor planning stage: converts belief (KF estimate, x)
        into desired head orientation. The proportional controller then
        generates velocities to achieve this desired position.
        

        Returns: Desired head azimuth in degrees
        
        __________________
        Basically, theta_target = x. In the test conditions x converges 
        toward the true target, pulling the head there over successive clicks.
        
        """
        return self.kf_target_estimate


    def _move_toward_target(self, target_pos:float) -> None:
        """move towards calculated target position with more realistic dynamics

        __________________________________________________________________
        Implements proportional control with velocity-based damping:
        - Proportional gain (Kp): converts position error to velocity change
        - Damping: reduces velocity each step to prevent overshooting
        
        Parameters adapt based on uncertainty P (confidence in target location):
        Damping increases as uncertainty decreases: (more certain -> more dampened)
        - High uncertainty -> low damping -> smooth, exploratory sweeps
        - Low uncertainty -> high damping -> precise, more responsive tracking
        """

        position_error = target_pos - self.position

        # Continuous adaptation of movement parameters
        if self.echo_guided:
            gain = 0.5  # More responsive when guidance is present
            # Damping increases as uncertainty decreases
            damping_coefficient = 0.05 + 0.1 * (1.0 - self.current_uncertainty/100.0)
        else:
            gain = 0.05  # Small gain for exploration
            damping_coefficient = 0.05  # Low damping for smooth & broader exploration
            
        #apply gain and damping
        self.velocity += position_error * gain
        self.velocity *= (1 - damping_coefficient)

        # limit velocity to max possible (just in case if out of bounds)
        self.velocity = np.clip(self.velocity, -self.max_vel, self.max_vel)

        # updt position
        self.position += self.velocity

    def _handle_boundaries(self) -> None:
        """
        Clip position to +-max_amp and apply a dampened velocity reversal.

        The 0.7 dampening on reversal prevents perfectly elastic bouncing
        and creates more natural variability in sweep width.
        """

        if abs(self.position) >= self.max_amp:
            # clip if position is about to move out of bounds
            self.position = np.clip(self.position, -self.max_amp, self.max_amp)

            # reverse direction by flipping the velocity sign
            # also slow down a bit for less robotic mvmnts
            self.velocity *= -0.7  # dampened reversal
            
            # keep track of direction (flag for debugging):
            self.direction *= -1

            # track when reversal is hit for tracking/debugging
            self.last_reversal_time = self.time_step



# ------------------------------------------------------------------
# AUDIO PROCESSING
# ------------------------------------------------------------------
class EchoAudioLibrary:
    """Manages audio files (.wav) organized by angle - 
    loads appropriate file based on head position at time [k]
    
    On construction the library scans the directory and maps filenames to
    integer azimuth angles. At each click the model requests the file
    with angle closest to the current head-target relative angle.

    """

    def __init__(self, audio_directory:str):
        self.audio_directory = Path(audio_directory)
        self.audio_cache:dict = {}
        self.angle_to_file:dict = {}
        self.available_angles:list = []
        self._scan_audio_files()

    def _scan_audio_files(self) -> None:
        """Find all .wav files and extract angles from filenames"""

        audio_files = list(self.audio_directory.glob("*.wav"))

        for file_path in audio_files:
            filename = file_path.name

            # Extract angle from filename
            # Patterns (e.g.): az-3_tdist100_Idist100_fs96000.wav", "stereo_click_0.wav",
            # "0deg.wav", "5deg.wav", "-10deg.wav", "angle_15.wav", "0.wav"
            patterns = [
                r'az(\+?-?\d+)_',
                r'stereo_click_(\+?-?\d+)',
                r'angle_(\+?-?\d+)',
                r'^(\+?-?\d+)\.wav$',
                r'^(\+?-?\d+)$'
            ]

            angle = None  # init angle (no angle yet)
            for pattern in patterns:  # loop thru each pattern in list of patterns above
                # clean filename, rm .wav xtnsn, find pattern in cleaned filename
                match = re.search(pattern, filename.replace('.wav', ''))
                if match:
                    try:
                        # xtract captured text from parentheses
                        angle = int(match.group(1))
                        break  # exit loop, valid match found
                    except ValueError:
                        continue  # try next pattern

            if angle is not None:
                # dictionary {angle: file_path}
                self.angle_to_file[angle] = file_path
                # list [angle1, angle2, ...]
                self.available_angles.append(angle)

        self.available_angles = sorted(self.available_angles)


    def get_audio_for_head_angle(self, head_angle:float) -> tuple:
        """
        Get audio data for the head angle (loads closest available file)
        
        Parameters
       ----------
       head_angle : float
           Desired angle in degrees (relative angle: target_az - head_az).

       Returns
       -------
       (audio_info, used_angle) : (dict or None, float)
           audio_info contains 'fs', 'audio_data', 'angle', 'file_path'.
           Returns (None, nan) if library is empty.
        """
        if not self.available_angles:
            return None, float('nan')

        # Find closest available angle
        closest_angle = min(self.available_angles,
                            key=lambda x: abs(x - head_angle))

        # Load if not cached
        if closest_angle not in self.audio_cache:
            file_path = self.angle_to_file[closest_angle]
            fs, audio_data = self._load_wav_file(file_path)
            if fs is not None:
                self.audio_cache[closest_angle] = {
                    'fs': fs,
                    'audio_data': audio_data,
                    'angle': closest_angle,
                    'file_path': str(file_path)
                }

        return self.audio_cache.get(closest_angle), closest_angle

    def _load_wav_file(self, file_path: Path) -> tuple:
        """
        Load stereo .wav file & convert to float32 [-1, 1]
        """
        try:
            fs, audio_data = wav.read(file_path)

            # Convert integer audio to float (-1.0 to 1.0 range) to avoid overflow warnings
            # for later ILD calc. It uses RMS
            # RMS, square root, correlation, etc., work better with float values between -1.0 and +1.0.
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.int32:
                audio_data = audio_data.astype(np.float32) / 2147483648.0
            elif audio_data.dtype == np.uint8:
                audio_data = (audio_data.astype(np.float32) - 128) / 128.0

            # Ensure stereo
            if len(audio_data.shape) == 1:
                # duplicates the mono signal to both channels to create stereo
                audio_data = np.column_stack([audio_data, audio_data])
            elif audio_data.shape[1] != 2:
                # makes sure audio is not multichannel
                raise ValueError(f"File {file_path} must be mono or stereo")

            return fs, audio_data
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None, None


class ITDILDProcessor:
    """
    Extract ITD/ILD from stereo audio - based on S. Teng's MATLAB approach
    """

    def __init__(self):
        self.speed_of_sound = 343  # m/s
        self.head_radius = 0.09    # metres
        self.ear_distance = 2 * self.head_radius

    def extract_itd_ild_from_audio(self, audio_data:np.ndarray, fs:int) -> dict | None:
        """
        Extract ITD, ILD angle estimate & quality from stereo audio data.
        
        Parameters
        ----------
        audio_data : np.ndarray, shape (N, 2)
            Stereo audio, float32, normalised to [-1, 1].
        fs : int
            Sampling rate in Hz.

        Returns
        -------
        dict or None
            Keys: 'ITD_seconds', 'ILD_db', 'angle_degrees', 'quality',
            'segment_length'. Returns None if the file is too short or
            processing fails.        
        """
        left_channel = audio_data[:, 0]
        right_channel = audio_data[:, 1]

        # Find the main peak (echo) in both channels
        # we will use left channel as reference
        left_peak_idx = np.argmax(np.abs(left_channel))

        # Adjust extraction window for short audio files
        audio_length = len(left_channel)

        if audio_length < 100:  # File too short, less than 100 samples
            return None

        # Use left peak as reference and extract segments around it
        # Use smaller, adaptive windows for short files
        if audio_length < 500:  # Short files (ours are at 279 samples)
            # Use the entire file or most of it
            # Use most of the file, leave 20 samples margin
            window_size = min(100, audio_length - 20)
            half_window = window_size // 2
            start_idx = max(0, left_peak_idx - half_window)  # before peak
            end_idx = min(audio_length, start_idx + window_size)  # after peak
            # Adjust start if end was clipped
            if end_idx == audio_length:
                start_idx = max(0, end_idx - window_size)
        else:  # Longer files
            # use 150 samples before peak
            start_idx = max(0, left_peak_idx - 150)
            # use 150 samples after peak
            end_idx = min(audio_length, left_peak_idx + 150)
          
        if end_idx - start_idx < 20:  # minimum viable samples segment
            print("Segment too short, returning None")
            return None  # Segment too short

        left_segment = left_channel[start_idx:end_idx]
        right_segment = right_channel[start_idx:end_idx]
        
        # Cross-correlation for ITD
        cross_corr = np.correlate(left_segment, right_segment, mode='full')
        lags = np.arange(-len(right_segment) + 1, len(left_segment))

        # Find lag with maximum correlation (Find the time shift that makes the
        # left and right ear signals most similar =  ITD)
        max_corr_idx = np.argmax(np.abs(cross_corr))
        time_lag_samples = lags[max_corr_idx]

        # Convert to time by dividing lag by sampling rate
        ITD = time_lag_samples / fs

        # Convert ITD to angle theta using spherical head model
        # Clamp to physically possible range
        max_itd = self.ear_distance / self.speed_of_sound
        ITD_clamped = np.clip(ITD, -max_itd, max_itd)

        # Given this ITD, what angle does it correspond to?
        #   derivation:
        #     path_difference = ear_distance × sin(θ)
        #     ITD = path_difference / speed_of_sound
        #     ITD = (ear_distance × sin(θ)) / speed_of_sound

        #   Solving for θ:
        #     ITD × speed_of_sound = ear_distance × sin(θ)
        #     sin(θ) = (ITD × speed_of_sound) / ear_distance
        #     θ = arcsin((ITD × speed_of_sound) / ear_distance)
        theta = np.degrees(
            np.arcsin((ITD_clamped * self.speed_of_sound) / self.ear_distance))

        # ILD calculation (level difference) in dB
        # RMS = "Perceived Loudness", effective power of a signal
        # RMS handles positive/negative oscillations & measures actual signal strength
        rms_left = np.sqrt(np.mean(left_segment**2))
        rms_right = np.sqrt(np.mean(right_segment**2))

        if rms_right > 0 and rms_left > 0:
            # dB formula based on Amplitude: dB = 20 × log10(A1/A2)
            ILD = 20 * np.log10(rms_left / rms_right)
        else:
            ILD = 0.0

        # Cross-correlation quality metric. Adaptive measurement noise in KF measurement
        # Measures how well the left and right signal correlate. Used to 
        # determine reliability of the ITD measurement (i.e. tell the KF how much to trust the ITD measurement)
        quality = np.max(np.abs(cross_corr)) / np.sqrt(
            np.sum(left_segment**2) * np.sum(right_segment**2)
        )

        return {
            'ITD_seconds': ITD,
            'ILD_db': ILD,
            'angle_degrees': theta,
            'quality': quality,
            'segment_length': len(left_segment)
        }


# ------------------------------------------------------------------
# UTILS & VISUALIZATION FUNCTIONS
# ------------------------------------------------------------------

# Simple Timing Analysis Function
def analyze_timing_results(results):
    """
    To analyze timing aspects of the results
    """

    max_steps = results['experiment_params']['max_steps']
    trial_duration = max_steps * DEFAULT_DT

    print("\n=== TIMING ANALYSIS ===")
    print(f"Trial duration: {trial_duration} seconds ({max_steps} steps)")
    print(f"Simulation rate: {1/DEFAULT_DT} Hz")

    all_response_times = []
    detections = 0
    total_trials = 0

    for subject in results['subjects']:
        for trial in subject['trials']:
            total_trials += 1
            steps_used = trial['num_steps_used']
            time_used = steps_used * DEFAULT_DT

            if trial.get('target_found', False):
                all_response_times.append(time_used)
                detections += 1

    detection_rate = detections / total_trials if total_trials > 0 else 0

    print(f"Overall detection rate: {detection_rate:.1%}")
    if all_response_times:
        mean_response = np.mean(all_response_times)
        std_response = np.std(all_response_times)
        print(
            f"Mean response time: {mean_response:.1f} ± {std_response:.1f} seconds")
        print(
            f"Response time range: {min(all_response_times):.1f} - {max(all_response_times):.1f} seconds")

def convert_numpy_types(obj):
    """
    Recursively convert numpy types to native Python types for JSON serialisation
    """
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def compute_directivity_pattern(theta, alpha, beta, apply_db_conversion=False):
    """
    Compute the power directivity pattern using a modified cardioid equation.
    Based on Thaler et al. (2017). Mouth-clicks used by blind expert human 
    echolocators – signal description and model based signal synthesis. 
    PLoS Computational Biology, 13(8), e1005670. 
    https://doi.org/10.1371/journal.pcbi.1005670

    
    Used for visualization in plot_fixed_view_rotating_head to draw the 
    cardioid-shaped directivity pattern on the polar plots.

    Params:
    -----------
    theta : float
        Angle in degrees between head direction and target
    alpha : float
        Horizontal axis scaling parameter
    beta : float
        Vertical axis scaling parameter
    apply_db_conversion : bool, optional
        Whether to apply the decibel conversion 

    Returns:
    --------
    float
        Directivity factor
    """
    # Convert theta to radians
    theta_rad = np.deg2rad(theta)

    # Following the paper's equation R(θ) = -(1 + cos(θ)) / √(α²cos²(θ) + β²sin²(θ))
    # Add pi to theta, which rotates the pattern by 180 degrees
    theta_shifted = theta_rad + np.pi

    # Calculate the modified cardioid
    numerator = -(1 + np.cos(theta_shifted))

    # Denominator provides elliptical modification
    denominator = np.sqrt(
        (alpha * np.cos(theta_shifted))**2 +
        (beta * np.sin(theta_shifted))**2
    )

    # Compute directivity ratio
    directivity_ratio = numerator / denominator

    # Apply decibel conversion if requested
    if apply_db_conversion:
        # Convert to power ratio (10^(dB/10))
        directivity_power = 10**(directivity_ratio / 10)
        return directivity_power
    else:
        # Return the absolute value for power calculations
        return np.abs(directivity_ratio)


def plot_fixed_view_rotating_head(
        head_positions,
        target_az,
        condition,
        alpha=2.0,
        beta=2.4,
        step_idx=None
):
    """
    Create a polar plot where:
    - The target is fixed at its absolute position
    - The head is fixed at the center
    - Only the head direction (arrow) and cardioid rotate
    - If condition='control', we won't plot the cardioid pattern

    Params:
    -----------
    head_positions : array
        Array of head positions at each click (to determine head direction)
    target_az : float
        True target location in degrees
    condition : str
        'test' or 'control' to determine visualization style
    alpha, beta : float
        Directivity pattern parameters
    step_idx : int or None
        Timestep to visualise. Defaults to final step.

    Returns:
    --------
    matplotlib.figure.Figure
        Figure containing the polar plot
    """
    # Choose head position for the current click
    if step_idx is not None and step_idx < len(head_positions):
        head_az = head_positions[step_idx]
        step_num = step_idx + 1  # For display
    else:
        head_az = head_positions[-1]
        step_num = len(head_positions)  # Final click

    # In this view:
    # - We use a fixed global reference frame (0 deg is up)
    # - The target is at its absolute position (target_az)
    # - The head's direction is at its absolute position (head_az)

    # Convert to radians for plotting
    target_rad = np.deg2rad(target_az)
    head_direction_rad = np.deg2rad(head_az)

    # Create the figure
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, polar=True)

    # Place the target at its fixed position
    target_distance = 0.95  # Place near the edge of the unit circle
    ax.plot([target_rad], [target_distance],
            'ro', markersize=12, label='Target')

    # Add a line from head to target
    ax.plot([0, target_rad], [0, target_distance], 'r--',
            linewidth=1.5, alpha=0.9, label='Head-Target Line')

    # Mark head position at the center
    ax.plot([0], [0], 'bo', markersize=10, label='Head')

    # Add an arrow showing the head's current direction
    arrow_length = 0.3
    ax.annotate('',
                xy=(head_direction_rad, arrow_length),
                xytext=(0, 0),
                arrowprops=dict(facecolor='green', shrink=0, width=3, headwidth=9))

    # If it's the test condition, draw the directivity pattern
    if condition == 'test':
        # Setup angle arrays for the directivity patterns
        # 361 points to ensure we close the circle
        theta_deg = np.linspace(-180, 180, 361)
        theta_rad = np.deg2rad(theta_deg)

        # Plot the cardioid directivity pattern
        # Calculate the directivity for angles relative to the head's direction
        outgoing_directivity = np.array([compute_directivity_pattern(t - head_az + 180, alpha, beta, apply_db_conversion=False)
                                        for t in theta_deg])

        # Normalize and scale
        outgoing_norm = outgoing_directivity / outgoing_directivity.max()
        scale_factor = 0.95  # Scale to reasonable size

        # Plot the outgoing directivity pattern
        ax.plot(theta_rad, outgoing_norm * scale_factor, 'b-', linewidth=2)
        ax.fill(theta_rad, outgoing_norm * scale_factor,
                'b', alpha=0.2, label='Outgoing Click')

    # Set title and formatting
    ax.set_title(
        f"{condition.capitalize()} condition. Step {step_num})", fontsize=12)
    ax.set_theta_zero_location("N")  # 0 degrees at the top
    ax.set_theta_direction(-1)       # clockwise
    ax.set_rlim(0, 1.0)              # Set radius limit
    ax.set_rticks([0.2, 0.4, 0.6, 0.8, 1.0])

    # Customize theta ticks to use ±180° notation
    ax.set_xticks(np.deg2rad([-180, -135, -90, -45, 0, 45, 90, 135, 180]))
    ax.set_xticklabels(['±180°', '-135°', '-90°', '-45°',
                       '0°', '45°', '90°', '135°', '±180°'])

    # Force the plot to show the full 360° circle
    ax.set_thetamin(-180)
    ax.set_thetamax(180)
    
    ax.grid(True)

    # Calculate relative angle between head and target
    rel_angle = target_az - head_az
    # Ensure angle is in the range -180 to 180
    while rel_angle > 180:
        rel_angle -= 360
    while rel_angle < -180:
        rel_angle += 360

    # Add information about head and target positions
    plt.figtext(0.5, 0.02,
                f"Step {step_num}: Head = {head_az:.1f}°, "
                f"Target = {target_az:.1f}°, "
                f"Relative angle = {rel_angle:.1f}°",
                ha='center', fontsize=11, bbox=dict(boxstyle='round', 
                                                    facecolor='white', 
                                                    alpha=0.7))

    # plt.close(fig)  # Close the figure to free memory. Comment out to see visualisation in test run
    return fig




##############################################################################

    #########   #########    #########    #########
        #       #            #                #
        #       ######       #########        #
        #       #                    #        #
        #       #########    #########        #

##############################################################################
# ---------------------------------------------------------------------------

def test_audio_library_setup(audio_directory):
    """
    Test that audio files are found and can be loaded
    """
    print(f"Testing audio library setup: {audio_directory}")

    library = EchoAudioLibrary(audio_directory)

    if len(library.available_angles) == 0:
        print("ERROR: No audio files found!")
        return False

    print(f"Found {len(library.available_angles)} audio files")


    # Test loading a few files
    test_angles = [0, 15, -30] if any(a in library.available_angles for a in [
                                      0, 15, -30]) else library.available_angles[:3]

    for angle in test_angles:
        audio_info, used_angle = library.get_audio_for_head_angle(angle)
        if audio_info:
            print(
                f"  Test angle {angle}° -> loaded {used_angle}° ({audio_info['audio_data'].shape})")
        else:
            print(f"  Test angle {angle}° -> FAILED!")

    return True


def test_single_trial_audio(audio_directory, target_az=30.0):
    """
    Test a single trial with audio input
    """
    print(f"Testing single trial with audio directory: {audio_directory}")
    
    from model_engine import simulate_test_condition_with_audio
    
    try:
        results = simulate_test_condition_with_audio(
            audio_directory=audio_directory,
            target_az=target_az,
            max_steps=100,
            angle_tolerance=0.5
        )

        head_pos, theta_target_positions, measurements, kf_est, kf_var, n_steps,\
            click_events, metadata, target_found = results

        print(f"Trial completed in {n_steps} steps")
        print(f"Final head position: {head_pos[-1]:.1f}°")
        print(f"Target was at: {target_az:.1f}°")
        print(
            f"Made {len(click_events)} clicks, {len(metadata)} successful measurements")

        # Quick plot
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        steps = np.arange(1, n_steps + 1)

        axes[0].plot(steps, head_pos, 'b-', label='Head Position')
        axes[0].plot(steps, kf_est, 'r-', label='KF Estimate')
        axes[0].axhline(target_az, color='g', linestyle='--', label='Target')

        # Mark measurements
        valid_meas = ~np.isnan(measurements)
        if np.any(valid_meas):
            axes[0].plot(steps[valid_meas], measurements[valid_meas],
                         'go', markersize=8, label='Measurements')

        axes[0].set_ylabel('Azimuth (°)')
        axes[0].set_title('Test Trial with Audio Feedback')
        axes[0].legend()
        axes[0].grid(True)

        error = np.abs(head_pos - target_az)
        axes[1].plot(steps, error, 'k-', linewidth=2)
        axes[1].set_xlabel('Step')
        axes[1].set_ylabel('Error (°)')
        axes[1].set_title('Localization Error')
        axes[1].grid(True)

        plt.tight_layout()
        plt.show()

        return True

    except Exception as e:
        print(f"Error in test trial: {e}")
        return False

def validate_audio_files(audio_directory):
    """
    Test a few audio files to see if ITD matches expected angles
    """
    library = EchoAudioLibrary(audio_directory)
    processor = ITDILDProcessor()

    test_angles = [0, 30, -30, 45, -45]

    print("-----AUDIO FILE VALIDATION:-----")
    print("expected vs calculated angles:")

    for expected_angle in test_angles:
        if expected_angle in library.available_angles:
            audio_info, used_angle = library.get_audio_for_head_angle(
                expected_angle)
            if audio_info:
                result = processor.extract_itd_ild_from_audio(
                    audio_info['audio_data'], audio_info['fs'])
                if result:
                    calculated_angle = result['angle_degrees']
                    print(
                        f"Expected: {expected_angle:.1f}°, File: {used_angle:.1f}°, \
                            Calculated: {calculated_angle:.1f}°, \
                                Diff: {abs(expected_angle - calculated_angle):.1f}°")
                else:
                    print(
                        f"Expected: {expected_angle:.1f}°, \
                            File: {used_angle:3.0f}°, PROCESSING FAILED")

