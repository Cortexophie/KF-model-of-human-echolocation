# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 11:36:52 2026
@author: sofia krasovskaya

KF.py
===========================

This file contains Kalman filter functions for the echolocolation model.


Two functions are provided so the caller can choose which step to apply
at each timestep:
    - kf_predict : always called (propagates state forward)
    - kf_update  : called only when a click measurement is available

"""
import numpy as np

def kf_predict(x: float, P: float, Q: float) -> tuple[float, float]:
    """
   Kalman filter predict step.

   Propagates the state estimate and covariance forward by one timestep.
   Because the target is static, x_pred = x (no state transition drift).
   Process noise Q accounts for residual model uncertainty.

    Parameters
    ----------
    x : float
        current target location estimate (deg)
    P : float
        current estimate confidence covariance (deg^2)
    Q : float
        Process noise covariance (deg^2)

    Returns
    -------
    tuple[x_pred, P_pred]:
        x_pred: float 
            predited target location estimate (deg)
        P_pred: float
            predicted covariance (P+Q)        

    """
    x_pred = x          # static target: no dynamics
    P_pred = P + Q      # uncertainty grows slightly each step
    
    return x_pred, P_pred



def kf_update (x_pred: float, P_pred: float, z: float, R_k:float) -> tuple[float, float]:
    """
    Kalman filter measurement update step.

    Incorporates a new angle measurement z into the state estimate.
    The measurement model is a direct observation: H = [1].

    Measurement noise R_k is adaptive — it is scaled by echo quality determined 
    by target reflectivity so that weaker echoes have less influence on the
    estimate (higher R_k → lower Kalman gain → less correction).
    
    Parameters
    ----------
    x_pred : float
        predicted target location (deg), from kalman_predict()
    P_pred : float
        predicted confidence covariance (deg^2), from kalman_predict().
    z : float
        measured target angle (absolute coordinates, deg^2)
    R_k : float
        measurement noise covariance for this measurement (deg^2). 
        Computed as σ_measurement² / max(enhanced_quality, 0.05),
        where σ_measurement = σ_base/sqrt(reflectivity)
    
    Returns
    -------
    tuple[x_post, P_post]
        x_post: float
            Updated (posterior) target location estimate (deg)
        P_post: float
            updated (posterior) covariance (deg^2)

    """
    
    # since H = np.array([1.0]), we drop it for simlification's sake.
    
    
    innovation = z - x_pred     # z - H*x_pred in general KF equation
    S = P_pred  + R_k           #innovation covariance
    K = P_pred / S              #Kalman gain
    
    x_post = x_pred + (K * innovation) 
    P_post = (1 - K ) * P_pred
    
    return x_post, P_post