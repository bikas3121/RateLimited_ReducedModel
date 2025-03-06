#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Noise-shaping with digital calibration.
This code impelemnts the noise-shaping with digital calibration using the transfer function of the noise shaping filter.
@author: Bikash Adhikari 
@date: 06.03.2024
@license: BSD 3-Clause
"""

import numpy as np
import math
from scipy import signal

# Noise shaping function for DAC quantization
# Implements error feedback to shape noise and improve performance

def noise_shaping(Nb, Xcs, b, a, Qstep, YQns, MLns, Vmin, QMODEL):
    """
    Implements noise shaping for quantization by using an error feedback filter.
    
    :param Nb: Number of bits in the quantizer
    :param Xcs: Input signal
    :param b: Feedback filter numerator coefficients
    :param a: Feedback filter denominator coefficients
    :param Qstep: Quantization step size
    :param YQns: Ideal quantization levels
    :param MLns: Measured quantization levels
    :param Vmin: Minimum quantization voltage
    :param QMODEL: Quantization model selection (1: Ideal, 2: Measured)
    :return: Quantized output codes
    """
    
    # Ensure input arrays are properly shaped
    YQns = YQns.squeeze()
    MLns = MLns.squeeze()
    
    C_NSD = np.zeros((1, Xcs.size)).astype(int)  # Storage for quantized codes
    err_buffer = np.zeros_like(b)  # Buffer for error feedback
    u_ns = np.zeros_like(Xcs)  # Storage for noise-shaped output
    
    # Iterate over the input signal to perform noise shaping
    for i in range(len(Xcs)):
        
        # Compute desired signal with noise shaping error feedback
        desired = Xcs[i] - np.dot(b, err_buffer)
        
        # Perform quantization using rounding
        q = math.floor(desired/Qstep + 0.5)  
        
        # Compute DAC code by adjusting with minimum quantization level
        c = q - math.floor(Vmin/Qstep)  
        
        # Clip the code within allowable DAC range
        if c > 2**Nb - 1:
            c = 2**Nb - 1
        elif c < 0:
            c = 0
        
        # Store computed DAC code
        C_NSD[0, i] = c  
        
        # Apply quantization model to determine actual output level
        match QMODEL:
            case 1:
                u_ns[i] = YQns[c]  # Use ideal quantization levels
            case 2:
                u_ns[i] = MLns[c]  # Use measured quantization levels
        
        # Compute quantization error
        error = u_ns[i] - desired 
        
        # Update error buffer for next iteration
        err_buffer[0] = - error  
        err_buffer[0] = - np.dot(a, err_buffer)  
        err_buffer = np.roll(err_buffer, 1)  # Shift buffer for next computation
    
    return C_NSD  # Return computed DAC codes


