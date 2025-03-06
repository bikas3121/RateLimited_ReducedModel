#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""".
Filter parameters for MHOQ simulations
@author: Bikash Adhikari
@date: 07.11.2024
@license: BSD 3-Clause
"""

import numpy as np
from scipy import signal

def filter_params(LPF, Fc):
    """
    # Returns filter parameters for MHOQ simulation
    LPF  - Type of filter 
    Fc   - Cutoff frequency of the filter
    """  
    match LPF:

        case 1: 
        # Phsycoacoustocially optimal noise shaping filter, Goodwin et al. 2003 
            Fc = 10000 
            b_lpf = np.array([1, 0.91, 0])
            a_lpf = np.array([1 , -1.335, 0.644])

        case 2:      
        # Butter worth  filter
            Fs = 1e6
            Fc = Fc
            Wn = Fc/(Fs/2)
            b_lpf,a_lpf = signal.butter(2, Wn, 'lowpass')
             
        case 3: # H2-Hinf nonextended
        # Optimal filter obtained solving optimisation problem in the paper. 
        # Optimal noise shaping filter that minimises the error variance at the filter output, 
        # Follows parameters are obtained using the relation in equation (16) in the paper.
            match Fc: # ||NTF || < 1.5
                case 1e4: 
                    b_lpf = np.array([1.000000000000000e+00 ,   -1.173964128527319e+00 ,    4.124321001734800e-01])
                    a_lpf = np.array([ 1.000000000000000e+00,    -1.911115041641464e+00  ,   9.148859050894880e-01])
                case 1e5:
                    b_lpf = np.array([1.000000000000000e+00 ,   -1.892962366058354e-01 ,    1.390289638645996e-01])
                    a_lpf = np.array([ 1.000000000000000e+00 ,   -1.142985437884713e+00  ,   4.128062411723096e-01])

        
        case 4: # H2-Hinf nonextended
        # Optimal filter obtained solving optimisation problem in the paper. 
        # Optimal noise shaping filter that minimises the error variance at the filter output, 
        # Follows parameters are obtained using the relation in equation (16) in the paper.
            match Fc: # ||NTF || < 1.5
                case 1e5:
                    b_lpf = np.array([1.000000000000000,   0.025502844105568,   1.242641581596813])
                    a_lpf = np.array([  1.000000000000000,  -1.142977298177462,   0.412799348250127])

        case 5: #H2-H2 nonextended
            match Fc: # ||R[z]-1|| < 1.5
                case 1e4: 
                    b_lpf = np.array([1.000000000000000 , -0.454982085457551 ,  0.208081021767825])
                    a_lpf = np.array([1.000000000000000 , -1.911448263896500 ,  0.915235243028894])
                case 1e5:
                    b_lpf = np.array([1.000000000000000,   0.291270720293934,   0.186607281290184])
                    a_lpf = np.array([ 1.000000000000000,  -1.142991438742473,   0.412813093997644])

        case 6: #H2-H2 nonextended
            match Fc: # ||R[z]-1|| < 2.5 
                case 1e5:
                    b_lpf = np.array([1.000000000000000,   0.237582946282937 ,  0.181565585843148])
                    a_lpf = np.array([ 1.000000000000000 , -1.142992449723465 ,  0.412813897936089])
        
        case 7: #H2-H2 extended
            match Fc: # ||R[z]-1 || < 1.5
                case 1e4: 
                    b_lpf = np.array([1.000000000000000 , -0.453344424878861 ,  1.693648049401660])
                    a_lpf = np.array([1.000000000000000,  -1.910627674787699 ,  0.914427083633073])
                case 1e5:
                    b_lpf = np.array([1.000000000000000 ,  0.214039915500633 ,  1.251327072306659])
                    a_lpf = np.array([1.000000000000000,  -1.142972742286985 ,  0.412794838079740])

        case 8: #H2-H2 extended
            match Fc: # ||R[z]-1 || < 2.5
                case 1e5:
                    b_lpf = np.array([1.000000000000000 ,  0.261952757879720,   1.260913433779185])
                    a_lpf = np.array([ 1.000000000000000,  -1.142972594032139,   0.412794601182426])
    return b_lpf, a_lpf, Fc



