#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dynamic rate limiter

@author: Arnfinn Eielsen
@date: 02.09.2024
@license: 
"""

import numpy as np


def sat(a, b, x):
    """
    Saturate; clamp a value to be between the limits a and b, a != b
    """
    
    if a < b:
        y = np.max([a, np.min([b, x])])
    # else:
    #     y = np.max([b, np.min([a, x])])
    
    return y


def rate_limiter(u0, u1, t0, t1, R, F):
    """
    Limit the rate of change.
    
    u0 - previous (ordinate) value
    u1 - current (ordinate) value
    t0 - previous time/abscissa value
    t1 - current time/abscissa value
    R - rising rate limit
    F - falling rate limit
    """
    
    delta_t = t1 - t0
    rate = (u1 - u0)/delta_t

    y = u0 + sat(F*delta_t, R*delta_t, rate)

    return y, rate


