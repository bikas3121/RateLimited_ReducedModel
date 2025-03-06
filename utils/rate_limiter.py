
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dynamic rate limiter based on the rate limiter on the MATLAB/Simulink

@author: Bikash Adhikari
@date: 04.12.2024
@license: 
"""
 
import numpy as np

def rate_limiter(u0, u1, t0, t1, R, F):
    """
    Limit the rate of change.
    
    u0 - previous (ordinate) value
    u1 - current (ordinate) value
    t0 - previous time/abscissa value
    t1 - current time/abscissa value
    R - rising slew rate  parameter
    F - falling slew rate parameter
    """
    
    delta_t = t1 - t0
    rate = (u1 - u0)/delta_t

    if rate + 1e-8 >= R:
        y = delta_t*R + u0
    elif rate -1e-8 <= F: 
        y = delta_t*F + u0
    elif rate <= R or rate >= F:
        y = u1
    return y



def rate_lim(Xcs_rls, t, R, F):    

    # Input signal
    Xcs_rls = Xcs_rls.squeeze()

    # Store rate limited signal
    Xcs_RL = np.zeros_like(Xcs_rls)
    Xcs_RL[0] = Xcs_rls[0] # same value for the initial time instance.

    u0 = Xcs_rls[0]
    for i in range(1,len(Xcs_rls)):
        u1 = Xcs_rls[i]
        t0 = t[i-1]
        t1 = t[i]
        u0= rate_limiter(u0, u1, t0,t1,R, F) 
        Xcs_RL[i] = u0
    return Xcs_RL