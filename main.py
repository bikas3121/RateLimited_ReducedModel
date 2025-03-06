"""  Run DAC simulation rate limited 

@author: Bikash Adhikari 
@date: 23.02.2024
@license: BSD 3-Clause
"""

# %% 
import numpy as np
from scipy  import signal
import scipy
import csv
import matplotlib.pyplot as plt
import statistics
import math 
import tqdm
import gurobipy as gp
from gurobipy import GRB
import matlab.engine


# %% Import custom utility functions
from utils.static_dac_model import quantise_signal, generate_code, generate_dac_output
from utils.quantiser_configurations import quantiser_configurations, get_measured_levels
from utils.MHOQ_RateLim import MHOQ_RLIM
from utils.nsdcal import noise_shaping
from utils.process_sim_output import process_sim_output
from utils.balreal import balreal
from utils.welch_psd import welch_psd
from utils.simulation_plots import bar_plot 
from utils.figures_of_merit import TS_SINAD
from utils.dac_slew import rate_limiter
from utils.rate_limiter import rate_lim

#%% # Generate test signal
def test_signal(SCALE, MAXAMP, FREQ, Rng,  OFFSET, t):
    Xcs = (SCALE/100)*MAXAMP*np.sin(2*np.pi*FREQ*t) + Rng/2  
    return Xcs 


# %% Quantiser configurations 
Qconfig = 4
Nb, Mq, Vmin, Vmax, Rng, Qstep, YQ, Qtype = quantiser_configurations(Qconfig)

# %% Sampling frequency and rate
Fs = 1e6
Ts = 1/Fs

# %% Generate time vector
Xcs_SCALE = 100
Xcs_FREQ = 9999
match 2:
    case 1:  # specify duration as number of samples and find number of periods
        Nts = 1e5  # no. of time samples
        Np = np.ceil(Xcs_FREQ*Ts*Nts).astype(int) # no. of periods for carrier

    case 2:  # specify duration as number of periods of carrier
        Np = 50 # no. of periods for carrier
        
Npt = 1  # no. of carrier periods to use to account for transients
Np = Np + Npt

t_end = Np/Xcs_FREQ  # time vector duration
t = np.arange(0, t_end, Ts)  # time vector

# %% Generate carrier/test signal
SIGNAL_MAXAMP = Rng/2
SIGNAL_OFFSET = -Qstep/2  # try to center given quantiser type
Xcs = test_signal(Xcs_SCALE, SIGNAL_MAXAMP, Xcs_FREQ, Rng,  SIGNAL_OFFSET, t)

# %% Reconstruction filter
N_lp = 3        # Filter order
Fc = 1e5       # Filter cutoff frequency

# %% Optimal filter parameters
# Change the filter order in get_optFilterParams.m file if necessary. The default value is set as N_lp = 3.
match 2:
    case 1:     # Calculates the optimal filter parmaters by running the MATLAB  code.
        eng = matlab.engine.start_matlab()
        B_LPF = eng.feval('get_optFilterParams', Fs, Fc)
        b_lpf = np.array(B_LPF)[0]
        a_lpf = np.array(B_LPF)[1]
        A, B, C, D  = signal.tf2ss(b_lpf, a_lpf)  # Tf to state space
    case 2:   # Optimal filter paramters for Fs = 1e6, Fc = 1e5, and N_lp =3
        b_lpf = np.array([ 1.        , -0.74906276,  0.35356745, -0.05045204])
        a_lpf = np.array([ 1.        , -1.76004281,  1.18289728, -0.27806204])
        A, B, C, D  = signal.tf2ss(b_lpf, a_lpf)  # Tf to state space

# %% Quatniser Model
# Quantiser model: 1 - Ideal , 2- Calibrated
QMODEL = 2

# %% Quantiser levels : Ideal and Measured
# Ideal levels 
YQns = YQ.squeeze()     
# Measured Levels
MLns  =  get_measured_levels(Qconfig).squeeze()

# %%  Add measurement noise to the measured levels MLns
if Nb == 6: 
    ML_err_rng = Qstep/pow(2, 12) # (try to emulate 18-bit measurements (add 12 bit))
elif Nb == 10 or Nb == 8 or Nb== 12: 
    ML_err_rng = Qstep/pow(2, 8) # (try to emulate 18-bit measurements (add 8 bit))
elif Nb == 16: 
    ML_err_rng = Qstep/pow(2, 2) # (try to emulate 18-bit measurements (add 2 bit))
else:
    sys.exit('MPC: Unknown QConfig for ML error')
MLns_err = np.random.uniform(-ML_err_rng, ML_err_rng, MLns.shape)
MLns_E = MLns + MLns_err # Measured levels with error


# %% Direct Quantization 
C_DQ = (np.floor(Xcs/Qstep +1/2)).astype(int)
match QMODEL:
    case 1:
        Xcs_DQ = generate_dac_output(C_DQ, YQns)
    case 2:
        Xcs_DQ = generate_dac_output(C_DQ, MLns)


# %% NSD CAL
b_nsf = b_lpf - a_lpf
a_nsf = b_lpf
C_NSQ = noise_shaping(Nb, Xcs, b_nsf, a_nsf, Qstep, YQns, MLns, Vmin, QMODEL)  
match QMODEL:
    case 1:
        Xcs_NSQ = generate_dac_output(C_NSQ, YQns)
    case 2:
        Xcs_NSQ = generate_dac_output(C_NSQ, MLns) 

# %% MHOQ Ratelimit
N_PRED = 1      # Prediction horizon 
# Steps constraints for the MPC search space. MPC is constrained by:    current input- Step  <= current input <=  current input +  Step  
Step = 200

MHOQ_RL = MHOQ_RLIM(Nb, Qstep, QMODEL, A, B, C, D)
C_MHOQ = MHOQ_RL.get_codes(N_PRED, Xcs, YQns, MLns, Step)
match QMODEL:
    case 1:
        Xcs_MHOQ = generate_dac_output(C_MHOQ, YQns)
    case 2:
        Xcs_MHOQ = generate_dac_output(C_MHOQ, MLns) 

# %% Trim vector lengths  
tm = t[:Xcs_MHOQ.shape[1]]
len_tm = len(tm)
TRANSOFF = np.floor(Npt*Fs/Xcs_FREQ).astype(int)  # remove transient effects from output

# %%  SINAD COmputation
SINAD_COMP_SEL = 1
plot_val = False

FXcs_DQ,  ENOB_DQ= process_sim_output(tm, Xcs_DQ, Fc, Fs, N_lp, TRANSOFF,SINAD_COMP_SEL,  plot=plot_val, descr="NSQ")
FXcs_NSQ,  ENOB_NSQ= process_sim_output(tm, Xcs_NSQ, Fc, Fs, N_lp, TRANSOFF,SINAD_COMP_SEL,  plot=plot_val, descr="NSQ")
FXcs_MHOQ1,  ENOB_MHOQ1= process_sim_output(tm, Xcs_MHOQ, Fc, Fs, N_lp, TRANSOFF,SINAD_COMP_SEL,  plot=plot_val, descr="MHOQ")
# 
bar_plot(descr= "ENOB", DQ = ENOB_DQ, NSDCAL = ENOB_NSQ,   MHOQ1 = ENOB_MHOQ1)


# %% Simulate rate limited DAC
# Parameters for rate limiting quantiser
R = 7e4  # rising rate 
F = -R   # falling rate
Xcs_DQ_RL = rate_lim(Xcs_DQ, t, R, F)
Xcs_NSQ_RL = rate_lim(Xcs_NSQ, t, R, F)
Xcs_MHOQ_RL = rate_lim(Xcs_MHOQ, t, R, F)
FXcs_DQ_RL,  ENOB_DQ_RL= process_sim_output(tm, Xcs_DQ_RL, Fc, Fs, N_lp, TRANSOFF,SINAD_COMP_SEL,  plot=plot_val, descr="NSQ")
FXcs_NSQ_RL,  ENOB_NSQ_RL= process_sim_output(tm, Xcs_NSQ_RL, Fc, Fs, N_lp, TRANSOFF,SINAD_COMP_SEL,  plot=plot_val, descr="NSQ")
FXcs_MHOQ1_RL,  ENOB_MHOQ1_RL= process_sim_output(tm, Xcs_MHOQ_RL, Fc, Fs, N_lp, TRANSOFF,SINAD_COMP_SEL,  plot=plot_val, descr="MHOQ")
# 
bar_plot(descr= "ENOB_RL", DQ = ENOB_DQ_RL, NSDCAL = ENOB_NSQ_RL,   MHOQ1 = ENOB_MHOQ1_RL)
 # %%
sl = 1000
fig, ax = plt.subplots()
ax.plot(t[10:sl], Xcs[10:sl])
ax.plot(t[10:sl], Xcs_MHOQ.squeeze()[10:sl])
ax.legend(['Ref','MPC output'])
plt.show()

# %%
