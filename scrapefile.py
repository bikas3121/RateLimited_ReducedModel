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

def state_prediction(st, con):
    return A @ st + B * con  # Compute next state

def q_scaling(X):
    return X.squeeze() / Qstep   # Normalize based on bit depth


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
Step = 2

# MHOQ_RL = MHOQ_RLIM(Nb, Qstep, QMODEL, A, B, C, D)
# C_MHOQ = MHOQ_RL.get_codes(N_PRED, Xcs, YQns, MLns, Step)
# match QMODEL:
#     case 1:
#         Xcs_MHOQ = generate_dac_output(C_MHOQ, YQns)
#     case 2:
#         Xcs_MHOQ = generate_dac_output(C_MHOQ, MLns) 


# %% Suboptimal MPC problem setup


# # Scale input signals for numerical stability
Xcs = q_scaling(Xcs)  # Scale reference signal
QL_I = q_scaling(YQns)  # Scale reference signal
QL_M = q_scaling(MLns)  # Scale reference signal

C_Store = []  # Storage container for DAC codes
len_MPC = Xcs.size - N_PRED  # Number of optimization iterations
x_dim = A.shape[0]  # Dimension of the state vector
init_state = np.zeros((x_dim, 1))  # Initialize state vector to zero

# Initialize previous control input based on first sample quantization
u_kminus1_ind = int(np.floor(Xcs[0]  + 0.5))
u_kminus1_ind = np.ones(N_PRED) * u_kminus1_ind  # Extend to prediction horizon

# # Constraint set depending on the step size 
# idx = np.where(QL_I == u_kminus1_ind)[0]
# idx = idx[0]
# lb = max(idx - Step,0)
# ub = min(idx + Step,2**Nb-1)+1

# # %%
# YQns_Const = YQns[lb:ub]
# QL_I_Const = QL_I[lb:ub]
# QL_M_Const = QL_M[lb:ub]
#%% Run MPC loop
# MPC loop
for j in tqdm.tqdm(range(len_MPC)):
# for j in range(1):
    env = gp.Env(empty=True)  # Initialize Gurobi environment
    env.setParam("OutputFlag", 0)  # Suppress solver logs
    env.start()
    m = gp.Model("MPC-INL-RL", env=env)  # Create optimization model


    # Reduced (Sub problem )
    # Constraint set depending on the step size 
    idx = np.where(QL_I == u_kminus1_ind)[0]
    idx = idx[0]
    lb = max(idx - Step,0)
    ub = min(idx + Step,2**Nb-1)+1

    YQns_Const = YQns[lb:ub]
    QL_I_Const = QL_I[lb:ub]
    QL_M_Const = QL_M[lb:ub]
    len_SP = len(QL_I_Const) 
    # Define decision variables
    u = m.addMVar((len_SP, N_PRED), vtype=GRB.BINARY, name="u")  # Control variables (binary choice)
    x = m.addMVar((x_dim * (N_PRED + 1), 1), vtype=GRB.CONTINUOUS,  lb = -GRB.INFINITY, ub = GRB.INFINITY, name = "x")  # State variables

    Obj = 0  # Initialize objective function (error minimization)
    m.addConstr(x[0:x_dim, :] == init_state)  # Set initial condition

    # Loop through prediction horizon to set constraints and objective
    for i in range(N_PRED):
        k = x_dim * i  # Index for state vector
        st = x[k:k+x_dim]  # Extract current state

        bin_con = QL_I_Const.reshape(1, -1) @ u[:, i].reshape(-1, 1)  # Convert binary input to actual control value
        con = bin_con - Xcs[j + i]  # Compute control deviation from reference signal

        e_t = C @ x[k:k+x_dim] + D * con  # Compute output error
        Obj += e_t * e_t  # Accumulate squared error in the objective function

        # State transition constraint
        f_value = A @ st + B * con  # Compute next state
        st_next = x[k + x_dim:k + 2 * x_dim]  # Extract next state variable
        m.addConstr(st_next == f_value)  # Enforce state transition constraint
        m.addConstr(gp.quicksum(u) == 1)  # Ensure only one control input is selected

    m.update()
    m.setObjective(Obj, GRB.MINIMIZE)  # Set objective function

    # Gurobi setting for precision  
    m.Params.IntFeasTol = 1e-9
    m.Params.IntegralityFocus = 1
    m.optimize()  # Solve optimization problem

    values = np.array(m.getAttr("X", m.getVars()))  # Extract solution values
    # u_val = values[:u.size].reshape(2**Nb, N_PRED).astype(int)
    u_val = values[:u.size].reshape(len_SP, N_PRED).astype(int)

    # Extract optimal DAC code
    opt_index = np.array([np.nonzero(u_val[:, i])[0][0] for i in range(N_PRED)])[0] 
    # opt_code = QL_I_Const[opt_index]
    opt_code = lb + opt_index 
    C_Store.append(opt_code)


    # Select DAC output based on the quantization model
    match QMODEL:
        case 1:
            U_opt = QL_I[opt_code] 
        case 2:
            U_opt = QL_M[opt_code] 

    # Predict next state based on optimal control input
    con = U_opt - Xcs[j]
    init_state = state_prediction(init_state, con)  # Update state
    u_kminus1_ind = opt_code 


C_MHOQ1  = np.array(C_Store).reshape(1, -1)
match QMODEL:
    case 1:
        Xcs_MHOQ1 = generate_dac_output(C_MHOQ1, YQns)
    case 2:
        Xcs_MHOQ1 = generate_dac_output(C_MHOQ1, MLns) 


# %% Trim vector lengths  
tm = t[:Xcs_MHOQ1.shape[1]]
len_tm = len(tm)
TRANSOFF = np.floor(Npt*Fs/Xcs_FREQ).astype(int)  # remove transient effects from output

# %%  SINAD COmputation
SINAD_COMP_SEL = 1
plot_val = False

FXcs_DQ,  ENOB_DQ= process_sim_output(tm, Xcs_DQ, Fc, Fs, N_lp, TRANSOFF,SINAD_COMP_SEL,  plot=plot_val, descr="NSQ")
FXcs_NSQ,  ENOB_NSQ= process_sim_output(tm, Xcs_NSQ, Fc, Fs, N_lp, TRANSOFF,SINAD_COMP_SEL,  plot=plot_val, descr="NSQ")
# FXcs_MHOQ,  ENOB_MHOQ= process_sim_output(tm, Xcs_MHOQ, Fc, Fs, N_lp, TRANSOFF,SINAD_COMP_SEL,  plot=plot_val, descr="MHOQ") 
FXcs_MHOQ1,  ENOB_MHOQ1= process_sim_output(tm, Xcs_MHOQ1, Fc, Fs, N_lp, TRANSOFF,SINAD_COMP_SEL,  plot=plot_val, descr="MHOQ") 
bar_plot(descr= "ENOB", DQ = ENOB_DQ, NSDCAL = ENOB_NSQ,   MHOQ_SP = ENOB_MHOQ1)


 # %%
# sl = 1000
# fig, ax = plt.subplots()
# ax.plot(t[10:sl], Xcs[10:sl])
# ax.plot(t[10:sl], Xcs_MHOQ.squeeze()[10:sl])
# ax.legend(['Ref','MPC output'])
# plt.show()

# %%
