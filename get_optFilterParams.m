function [B_LPF]  =  get_optFilterParams(Fs, Fc)
% MATLAB implementation of the output filter aware noise-shaping functions
% using YALMIP, a modeling tool for optimization problems

    yalmip('clear')
    format long
    %% Sampling frequency/ rate
    Ts = 1/Fs;   % Compute the sampling period
    
    %% Plant/Reconstruction Filter
    n = 3; % Filter order. 
    Wn = Fc/(Fs/2);
    [b1,a1] = butter(n,Wn,"low");  % Design a low-pass Butterworth filter
    
    % Convert transfer function to state-space representation
    [Ah, Bh, Ch, Dh] = tf2ss(b1,a1);
    
    %% Optimization problem setup     
    % Define optimization variables using YALMIP
    
    Pf = sdpvar(n);
    Pg = sdpvar(n);
    Wf = sdpvar(1,n);
    Wg = sdpvar(n,1);
    L = sdpvar(n);
    mu_e  = sdpvar(1);
    
    % Define constraints
    gamma_lee = 1.5^2;  % Lee's constraint
    
    % Construct matrices for optimization
    MA = [Ah*Pf + Bh*Wf , Ah; L ,Pg*Ah];
    MB = [Bh ; Wg];
    MC = [Ch*Pf + Dh*Wf ,  Ch];
    MP = [Pf, eye(n); eye(n), Pg];
    
    MC_tilde = [Wf, zeros(1,n)];
    
    % Define constraint matrices
    C1 = [MP MA MB; MA' MP zeros(2*n,1); MB' zeros(1,2*n) eye(1)];
    C2 = [mu_e MC Dh'; MC' MP zeros(2*n,1); Dh zeros(1,2*n) eye(1)];
    C3 = [MP MA  MB zeros(2*n,1); MA' MP zeros(2*n,1) MC_tilde'; MB' zeros(1, 2*n) gamma_lee eye(1); zeros(1,2*n) MC_tilde 1 gamma_lee];
    
    % Define optimization constraints and settings
    F = [Pf  >= 0, Pg  >= 0, C1   >= 0, C2 >= 0, C3 >= 0];
    ops = sdpsettings('solver','mosek');
    ops.verbose = 0;
    ops.showprogress = 0;
    ops.debug = 0;
    optimize(F, mu_e, ops)
    
    %% Extract values of optimization variables
    Pf = value(Pf);
    Pg = value(Pg);

    
    % Retrieve optimized values
    Wf = value(Wf);
    Wg = value(Wg);
    L = value(L);
    value(mu_e)
    inv_Pg = inv(Pg);
    Sf = Pf - inv_Pg;
    inv_Sf = inv(Sf);
    
    
    %% Compute noise transfer function (NTF)
    %State state representation of R(z) ; see publication 
    Ar = (Bh*Wf - inv_Pg*(L - Pg*Ah*Pf))*inv_Sf;
    Br = Bh - inv_Pg*Wg;
    Cr = Wf*inv_Sf;
    Dr = 1;
    [br, ar] = ss2tf(Ar, Br, Cr, Dr);

    %% LOW PASS FILTER -  PARAMTERS TO BE USED FOR MPC SIMULATION
    % Compute transfer function for low pass filter H(z)
    b_lpf = ar;
    a_lpf = br;
    B_LPF = [b_lpf; a_lpf];
end