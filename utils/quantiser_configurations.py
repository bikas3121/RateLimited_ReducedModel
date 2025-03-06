# -*- coding: utf-8 -*-
"""
@author: Bikash Adhikari 
@date: 22.02.2024
@license: BSD 3-Clause
"""

import numpy as np
import os
import scipy
import csv
import sys


# Quantiser type
class quantiser_type:
    midtread = 1        # midtread quantiser
    midriser = 2        # midrise quantiser

# Quantiser_word_size
class qws:  
    w_04bit = 1
    w_06bit = 2
    w_08bit = 3
    w_12bit = 4
    w_16bit = 5


class mhoq_type:
    binary = 1
    scaled = 2

def quantiser_configurations(QConfig):
    """
    Return specified quantiser model configuration, given QConfig selector.
    """
    
    match QConfig:
        case qws.w_04bit:
            Nb = 4 # word-size
            Mq = 2**Nb - 1; # max. code
            Qtype = quantiser_type.midtread
        case qws.w_06bit:
            Nb = 6 # word-size
            Mq = 2**Nb - 1; # max. code
            Qtype = quantiser_type.midtread
        case qws.w_08bit:
            Nb = 8 # word-size
            Mq = 2**Nb - 1; # max. code
            Qtype = quantiser_type.midtread
        case qws.w_12bit:
            Nb = 12 # word-size
            Mq = 2**Nb - 1; # max. code
            Qtype = quantiser_type.midtread
        case qws.w_16bit:
            Nb = 16 # word-size
            Mq = 2**Nb - 1; # max. code
            Qtype = quantiser_type.midtread
        case _:
            sys.exit("Invalid quantiser configuration selected.")

    # Qantiser range scaled according to the quantiser levels
    Vmin = 0
    Vmax = 2
    Rng = Vmax - Vmin
    # Quantiser step size     
    Qstep = Rng/Mq  # step-size (LSB)
    YQ = np.arange(Vmin, Vmax+Qstep, Qstep)  # ideal ouput levels (mid-tread quantizer)
    YQ = np.reshape(YQ, (-1, YQ.shape[0]))  # generate 2d array with 1 row
    
    return Nb, Mq, Vmin, Vmax, Rng, Qstep, YQ, Qtype


def get_measured_levels(Qconfig):

    # Generate ideal levels
    Nb, Mq, Vmin, Vmax, Rng, Qstep, YQ, Qtype = quantiser_configurations(Qconfig)


    # inpath = '/Volumes/Work/Work_Reports/MPC_ratelimiter_switchingminimisation_for_slewrate/Python/Measured_INL'
    inpath = 'Measured_INL'
    infile = ''

    match Qconfig:
        case qws.w_04bit:
            infile = 'INL_levels_04bit.csv'
            if os.path.exists(os.path.join(inpath, infile)):
                with open(os.path.join(inpath, infile),'r') as f1:
                    reader = csv.reader(f1,delimiter='\t')
                    for row in reader:
                        INL = row
            else: # can't recover from this
                raise SystemExit('No level measurements file found.')

        case qws.w_06bit:
            infile = 'INL_levels_06bit.csv'
            if os.path.exists(os.path.join(inpath, infile)):
                with open(os.path.join(inpath, infile),'r') as f1:
                    reader = csv.reader(f1,delimiter='\t')
                    for row in reader:
                        INL = row
            else: # can't recover from this
                raise SystemExit('No level measurements file found.')
        case qws.w_08bit:

            infile = 'INL_levels_08bit.csv'
            if os.path.exists(os.path.join(inpath, infile)):
                with open(os.path.join(inpath, infile),'r') as f1:
                    reader = csv.reader(f1,delimiter='\t')
                    for row in reader:
                        INL = row
            else: # can't recover from this
                raise SystemExit('No level measurements file found.')
        case qws.w_12bit:

            infile = 'INL_levels_12bit.csv'
            if os.path.exists(os.path.join(inpath, infile)):
                with open(os.path.join(inpath, infile),'r') as f1:
                    reader = csv.reader(f1,delimiter='\t')
                    for row in reader:
                        INL = row
            else: # can't recover from this
                raise SystemExit('No level measurements file found.')
        case qws.w_16bit:

            infile = 'INL_levels_16bit.csv'
            if os.path.exists(os.path.join(inpath, infile)):
                with open(os.path.join(inpath, infile),'r') as f1:
                    reader = csv.reader(f1,delimiter='\t')
                    for row in reader:
                        INL = row
            else: # can't recover from this
                raise SystemExit('No level measurements file found.')
        case _:
            raise SystemExit('No measurement file exists')
            
    INL = np.array([float(i) for i in INL]).reshape(1,-1)
    ML =  YQ + INL*Qstep
    return ML
