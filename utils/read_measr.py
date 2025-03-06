#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 10:00:14 2023

@author: bikashadhikari
"""

import csv
import matplotlib.pyplot as plt
import numpy as np


# Inputs:
    # file_name : name name and location of the file with INL data
    # nob : number of bits, reads data only for the specified number of bits. 
    
# Output
    # level : DAC levels
    # INL : INL measurements in LSB
    # DNL : DNL measurements in LSB

def read_INL_measurements(file_name, nob):
    headings = []
    level = []
    measured_voltage = []
    with open(file_name, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if row[0].startswith('#'):
                continue
            if not headings:
                headings = row
                continue
            level.append(int(row[0]))
            measured_voltage.append(float(row[2]))
            
    # Take a subset of the measured voltage levels
    measured_voltage = measured_voltage[0:2**nob]
    level = level[0:2**nob]

    ideal_voltage = np.linspace(measured_voltage[0], measured_voltage[-1], num=len(level))

    deviation_in_v = ideal_voltage - measured_voltage
    v_per_lsb = (measured_voltage[-1] - measured_voltage[0]) / len(level)
    deviation_in_lsb = deviation_in_v / v_per_lsb

    DNL = [0] * len(level)
    for code in range(1, level[-1]):
        DNL[code] = 1 - ((measured_voltage[code] - measured_voltage[code-1]) / v_per_lsb)

    return  deviation_in_lsb

## for 8 bit
# level, deviation_in_lsb, DNL  = read_INL_measurements('MeasureINL/measurement_11Oct23.csv', 8)



# plt.figure()
# plt.plot(level, deviation_in_lsb)

# plt.figure()
# plt.plot(level, DNL)