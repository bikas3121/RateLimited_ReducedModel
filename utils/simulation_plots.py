#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""  
Plot the variance plots for the given methods.

This script provides functions to generate bar and line plots for analyzing 
error variance, Signal-to-Noise and Distortion Ratio (SINAD), or 
Effective Number of Bits (ENOB) for different methods.

@author: Bikash Adhikari
@date: 24.03.2024
@license: BSD 3-Clause
"""

import numpy as np
import matplotlib.pyplot as plt

# %% Bar plot function for error variance, SINAD, or ENOB
def bar_plot(descr='', **kwargs):
    """
    Generate a bar plot for the given key-value pairs.

    Parameters:
    descr (str): Title/description of the plot.
    kwargs: Dictionary of method names (keys) and their corresponding values (values).

    The function rounds values to 5 decimal places for display and plots them as a bar chart.
    """
    x = []  # List to store method names (keys)
    y = []  # List to store rounded values for plotting
    y1 = []  # List to store original values

    # Extract keys (method names) and values into separate lists
    for key, value in kwargs.items():
        x.append(key)
        y.append(np.round(value, 5))  # Round values to 5 decimal places for display
        y1.append(value)  # Store original values

    # Create figure and axis for the plot
    fig, ax = plt.subplots()

    # Define x-axis positions for bars
    x_pos = np.arange(len(x))

    # Create bar plot
    ax.bar(x, y, label=x)
    plt.xticks(x_pos, x)  # Set x-axis labels

    # Annotate bars with their respective values
    for i in range(len(x)):
        plt.text(x=x_pos[i] - 0.1, y=y[i] + 0.015 * y[i], s=y[i], size=10)

    # Set labels and title
    ax.set_ylabel('ENOB')
    ax.set_title(descr)
    ax.legend()

    # Display the plot
    plt.show()

    # Print values for reference
    for i in range(len(x)):
        print(f"{x[i]} {descr}: {y1[i]}")

# %% Line plot function for different linearization methods
def line_plot(lin_methods, values1, values2, SINAD_th, descr=''):
    """
    Generate a line plot comparing different linearization methods.

    Parameters:
    lin_methods (list): List of linearization method names.
    values1 (list): List of SINAD values for the first method.
    values2 (list): List of SINAD values for the second method.
    SINAD_th (float): Theoretical SINAD threshold value.
    descr (str): Label for the y-axis.

    The function plots SINAD values for two different methods and marks the theoretical SINAD threshold.
    """
    # Generate x-axis values corresponding to methods
    x = np.linspace(1, len(lin_methods), len(lin_methods))

    # Define the annotation text for the theoretical SINAD threshold
    txt = f"SINAD_TH = {SINAD_th} dB"

    # Create figure and axis for the plot
    fig, ax = plt.subplots()

    # Plot values for both methods with markers
    ax.plot(x, values1, linestyle="-", marker="o", label="$H[z]$")
    ax.plot(x, values2, linestyle="-", marker="o", label="$H^{*}[z]$")

    # Add horizontal line for theoretical SINAD threshold
    ax.axhline(y=79.9, color='r', linestyle='--')

    # Annotate the threshold line
    ax.text(3, 82, txt)

    # Configure x-axis labels
    ax.set_xticks(x)
    ax.set_xticklabels(lin_methods)

    # Set axis labels and title
    ax.set_xlabel("Linearisation methods")
    ax.set_ylabel(descr)

    # Enable grid lines on the y-axis
    ax.grid(visible=True, which='both', axis='y', linestyle=':')

    # Add legend
    ax.legend()

    # Set y-axis limits
    ax.set_ylim(20, 90)

    # Display the plot
    plt.show()
