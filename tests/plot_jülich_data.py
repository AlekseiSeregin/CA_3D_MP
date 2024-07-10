import re
import matplotlib.pyplot as plt
import pandas as pd
import glob
import numpy as np


# Function to parse a single file
def parse_file(file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()

    # Extract time stamp
    time_stamp = None
    for line in data:
        if "Current time:" in line:
            time_stamp = float(re.search(r"Current time:\s+([\d.E+-]+)", line).group(1))
            break

    # Extract SPINEL data
    spinel_data = []
    for ind, line in enumerate(data):
        if ind > 2:
            values = line.split()
            spinel = float(values[7])
            corundum = float(values[8])
            sum_of_both = spinel + corundum
            spinel_data.append(sum_of_both)

    return time_stamp, spinel_data


# Function to process all files
def process_files(file_pattern):
    file_paths = glob.glob(file_pattern)
    time_series = []
    spinel_series = []

    for file_path in file_paths:
        time_stamp, spinel_data = parse_file(file_path)
        time_series.append(time_stamp)
        spinel_series.append(spinel_data)

    return time_series, spinel_series



def plot_spinel_over_time(time_series, spinel_series, layers):
    plt.figure(figsize=(10, 6))
    for layer in layers:
        plt.plot(time_series, [spinel[layer] for spinel in spinel_series], label=f'Layer {layer}')

    plt.xlabel('Time')
    plt.ylabel('SPINEL')
    plt.title('SPINEL Data Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()


# Example usage
file_pattern = 'C:/Users/adam-wrmjvo101twvweh/Downloads/outputs/out*.txt'
time_series, spinel_series = process_files(file_pattern)
layers = np.arange(0, 100, 10)
# layers = [0, 13, 20, 50, 90, 99]
# Specify the layers you want to plot
plot_spinel_over_time(time_series, spinel_series, layers)
