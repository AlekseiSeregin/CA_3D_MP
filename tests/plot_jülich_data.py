import re
import matplotlib.pyplot as plt
import pandas as pd
import glob
import numpy as np
import pickle
from scipy.spatial import KDTree
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
    width_data = []
    int_width = []
    for ind, line in enumerate(data):
        if ind > 2:
            values = line.split()
            spinel = float(values[7])
            corundum = float(values[8])
            int_width.append(float(values[10]))
            sum_of_both = spinel + corundum
            spinel_data.append(sum_of_both)

    prev = 0
    for item in int_width:
        width_data.append(prev + item)
        prev += item

    return time_stamp, spinel_data, width_data


# Function to process all files
def process_files(file_pattern):
    file_paths = glob.glob(file_pattern)
    time_series = []
    spinel_series = []
    width_series = []

    for file_path in file_paths:
        time_stamp, spinel_data, width_data = parse_file(file_path)
        time_series.append(time_stamp)
        spinel_series.append(spinel_data)
        width_series.append(width_data)

    return time_series, spinel_series, width_series



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

def write_data_to_file(data, output_file):
    with open(output_file, "wb") as file:
        pickle.dump(data, file)


def load_data_from_file(input_file):
    with open(input_file, "rb") as file:
        data = pickle.load(file)
    return data


def find_closest_key(target, tree, keys):
    dist, idx = tree.query(target)
    return keys[idx]


if __name__ == "__main__":
    # Example usage
    # file_pattern = 'C:/Users/adam-wrmjvo101twvweh/Downloads/outputs/out*.txt'
    file_pattern = 'C:/Users/alexe/Downloads/ni5cr_1000it/ni5cr_1000it/out*.txt'

    # output_file = "C:/CA_3D_MP/thermodynamics/LUT_NiCr5.pkl"
    #
    time_series, spinel_series, width_series = process_files(file_pattern)
    # print()
    #
    # data = {}
    # for time, spin, widths in zip(time_series, spinel_series, width_series):
    #     for ind, pos in enumerate(widths):
    #         data[(time, pos)] = spin[ind]
    #
    # # Write data to a single file
    # write_data_to_file(data, output_file)
    #
    # # Load data from the consolidated file
    # consolidated_data = load_data_from_file(output_file)
    # # keys = [(key,) for key in consolidated_data.keys()]
    # keys = list(consolidated_data.keys())
    # tree = KDTree(keys)
    #
    # target_values = [(15010, 3e-7), (10010, 4e-7), (20010, 2e-7), (5010, 5e-7)]
    #
    # for target_value in target_values:
    #
    #     closest_key = find_closest_key(target_value, tree, keys)
    #     print(closest_key)
    #
    #     some = consolidated_data[closest_key]
    #     print()
    layers = np.arange(0, 102, 1)

    # layers = [0, 1, 2, 3, 4, 5]
    # Specify the layers you want to plot
    plot_spinel_over_time(time_series, spinel_series, layers)

