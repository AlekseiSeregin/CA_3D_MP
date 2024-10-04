import numpy as np
import pickle
from scipy.spatial import KDTree
import os


class KinDATA:
    def __init__(self, file_name):
        # Get the absolute path of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Specify the file name (e.g., 'data.txt') in the same directory
        self.TD_file = os.path.join(script_dir, file_name)
        self.TD_lookup = None
        self.keys = None
        self.tree = None

    def fetch_look_up_from_file(self):
        with open(self.TD_file, "rb") as file:
            self.TD_lookup = pickle.load(file)
        self.keys = list(self.TD_lookup.keys())
        self.tree = KDTree(self.keys)
        self.keys = np.array(self.keys)

    def get_look_up_data(self, times, positions):
        # Convert input lists to numpy arrays
        targets = np.array((times, positions)).T
        # Query the KD-tree for nearest neighbors
        distances, indexes = self.tree.query(targets)
        # Retrieve nearest keys directly from indexes
        nearest_keys = self.keys[indexes]
        # Retrieve objects directly from TD_lookup using nearest_keys
        return np.array([self.TD_lookup[tuple(key)] for key in nearest_keys])


if __name__ == '__main__':
    test_data = KinDATA("LUT_NiCr5.pkl")
    test_data.fetch_look_up_from_file()
    # t = [15010, 10010, 20010, 5010, 11411]
    t = [11411, 11411, 11411, 11411, 11411]
    p = [10e-6, 20e-6, 30e-6, 50e-6, 40e-6]
    some = test_data.get_look_up_data(t, p)
    print()

