import numpy as np
import pickle
from scipy.spatial import KDTree
import os


class CompPool:
    def __init__(self):
        self.corundum_cr = 0.0
        self.corundum_al = 0.0
        self.spinel_cr = 0.0
        self.spinel_al = 0.0
        self.halite = 0.0


class TdDATA:
    def __init__(self):
        # Get the absolute path of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Specify the file name (e.g., 'data.txt') in the same directory
        self.TD_file = os.path.join(script_dir, 'TD_look_up.pkl')
        self.TD_lookup = None
        self.keys = None
        self.tree = None

    def fetch_look_up_from_file(self):
        with open(self.TD_file, "rb") as file:
            self.TD_lookup = pickle.load(file)
        self.keys = list(self.TD_lookup.keys())
        self.tree = KDTree(self.keys)
        self.keys = np.array(self.keys)

    def get_look_up_data(self, primary, secondary, oxidant):
        # Convert input lists to numpy arrays
        targets = np.array((primary, secondary, oxidant)).T
        # Query the KD-tree for nearest neighbors
        distances, indexes = self.tree.query(targets)
        # Retrieve nearest keys directly from indexes
        nearest_keys = self.keys[indexes]
        # Retrieve objects directly from TD_lookup using nearest_keys
        objects = [self.TD_lookup[tuple(key)] for key in nearest_keys]
        # Extract primary and secondary attributes from objects
        return np.array([[obj.corundum_cr, obj.corundum_al, obj.spinel_cr, obj.spinel_al, obj.halite]
                         for obj in objects]).T


if __name__ == '__main__':
    test_data = TdDATA()
    test_data.fetch_look_up_from_file()

