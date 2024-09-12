import numpy as np
import pickle
from scipy.spatial import KDTree
import os


class Component:
    def __init__(self, constitution, amount):
        self.constitution = constitution
        self.amount = amount


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

    def gen_table_nested_dict(self):
        # Define the composition ranges for Cr, Al, and O
        cr_range = np.around(np.linspace(0, 25, 10), decimals=4)  # Example Cr composition values
        al_range = np.around(np.linspace(0, 2.5, 10), decimals=4)  # Example Al composition values
        o_range = np.around(np.linspace(0, 60, 10), decimals=8)  # Example O composition values

        # Populate the lookup table with probabilities
        for cr in cr_range:
            self.lookup_table[cr] = {}  # Create nested dictionary for Cr
            for al in al_range:
                self.lookup_table[cr][al] = {}  # Create nested dictionary for Al
                for o in o_range:
                    # Assign the corresponding probability for the composition values
                    self.lookup_table[cr][al][o] = 5.1

    def gen_table_dict(self):
        # Define the composition ranges for Cr, Al, and O
        cr_range = np.around(np.linspace(0, 40, 101), decimals=4)  # Example Cr composition values
        al_range = np.around(np.linspace(0, 40, 101), decimals=4)  # Example Al composition values
        o_range = np.around(np.linspace(0, 60, 101), decimals=8)  # Example O composition values

        # Populate the lookup table with probabilities
        for cr in cr_range:
            for al in al_range:
                for o in o_range:
                    sum_conc = cr + al + o
                    if cr == 0 and al == 0 and o == 0:
                        continue
                    elif sum_conc > 100:
                        continue
                    elif al >= cr and o != 0:
                        # Assign the corresponding probability for the composition values
                        new_comp_pool = CompPool()
                        new_comp_pool.primary = Component("Al2O3", 1)
                        new_comp_pool.secondary = Component("Cr2O3", 0)
                        self.lookup_table[cr, al, o, sum_conc] = new_comp_pool
                        # print(f"{cr} {al} {o:.5f}")
                    elif cr > al and o != 0:
                        new_comp_pool = CompPool()
                        new_comp_pool.primary = Component("Al2O3", 0)
                        new_comp_pool.secondary = Component("Cr2O3", 1)
                        self.lookup_table[cr, al, o, sum_conc] = new_comp_pool

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

