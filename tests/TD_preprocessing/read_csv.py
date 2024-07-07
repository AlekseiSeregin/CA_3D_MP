import csv
import pickle
import os
from scipy.spatial import KDTree
import random


class CompPool:
    def __init__(self):
        self.primary = 0
        self.secondary = 0


def read_csv_files(directory):
    data = {}
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            with open(file_path, "r") as file:
                reader = csv.DictReader(file)
                data[filename] = [row for row in reader]
    return data


def read_csv_files2(directory):
    data = {}
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            with open(file_path, "r") as file:
                reader = csv.DictReader(file)
                for row in reader:
                    concentrations = {'Cr': float(row['Cr']), 'Al': float(row['Al']), 'O': float(row['O'])}
                    comp_pool = CompPool()
                    # Check if Corundum1_Content or Corundum2_Content is not empty
                    corundum1_content = row.get('Corundum1_Content', '').strip()
                    corundum2_content = row.get('Corundum2_Content', '').strip()
                    if corundum1_content or corundum2_content:
                        # Determine whether it's Cr2O3 or Al2O3 based on Al and Cr concentrations
                        corundum1_al_cont = row.get('Corundum1_Al_Cont', '').strip()
                        corundum1_cr_cont = row.get('Corundum1_Cr_Cont', '').strip()
                        corundum2_al_cont = row.get('Corundum2_Al_Cont', '').strip()
                        corundum2_cr_cont = row.get('Corundum2_Cr_Cont', '').strip()

                        if corundum1_content and (
                                not corundum2_content or float(corundum1_content) >= float(corundum2_content)):
                            if corundum1_al_cont and (
                                    not corundum1_cr_cont or float(corundum1_al_cont) >= float(corundum1_cr_cont)):
                                comp_pool.secondary = float(corundum1_content)
                            else:
                                comp_pool.primary = float(corundum1_content)
                        elif corundum2_content:
                            if corundum2_al_cont and (
                                    not corundum2_cr_cont or float(corundum2_al_cont) >= float(corundum2_cr_cont)):
                                comp_pool.secondary = float(corundum2_content)
                            else:
                                comp_pool.primary = float(corundum2_content)
                    data[tuple(concentrations.values())] = comp_pool
    return data


def read_csv_files3(directory):
    data = {}
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            with open(file_path, "r") as file:
                reader = csv.DictReader(file)
                for row in reader:
                    concentrations = {'Cr': float(row['Cr']), 'Al': float(row['Al']), 'O': float(row['O'])}
                    comp_pool = CompPool()
                    # Check if Corundum1_Content or Corundum2_Content is not empty
                    corundum1_content = row.get('Corundum1_Content', '').strip()
                    corundum2_content = row.get('Corundum2_Content', '').strip()
                    # Check if Spinel1_Content or Spinel2_Content is not empty
                    spinel1_content = row.get('Spinel1_Content', '').strip()
                    spinel2_content = row.get('Spinel2_Content', '').strip()
                    if corundum1_content or corundum2_content or spinel1_content or spinel2_content:
                        # Determine whether it's Cr2O3 or Al2O3 based on Al and Cr concentrations
                        def determine_phase(primary_content, secondary_content, primary_al_cont, primary_cr_cont,
                                            secondary_al_cont, secondary_cr_cont):
                            if primary_content and (
                                    not secondary_content or float(primary_content) >= float(secondary_content)):
                                if primary_al_cont and (
                                        not primary_cr_cont or float(primary_al_cont) >= float(primary_cr_cont)):
                                    return 'secondary'
                                else:
                                    return 'primary'
                            elif secondary_content:
                                if secondary_al_cont and (
                                        not secondary_cr_cont or float(secondary_al_cont) >= float(secondary_cr_cont)):
                                    return 'secondary'
                                else:
                                    return 'primary'
                            return None

                        primary_phase = determine_phase(corundum1_content, corundum2_content,
                                                        row.get('Corundum1_Al_Cont', '').strip(),
                                                        row.get('Corundum1_Cr_Cont', '').strip(),
                                                        row.get('Corundum2_Al_Cont', '').strip(),
                                                        row.get('Corundum2_Cr_Cont', '').strip())
                        if primary_phase == 'primary':
                            comp_pool.primary += float(corundum1_content) if corundum1_content else 0
                            comp_pool.primary += float(corundum2_content) if corundum2_content else 0
                        elif primary_phase == 'secondary':
                            comp_pool.secondary += float(corundum1_content) if corundum1_content else 0
                            comp_pool.secondary += float(corundum2_content) if corundum2_content else 0

                        spinel1_phase = determine_phase(spinel1_content, None, row.get('Spinel1_Al_Cont', '').strip(),
                                                        row.get('Spinel1_Cr_Cont', '').strip(), None, None)
                        if spinel1_phase == 'primary':
                            comp_pool.primary += float(spinel1_content) if spinel1_content else 0
                        elif spinel1_phase == 'secondary':
                            comp_pool.secondary += float(spinel1_content) if spinel1_content else 0

                        spinel2_phase = determine_phase(spinel2_content, None, row.get('Spinel2_Al_Cont', '').strip(),
                                                        row.get('Spinel2_Cr_Cont', '').strip(), None, None)
                        if spinel2_phase == 'primary':
                            comp_pool.primary += float(spinel2_content) if spinel2_content else 0
                        elif spinel2_phase == 'secondary':
                            comp_pool.secondary += float(spinel2_content) if spinel2_content else 0

                    data[tuple(concentrations.values())] = comp_pool
    return data


def write_data_to_file(data, output_file):
    with open(output_file, "wb") as file:
        pickle.dump(data, file)


def load_data_from_file(input_file):
    with open(input_file, "rb") as file:
        data = pickle.load(file)
    return data


def post_process_dict(my_dict):
    prev_value = None
    for key, value in my_dict.items():
        if value.primary == 0 and value.secondary == 0:
            if prev_value is not None:
                my_dict[key] = prev_value
        else:
            prev_value = value


def find_closest_key(target, tree, keys):
    dist, idx = tree.query(target)
    return keys[idx]


if __name__ == "__main__":
    # Example usage:
    # directory = "W:/SIMCA/TC/Simulations_Klaus_first_ALL/"
    # output_file = "consolidated_data.pkl"
    p_output_file = "TD_look_up.pkl"

    # Read data from CSV files
    # data = read_csv_files3(directory)

    # Write data to a single file
    # write_data_to_file(data, output_file)

    # Load data from the consolidated file
    # consolidated_data = load_data_from_file(output_file)

    # post_process_dict(consolidated_data)

    # write_data_to_file(consolidated_data, p_output_file)

    p_consolidated_data = load_data_from_file(p_output_file)

    keys = list(p_consolidated_data.keys())
    tree = KDTree(keys)

    # for key, value in consolidated_data.items():
    #     if value.primary != 0 and value.secondary != 0:
    #         print(key)

    for _ in range(100):
        # print("next value: ")
        # print("Cr: ")
        # cr_c = float(input())
        # print("Al: ")
        # al_c = float(input())
        # print("O: ")
        # o_c = float(input())

        # cr_c = np.around(np.linspace(0, 25, 10), decimals=4)
        cr_c = random.uniform(0, 40)
        # al_c = np.around(np.linspace(0, 2.5, 10), decimals=4)
        al_c = random.uniform(0, 40)
        # o_c = np.around(np.linspace(0, 60, 10), decimals=8)
        o_c = random.uniform(0, 60)

        target_value = (cr_c, al_c, o_c)
        print(target_value)

        closest_key = find_closest_key(target_value, tree, keys)
        print(closest_key)

        value_from_dict = p_consolidated_data[closest_key]

        print("Cr_oxide: ", value_from_dict.primary)
        print("Al_oxide: ", value_from_dict.secondary)
