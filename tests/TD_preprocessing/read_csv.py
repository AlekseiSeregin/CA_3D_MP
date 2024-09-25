import csv
import pickle
import os

import numpy as np
from scipy.spatial import KDTree
import random


class CompPool:
    def __init__(self):
        self.corundum_cr = 0.0
        self.corundum_al = 0.0
        self.spinel_cr = 0.0
        self.spinel_al = 0.0
        self.halite = 0.0


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


def read_csv_files4(directory):
    data = {}
    last_added_pool = CompPool()
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            with open(file_path, "r") as file:
                reader = csv.DictReader(file)
                for row in reader:
                    concentrations = {'Cr': float(row['Cr']), 'Al': float(row['Al']), 'O': float(row['O'])}
                    comp_pool = CompPool()

                    corundum1_cont = float(row.get('Corundum1_Content', 0).strip() or 0)
                    corundum2_cont = float(row.get('Corundum2_Content', 0).strip() or 0)
                    spinel1_cont = float(row.get('Spinel1_Content', 0).strip() or 0)
                    spinel2_cont = float(row.get('Spinel2_Content', 0).strip() or 0)
                    halite_cont = float(row.get('Halite_Content', 0).strip() or 0)

                    if halite_cont >0:
                        print()

                    sum_all = corundum1_cont + corundum2_cont + spinel1_cont + spinel2_cont + halite_cont
                    if sum_all > 0:
                        if corundum1_cont > 0:
                            corundum1_cr_cont = float(row.get('Corundum1_Cr_Cont', 0).strip() or 0)
                            corundum1_al_cont = float(row.get('Corundum1_Al_Cont', 0).strip() or 0)
                            if corundum1_cr_cont > corundum1_al_cont:
                                comp_pool.corundum_cr += corundum1_cont
                            else:
                                comp_pool.corundum_al += corundum1_cont

                        if corundum2_cont > 0:
                            corundum2_cr_cont = float(row.get('Corundum2_Cr_Cont', 0).strip() or 0)
                            corundum2_al_cont = float(row.get('Corundum2_Al_Cont', 0).strip() or 0)
                            if corundum2_cr_cont > corundum2_al_cont:
                                comp_pool.corundum_cr += corundum2_cont
                            else:
                                comp_pool.corundum_al += corundum2_cont

                        if spinel1_cont > 0:
                            spinel1_cr_cont = float(row.get('Spinel1_Cr_Cont', 0).strip() or 0)
                            spinel1_al_cont = float(row.get('Spinel1_Al_Cont', 0).strip() or 0)
                            if spinel1_cr_cont > spinel1_al_cont:
                                comp_pool.spinel_cr += spinel1_cont
                            else:
                                comp_pool.spinel_al += spinel1_cont

                        spinel2_cont = float(row.get('Spinel2_Content', 0).strip() or 0)
                        if spinel2_cont > 0:
                            spinel2_cr_cont = float(row.get('Spinel2_Cr_Cont', 0).strip() or 0)
                            spinel2_al_cont = float(row.get('Spinel2_Al_Cont', 0).strip() or 0)
                            if spinel2_cr_cont > spinel2_al_cont:
                                comp_pool.spinel_cr += spinel2_cont
                            else:
                                comp_pool.spinel_al += spinel2_cont

                        comp_pool.halite = halite_cont
                        # comp_pool.sum = comp_pool.corundum_cr + comp_pool.corundum_al + comp_pool.spinel_cr + comp_pool.spinel_al + comp_pool.halite
                        data[tuple(concentrations.values())] = comp_pool

                        last_added_pool.corundum_cr = comp_pool.corundum_cr
                        last_added_pool.corundum_al = comp_pool.corundum_al
                        last_added_pool.spinel_cr = comp_pool.spinel_cr
                        last_added_pool.spinel_al = comp_pool.spinel_al
                        last_added_pool.halite = comp_pool.halite
                        # last_added_pool.sum = comp_pool.sum
                    else:
                        data[tuple(concentrations.values())] = last_added_pool

    return data


def read_csv_files5(directory):
    data = {}
    last_added_pool = CompPool()
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            with open(file_path, "r") as file:
                reader = csv.DictReader(file)
                for row in reader:
                    concentrations = {'Cr': float(row['Cr']), 'Al': float(row['Al']), 'O': float(row['O'])}
                    comp_pool = CompPool()

                    corundum1_cont = float(row.get('Corundum1_Content', 0).strip() or 0)
                    corundum1_cr_cont = float(row.get('Corundum1_Cr_Cont', 0).strip() or 0)
                    corundum1_al_cont = float(row.get('Corundum1_Al_Cont', 0).strip() or 0)

                    corundum2_cont = float(row.get('Corundum2_Content', 0).strip() or 0)
                    corundum2_cr_cont = float(row.get('Corundum2_Cr_Cont', 0).strip() or 0)
                    corundum2_al_cont = float(row.get('Corundum2_Al_Cont', 0).strip() or 0)

                    spinel1_cont = float(row.get('Spinel1_Content', 0).strip() or 0)
                    spinel1_cr_cont = float(row.get('Spinel1_Cr_Cont', 0).strip() or 0)
                    spinel1_al_cont = float(row.get('Spinel1_Al_Cont', 0).strip() or 0)

                    spinel2_cont = float(row.get('Spinel2_Content', 0).strip() or 0)
                    spinel2_cr_cont = float(row.get('Spinel2_Cr_Cont', 0).strip() or 0)
                    spinel2_al_cont = float(row.get('Spinel2_Al_Cont', 0).strip() or 0)

                    halite_cont = float(row.get('Halite_Content', 0).strip() or 0)

                    sum_all = corundum1_cont + corundum2_cont + spinel1_cont + spinel2_cont + halite_cont
                    if sum_all > 0:
                        comp_pool.corundum_cr = (corundum1_cr_cont / 0.4) * corundum1_cont + (
                                    corundum2_cr_cont / 0.4) * corundum2_cont
                        comp_pool.corundum_al = (corundum1_al_cont / 0.4) * corundum1_cont + (
                                    corundum2_al_cont / 0.4) * corundum2_cont

                        # check = comp_pool.corundum_cr + comp_pool.corundum_al
                        # check2 = corundum1_cont + corundum2_cont

                        comp_pool.spinel_cr = (spinel1_cr_cont / (2 / 7)) * spinel1_cont + (
                                    spinel2_cr_cont / (2 / 7)) * spinel2_cont
                        comp_pool.spinel_al = (spinel1_al_cont / (2 / 7)) * spinel1_cont + (
                                    spinel2_al_cont / (2 / 7)) * spinel2_cont

                        # check = comp_pool.spinel_cr + comp_pool.spinel_al
                        # check2 = spinel1_cont + spinel2_cont

                        # if corundum1_cont > 0:
                        #     corundum1_cr_cont = float(row.get('Corundum1_Cr_Cont', 0).strip() or 0)
                        #     corundum1_al_cont = float(row.get('Corundum1_Al_Cont', 0).strip() or 0)
                        #     if corundum1_cr_cont > corundum1_al_cont:
                        #         comp_pool.corundum_cr += corundum1_cont
                        #     else:
                        #         comp_pool.corundum_al += corundum1_cont
                        #
                        # if corundum2_cont > 0:
                        #     corundum2_cr_cont = float(row.get('Corundum2_Cr_Cont', 0).strip() or 0)
                        #     corundum2_al_cont = float(row.get('Corundum2_Al_Cont', 0).strip() or 0)
                        #     if corundum2_cr_cont > corundum2_al_cont:
                        #         comp_pool.corundum_cr += corundum2_cont
                        #     else:
                        #         comp_pool.corundum_al += corundum2_cont
                        #
                        # if spinel1_cont > 0:
                        #     spinel1_cr_cont = float(row.get('Spinel1_Cr_Cont', 0).strip() or 0)
                        #     spinel1_al_cont = float(row.get('Spinel1_Al_Cont', 0).strip() or 0)
                        #     if spinel1_cr_cont > spinel1_al_cont:
                        #         comp_pool.spinel_cr += spinel1_cont
                        #     else:
                        #         comp_pool.spinel_al += spinel1_cont
                        #
                        # spinel2_cont = float(row.get('Spinel2_Content', 0).strip() or 0)
                        # if spinel2_cont > 0:
                        #     spinel2_cr_cont = float(row.get('Spinel2_Cr_Cont', 0).strip() or 0)
                        #     spinel2_al_cont = float(row.get('Spinel2_Al_Cont', 0).strip() or 0)
                        #     if spinel2_cr_cont > spinel2_al_cont:
                        #         comp_pool.spinel_cr += spinel2_cont
                        #     else:
                        #         comp_pool.spinel_al += spinel2_cont

                        comp_pool.halite = halite_cont
                        data[tuple(concentrations.values())] = comp_pool

                        last_added_pool.corundum_cr = comp_pool.corundum_cr
                        last_added_pool.corundum_al = comp_pool.corundum_al
                        last_added_pool.spinel_cr = comp_pool.spinel_cr
                        last_added_pool.spinel_al = comp_pool.spinel_al
                        last_added_pool.halite = comp_pool.halite
                    else:
                        data[tuple(concentrations.values())] = last_added_pool

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
    # directory = "D:/PhD/TC/Simulations_Klaus_first_ALL/"
    directory = "W:/SIMCA/TC/Simulations_Klaus_first_ALL/"
    # output_file = "C:/CA_3D_MP/thermodynamics/TD_look_up.pkl"
    output_file = "C:/Users/adam-wrmjvo101twvweh/PycharmProjects/CA_3D_MP/thermodynamics/TD_look_up.pkl"
    # p_output_file = "TD_look_up.pkl"

    # Read data from CSV files
    # data = read_csv_files5(directory)
    # Write data to a single file
    # write_data_to_file(data, output_file)
    # Load data from the consolidated file
    consolidated_data = load_data_from_file(output_file)
    keys = list(consolidated_data.keys())
    tree = KDTree(keys)

    o_c = np.arange(60)

    for o in o_c:

        # cr_c = random.uniform(0, 40)
        # al_c = random.uniform(0, 40)
        # o_c = random.uniform(0, 60)

        cr_c = 2
        al_c = 8
        # o_c = random.uniform(0, 60)

        target_value = (cr_c, al_c, o)
        print(target_value)

        closest_key = find_closest_key(target_value, tree, keys)
        print(closest_key)

        value_from_dict = consolidated_data[closest_key]

        print("Cr_oxide: ", value_from_dict.corundum_cr)
        print("Al_oxide: ", value_from_dict.corundum_al)
        print("Cr_spinel: ", value_from_dict.spinel_cr)
        print("Al_spinel: ", value_from_dict.spinel_al)

