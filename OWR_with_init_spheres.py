from engine import *
import traceback
from configuration import Config, save_script_contents_as_string
import itertools
import random

if __name__ == '__main__':
    save_script_contents_as_string(__file__, Config)
    """
    Nucleation and dissolution throughout the whole simulation (both schemes applied). Also with kinetic coeeficient!!!
    Go along the kinetic growth line! Check the kinetic file as well!!!

    CHANGED THE SCHEMES OF NUCLEATION AND DISSOLUTION ->>> NOW ALSO THE PARTIAL NEIGHBOURS ARE CONSIDERED!!!!
    """

    eng = CellularAutomata()

    eng.primary_oxidant.diffuse = eng.primary_oxidant.diffuse_with_scale
    eng.primary_active.diffuse = eng.primary_active.diffuse_with_scale

    eng.precip_func = eng.precipitation_first_case
    eng.get_combi_ind = eng.get_combi_ind_standard
    eng.precip_step = eng.precip_step_standard
    eng.check_intersection = eng.ci_single

    eng.decomposition = eng.dissolution_test
    eng.decomposition_intrinsic = eng.dissolution_zhou_wei_with_bsf_aip_UPGRADE_BOOL

    eng.cur_case = eng.cases.first
    eng.cases.first.go_around_func_ref = eng.go_around_mult_oxid_n_also_partial_neigh_aip

    eng.cur_case.nucleation_probabilities = utils.NucleationProbabilities(Config.PROBABILITIES.PRIMARY,
                                                                          Config.PRODUCTS.PRIMARY)
    eng.cur_case.dissolution_probabilities = utils.DissolutionProbabilities(Config.PROBABILITIES.PRIMARY,
                                                                            Config.PRODUCTS.PRIMARY)

    # set spheres
    number_of_spheres = 25
    max_rad = 5
    depth = 20

    eng.primary_active.transform_to_3d(Config.N_CELLS_PER_AXIS)
    eng.primary_oxidant.transform_to_3d()

    # some = list(np.arange(8, 58, 12))
    # grid_points = np.array(list(itertools.product(some, repeat=2)))

    for _ in range(number_of_spheres):
        rad = random.randint(1, max_rad)

        x_pos = random.randint(0, depth)
        y_pos = random.randint(0, Config.N_CELLS_PER_AXIS - 1)
        z_pos = random.randint(0, Config.N_CELLS_PER_AXIS - 1)

    # for x_pos in range(6, 30, 8):
    #     for centers in grid_points:
    #
    #         middle = centers

        x_minus = x_pos - rad
        x_plus = x_pos + rad

        y_minus = y_pos - rad
        y_plus = y_pos + rad

        z_minus = z_pos - rad
        z_plus = z_pos + rad

        for x in range(x_minus, x_plus):
            for y in range(y_minus, y_plus):
                for z in range(z_minus, z_plus):
                    if 0 <= x < Config.N_CELLS_PER_AXIS and 0 <= y < Config.N_CELLS_PER_AXIS and 0 <= z < Config.N_CELLS_PER_AXIS:
                        if (x - x_pos) ** 2 + (y - y_pos) ** 2 + (z - z_pos) ** 2 <= rad ** 2:
                            eng.primary_product.c3d[z, y, x] = eng.primary_oxid_numb
                            eng.primary_product.full_c3d[z, y, x] = True
                            eng.product_x_nzs[x] = True
                            eng.primary_oxidant.c3d[z, y, x] = 0
                            eng.primary_active.c3d[z, y, x] = 0

    eng.primary_active.c3d[:, :, :depth] = 0

    eng.primary_oxidant.transform_to_descards()
    eng.primary_active.transform_to_descards()

    try:
        eng.simulation()
    finally:
        try:
            if not Config.SAVE_WHOLE:
                eng.save_results()

        except (Exception,):
            eng.save_results()
            print("Not SAVED!")
        iterations = np.arange(eng.cumul_prod.last_in_buffer) * eng.precipitation_stride
        data = np.column_stack(
            (iterations, eng.cumul_prod.get_buffer(), eng.growth_rate.get_buffer()))
        output_file_path = "C:/test_runs_data/" + Config.GENERATED_VALUES.DB_ID + "_kinetics.txt"
        with open(output_file_path, "w") as f:
            for row in data:
                f.write(" ".join(map(str, row)) + "\n")

        eng.insert_last_it()
        eng.utils.db.conn.commit()
        print()
        print("____________________________________________________________")
        print("Simulation was closed at Iteration: ", eng.iteration)
        print("____________________________________________________________")
        print()
        traceback.print_exc()
