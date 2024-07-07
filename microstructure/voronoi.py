import pyvoro
import matplotlib.pyplot as plt
from .bresenham import *
# from bresenham import *
import numpy as np
import time
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from .own_seeds import *
# from own_seeds import *


class VoronoiMicrostructure:
    def __init__(self, n_cells_per_axis):
        self.lines = np.array([[[0,0,0], [0,0,0]]])
        self.lines_faces = np.array([[[0,0,0], [0,0,0]]])
        self.n_cells_per_axis = n_cells_per_axis
        self.shape = (self.n_cells_per_axis, self.n_cells_per_axis, self.n_cells_per_axis)
        self.grain_boundaries = np.full(self.shape, False, dtype=bool)
        self.divisor = None
        self.diff_jump = 0.1
        self.jump_size = 1
        self.jump_directions = None
        self.ca_edges = None
        self.ca_faces = None

    def generate_voronoi_3d(self, number_of_grains, periodic=False, seeds=None):
        self.divisor = int(self.n_cells_per_axis / 0.5)
        if seeds is None:
            seeds = np.random.random_sample((number_of_grains, 3))

            for seed in seeds:
                print(seed)

        # regular arrangement of seeds for some tests
        elif seeds == 'regular8':
            seeds = [[(1 / 4), (1 / 4), (1 / 4)],
                     [(1 / 4), (1 / 4) * 3, (1 / 4)],
                     [(1 / 4), (1 / 4) * 3, (1 / 4) * 3],
                     [(1 / 4), (1 / 4), (1 / 4) * 3],
                     [(1 / 4) * 3, (1 / 4), (1 / 4)],
                     [(1 / 4) * 3, (1 / 4) * 3, (1 / 4)],
                     [(1 / 4) * 3, (1 / 4) * 3, (1 / 4) * 3],
                     [(1 / 4) * 3, (1 / 4), (1 / 4) * 3]]

        elif seeds == 'standard':
            seeds = np.array([[0.71363424, 0.18968331, 0.22064598],
                              [0.06832179, 0.28305906, 0.10689959],
                              [0.81873141, 0.07915909, 0.28691126],
                              [0.90147317, 0.02136836, 0.87818224],
                              [0.29456042, 0.29805038, 0.2433283 ]])

        elif seeds == 'demo':
            seeds = np.array([[0.99073393, 0.04600333, 0.30824473],
                              [0.91127096, 0.96590519, 0.09132393],
                              [0.78442506, 0.70838989, 0.31115133],
                              [0.17121934, 0.65835036, 0.09817317],
                              [0.86900279, 0.63286860, 0.99611049],
                              [0.10538952, 0.91241952, 0.26677793]])
        elif seeds == 'own':
            seeds = np.array(G_100)

        elif seeds == "plane":
            seeds = np.array([[0.1, 0.5, 0.5],
                              [0.9, 0.5, 0.5]])

        elif seeds == "mult_planes":
            seeds = np.array([[0.000001, 0.5, 0.5],
                              [0.25, 0.5, 0.5],
                              [0.5, 0.5, 0.5],
                              [0.75, 0.5, 0.5],
                              [0.99999, 0.5, 0.5]])

        vor = pyvoro.compute_voronoi(seeds, [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]], 10,
                                     periodic=[periodic, periodic, periodic])

        # list of indexes of already checked cells
        ind_done_cells = []
        for cell_ind, cell in enumerate(vor):
            for face in cell['faces']:
                # check if face of adjacent cell is already checked
                if face['adjacent_cell'] not in ind_done_cells and face['adjacent_cell'] >= 0:
                # if face['adjacent_cell'] >= 0:
                    # list of coordinates
                    face_vertices_coordinates = np.array([cell['vertices'][index] for index in face['vertices']])
                    if len(face_vertices_coordinates) == 3:
                        self.three(face_vertices_coordinates)
                    elif len(face_vertices_coordinates) == 4:
                        self.four(face_vertices_coordinates)
                    elif len(face_vertices_coordinates) == 5:
                        self.five(face_vertices_coordinates)
                    elif len(face_vertices_coordinates) == 6:
                        self.six(face_vertices_coordinates)
                    elif len(face_vertices_coordinates) == 7:
                        self.seven(face_vertices_coordinates)
                    elif len(face_vertices_coordinates) == 8:
                        self.eight(face_vertices_coordinates)
                    elif len(face_vertices_coordinates) == 9:
                        self.nine(face_vertices_coordinates)
                    elif len(face_vertices_coordinates) == 10:
                        self.ten(face_vertices_coordinates)
                    else:
                        print(f"Write Function for {len(face_vertices_coordinates)} vertices ")
                        self.all_forms(face_vertices_coordinates)
            ind_done_cells.append(cell_ind)

        starts_edges = np.array(np.array(self.lines)[:, 0] * (self.n_cells_per_axis - 1), dtype=np.short)
        ends_edges = np.array(np.array(self.lines)[:, 1] * (self.n_cells_per_axis - 1), dtype=np.short)
        res_edges = np.array([bresenhamlines(np.array([start]), np.array([finish]), -1)[0] for start, finish
                              in zip(starts_edges, ends_edges)], dtype=object)

        shift = self.n_cells_per_axis
        bresenham_s_edges = res_edges + shift
        self.jump_directions = np.zeros((self.n_cells_per_axis + 2 * shift, self.n_cells_per_axis + 2 * shift,
                                         self.n_cells_per_axis + 2 * shift, 3), dtype=np.short)

        for line in bresenham_s_edges:
            if len(line) > 0:
                direction = line[-1] - line[0]
                path_length = np.linalg.norm(direction)
                vector_step = self.jump_size * direction / path_length
                vector_step = np.array(np.rint(vector_step), dtype=np.short)

                rolled_line = np.roll(line, -1, axis=0)
                diffs = rolled_line - line
                diffs[-1] = vector_step

                for coordinate, jump_vec in zip(line, diffs):
                    self.jump_directions[coordinate[0], coordinate[1], coordinate[2]] = jump_vec

        bresenham_points_s_edges = np.array([point for array2 in bresenham_s_edges for point in array2],
                                            dtype=np.short).transpose()
        ca_shifted_edges = np.full((self.n_cells_per_axis + 2 * shift, self.n_cells_per_axis + 2 * shift,
                                     self.n_cells_per_axis + 2 * shift), False, dtype=bool)
        ca_shifted_edges[bresenham_points_s_edges[0], bresenham_points_s_edges[1], bresenham_points_s_edges[2]] = True
        ca_edges = ca_shifted_edges[shift:(self.n_cells_per_axis + shift),
                                    shift:(self.n_cells_per_axis + shift),
                                    shift:(self.n_cells_per_axis + shift)]

        # working with faces
        starts_faces = np.array(np.array(self.lines_faces)[:, 0] * (self.n_cells_per_axis - 1), dtype=np.short)
        ends_faces = np.array(np.array(self.lines_faces)[:, 1] * (self.n_cells_per_axis - 1), dtype=np.short)
        res_faces = np.array([bresenhamlines(np.array([start]), np.array([finish]), -1)[0] for start, finish
                              in zip(starts_faces, ends_faces)], dtype=object)
        shift = self.n_cells_per_axis
        bresenham_s_faces = res_faces + shift
        for line in bresenham_s_faces:
            if len(line) > 0:
                direction = line[-1] - line[0]
                path_length = np.linalg.norm(direction)
                vector_step = self.jump_size * direction / path_length
                vector_step = np.array(np.rint(vector_step), dtype=np.short)

                rolled_line = np.roll(line, -1, axis=0)
                diffs = rolled_line - line
                diffs[-1] = vector_step

                for coordinate, jump_vec in zip(line, diffs):
                    self.jump_directions[coordinate[0], coordinate[1], coordinate[2]] = jump_vec

        bresenham_points_s_faces = np.array([point for array2 in bresenham_s_faces for point in array2]).transpose()
        ca_shifted_faces = np.full((self.n_cells_per_axis + 2 * shift, self.n_cells_per_axis + 2 * shift,
                                    self.n_cells_per_axis + 2 * shift), False, dtype=bool)
        ca_shifted_faces[bresenham_points_s_faces[0], bresenham_points_s_faces[1], bresenham_points_s_faces[2]] = True

        ca_faces = ca_shifted_faces[shift:(self.n_cells_per_axis + shift),
                                    shift:(self.n_cells_per_axis + shift),
                                    shift:(self.n_cells_per_axis + shift)]
        self.jump_directions = self.jump_directions[shift:(self.n_cells_per_axis + shift),
                                    shift:(self.n_cells_per_axis + shift),
                                    shift:(self.n_cells_per_axis + shift)]

        # edges
        # removing plane Z = 0
        ca_edges[0, :, :] = 0
        # removing plane Z = max
        ca_edges[self.n_cells_per_axis - 1, :, :] = 0
        # removing plane X = 0
        ca_edges[:, 0, :] = 0
        # removing plane X = max
        ca_edges[:, self.n_cells_per_axis - 1, :] = 0
        # removing plane Y = 0
        ca_edges[:, :, 0] = 0
        # removing plane Y = max
        ca_edges[:, :, self.n_cells_per_axis - 1] = 0

        # faces
        # removing plane Z = 0
        ca_faces[0, :, :] = 0
        # removing plane Z = max
        ca_faces[self.n_cells_per_axis - 1, :, :] = 0
        # removing plane X = 0
        ca_faces[:, 0, :] = 0
        # removing plane X = max
        ca_faces[:, self.n_cells_per_axis - 1, :] = 0
        # removing plane Y = 0
        ca_faces[:, :, 0] = 0
        # removing plane Y = max
        ca_faces[:, :, self.n_cells_per_axis - 1] = 0

        # destinations
        # removing plane Z = 0
        self.jump_directions[0, :, :] = [0, 0, 0]
        # removing plane Z = max
        self.jump_directions[self.n_cells_per_axis - 1, :, :] = [0, 0, 0]
        # removing plane X = 0
        self.jump_directions[:, 0, :] = [0, 0, 0]
        # removing plane X = max
        self.jump_directions[:, self.n_cells_per_axis - 1, :] = [0, 0, 0]
        # removing plane Y = 0
        self.jump_directions[:, :, 0] = [0, 0, 0]
        # removing plane Y = max
        self.jump_directions[:, :, self.n_cells_per_axis - 1] = [0, 0, 0]

        self.ca_edges = np.nonzero(ca_edges)
        self.ca_faces = np.nonzero(ca_faces)

        self.grain_boundaries[self.ca_edges[0], self.ca_edges[1], self.ca_edges[2]] = True
        self.grain_boundaries[self.ca_faces[0], self.ca_faces[1], self.ca_faces[2]] = True

        # return np.array(ca_faces), np.array(ca_edges)
        # return np.nonzero(ca_faces), np.nonzero(ca_edges)

    @staticmethod
    def generate_voronoi_3d_continious(number_of_grains, seeds=None):
        if seeds is None:
            seeds = np.random.random_sample((number_of_grains, 3))
            for item in seeds:
                print(item)
        # regular arrangement of seeds for some tests
        elif seeds == 'regular8':
            seeds = [[(1 / 4), (1 / 4), (1 / 4)],
                     [(1 / 4), (1 / 4) * 3, (1 / 4)],
                     [(1 / 4), (1 / 4) * 3, (1 / 4) * 3],
                     [(1 / 4), (1 / 4), (1 / 4) * 3],
                     [(1 / 4) * 3, (1 / 4), (1 / 4)],
                     [(1 / 4) * 3, (1 / 4) * 3, (1 / 4)],
                     [(1 / 4) * 3, (1 / 4) * 3, (1 / 4) * 3],
                     [(1 / 4) * 3, (1 / 4), (1 / 4) * 3]]

        elif seeds == 'standard':
            seeds = np.array([[0.71363424, 0.18968331, 0.22064598],
                              [0.06832179, 0.28305906, 0.10689959],
                              [0.81873141, 0.07915909, 0.28691126],
                              [0.90147317, 0.02136836, 0.87818224],
                              [0.29456042, 0.29805038, 0.2433283 ]])

        elif seeds == 'demo':
            seeds = np.array([[0.99073393, 0.04600333, 0.30824473],
                              [0.91127096, 0.96590519, 0.09132393],
                              [0.78442506, 0.70838989, 0.31115133],
                              [0.17121934, 0.65835036, 0.09817317],
                              [0.86900279, 0.63286860, 0.99611049],
                              [0.10538952, 0.91241952, 0.26677793]])
        elif seeds == 'own':
            seeds = np.array(G_100)

        elif seeds == "plane":
            seeds = np.array([[0.1, 0.5, 0.5],
                              [0.9, 0.5, 0.5]])

        vor = pyvoro.compute_voronoi(seeds, [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]], 10,
                                     periodic=[False, False, False])

        starts = np.array([[0, 0, 0]])
        ends = np.array([[0, 0, 0]])
        colors = ['b', 'k', 'r', 'g']
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot each plane
        def plot_plane(ax, vertices, color, alpha):
            plane = Poly3DCollection([vertices], alpha=alpha, color=color)
            ax.add_collection3d(plane)

        ind_done_cells = []
        for cell_ind, cell in enumerate(vor):
            for face in cell['faces']:
                # check if face of adjacent cell is already checked
                if face['adjacent_cell'] not in ind_done_cells and face['adjacent_cell'] >= 0:
                    # list of coordinates
                    face_vertices_coordinates = np.array([cell['vertices'][index] for index in face['vertices']])

                    plot_plane(ax, face_vertices_coordinates, np.random.choice(colors), 0.5)

                    # starts = np.concatenate((starts, np.array(face_vertices_coordinates)), axis=0)
                    # ends = np.concatenate((ends, np.array(np.roll(face_vertices_coordinates, -1, axis=0))), axis=0)

        # Set limits and labels
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.show()

    def three(self, face_vert_coords):
        #  find the line closest to origin
        min_x = np.min(face_vert_coords[:, 2])
        min_ind = np.array(np.where(face_vert_coords[:, 2] == min_x)[0])
        min_ind = min_ind[0]
        base_line_start = face_vert_coords[min_ind]
        face_vert_coords = np.roll(face_vert_coords, -min_ind, axis=0)

        if face_vert_coords[-1, 2] < face_vert_coords[1, 2]:
            base_line_end = face_vert_coords[-1]
            middle_point = face_vert_coords[1]
        else:
            base_line_end = face_vert_coords[1]
            middle_point = face_vert_coords[-1]

        base_line_m = (base_line_end - base_line_start) / 2
        base_line_m = base_line_start + base_line_m

        #  filling two areas
        first_area = self.fill_plane(base_line_start, base_line_start, base_line_m, middle_point)
        second_area = self.fill_plane(base_line_m, middle_point, base_line_end, base_line_end)
        self.lines_faces = np.append(self.lines_faces, first_area, axis=0)
        self.lines_faces = np.append(self.lines_faces, second_area, axis=0)

        #  base line vect
        self.lines = np.append(self.lines, [[base_line_start, base_line_end]], axis=0)
        #  first line
        self.lines = np.append(self.lines, [[base_line_start, middle_point]], axis=0)
        #  second line
        self.lines = np.append(self.lines, [[base_line_end, middle_point]], axis=0)

    def four(self, face_vert_coords):
        #  find the line closest to origin
        min_x = np.min(face_vert_coords[:, 2])
        min_ind = np.array(np.where(face_vert_coords[:, 2] == min_x)[0])
        min_ind = min_ind[0]
        base_line_start = face_vert_coords[min_ind]
        face_vert_coords = np.roll(face_vert_coords, -min_ind, axis=0)

        if face_vert_coords[-1, 2] < face_vert_coords[1, 2]:
            base_line_end = face_vert_coords[-1]
            pair_line_start = face_vert_coords[1]
            pair_line_end = face_vert_coords[2]
        else:
            base_line_end = face_vert_coords[1]
            pair_line_start = face_vert_coords[3]
            pair_line_end = face_vert_coords[2]

        #  filling  area
        area = self.fill_plane(base_line_start, pair_line_start, base_line_end, pair_line_end)
        self.lines_faces = np.append(self.lines_faces, area, axis=0)
        #  baseline vect
        self.lines = np.append(self.lines, [[base_line_start, base_line_end]], axis=0)
        #  first line
        self.lines = np.append(self.lines, [[base_line_start, pair_line_start]], axis=0)
        #  second line
        self.lines = np.append(self.lines, [[base_line_end, pair_line_end]], axis=0)
        #  third line
        self.lines = np.append(self.lines, [[pair_line_start, pair_line_end]], axis=0)

    def five(self, face_vert_coords):
        order = [[0, 1], [0, 4], [4, 3], [1, 2], [2, 3]]
        #  find the line closest to origin
        min_x = np.min(face_vert_coords[:, 2])
        min_ind = np.array(np.where(face_vert_coords[:, 2] == min_x)[0])
        min_ind = min_ind[0]
        base_line_start = face_vert_coords[min_ind]
        face_vert_coords = np.roll(face_vert_coords, -min_ind, axis=0)

        if face_vert_coords[-1, 2] < face_vert_coords[1, 2]:
            base_line_end = face_vert_coords[-1]
            pair_line_start = face_vert_coords[2]
            pair_line_end = face_vert_coords[3]
            tri_line_b_e = face_vert_coords[1]
            tri_line_p_s = face_vert_coords[2]
        else:
            base_line_end = face_vert_coords[1]
            pair_line_start = face_vert_coords[3]
            pair_line_end = face_vert_coords[2]
            tri_line_b_e = face_vert_coords[-1]
            tri_line_p_s = face_vert_coords[3]

        #  filling  area
        area = self.fill_plane(base_line_start, pair_line_start, base_line_end, pair_line_end)
        self.lines_faces = np.append(self.lines_faces, area, axis=0)
        #  filling  triangle
        triangle = self.fill_plane(base_line_start, tri_line_p_s, tri_line_b_e, tri_line_b_e)
        self.lines_faces = np.append(self.lines_faces, triangle, axis=0)
        edges_vectors = np.array([[face_vert_coords[pair[0]], face_vert_coords[pair[1]]] for pair in order])
        self.lines = np.append(self.lines, edges_vectors, axis=0)

    def six(self, face_vert_coords):
        order = [[0, 1], [0, 5], [5, 4], [4, 3], [1, 2], [2, 3]]
        #  find the line closest to origin
        min_x = np.min(face_vert_coords[:, 2])
        min_ind = np.array(np.where(face_vert_coords[:, 2] == min_x)[0])
        min_ind = min_ind[0]
        base_line_start = face_vert_coords[min_ind]
        face_vert_coords = np.roll(face_vert_coords, -min_ind, axis=0)

        if face_vert_coords[-1, 2] < face_vert_coords[1, 2]:
            base_line_end = face_vert_coords[-1]
            pair_line_start = face_vert_coords[2]
            pair_line_end = face_vert_coords[3]
            first_tri_p = face_vert_coords[1]
            sec_tri_p = face_vert_coords[4]
            first_tri_e = face_vert_coords[2]
            sec_tri_e = face_vert_coords[3]

        else:
            base_line_end = face_vert_coords[1]
            pair_line_start = face_vert_coords[4]
            pair_line_end = face_vert_coords[3]
            first_tri_p = face_vert_coords[5]
            sec_tri_p = face_vert_coords[2]
            first_tri_e = face_vert_coords[4]
            sec_tri_e = face_vert_coords[3]

        #  filling  area
        area = self.fill_plane(base_line_start, pair_line_start, base_line_end, pair_line_end)
        self.lines_faces = np.append(self.lines_faces, area, axis=0)
        #  filling first triangle
        triangle = self.fill_plane(base_line_start, first_tri_e, first_tri_p, first_tri_p)
        self.lines_faces = np.append(self.lines_faces, triangle, axis=0)
        #  filling second triangle
        triangle = self.fill_plane(base_line_end, sec_tri_e, sec_tri_p, sec_tri_p)
        self.lines_faces = np.append(self.lines_faces, triangle, axis=0)
        edges_vectors = np.array([[face_vert_coords[pair[0]], face_vert_coords[pair[1]]] for pair in order])
        self.lines = np.append(self.lines, edges_vectors, axis=0)

    def seven(self, face_vert_coords):
        #  find the line closest to origin
        min_x = np.min(face_vert_coords[:, 2])
        min_ind = np.array(np.where(face_vert_coords[:, 2] == min_x)[0])
        min_ind = min_ind[0]
        face_vert_coords = np.roll(face_vert_coords, -min_ind, axis=0)

        if face_vert_coords[-1, 2] < face_vert_coords[1, 2]:
            self.four(face_vert_coords[[0, 1, 2, 3], :])
            self.four(face_vert_coords[[0, 3, 4, 6], :])
            self.three(face_vert_coords[[6, 4, 5], :])
        else:
            self.four(face_vert_coords[[0, 1, 3, 4], :])
            self.four(face_vert_coords[[0, 4, 5, 6], :])
            self.three(face_vert_coords[[1, 2, 3], :])

    def eight(self, face_vert_coords):
        #  find the line closest to origin
        min_x = np.min(face_vert_coords[:, 2])
        min_ind = np.array(np.where(face_vert_coords[:, 2] == min_x)[0])
        min_ind = min_ind[0]

        face_vert_coords = np.roll(face_vert_coords, -min_ind, axis=0)

        if face_vert_coords[-1, 2] < face_vert_coords[1, 2]:
            self.four(face_vert_coords[[0, 1, 2, 3], :])
            self.four(face_vert_coords[[0, 3, 4, 7], :])
            self.four(face_vert_coords[[7, 4, 5, 6], :])
        else:
            self.four(face_vert_coords[[1, 2, 3, 4], :])
            self.four(face_vert_coords[[1, 4, 5, 0], :])
            self.four(face_vert_coords[[0, 5, 6, 7], :])

    def nine(self, face_vert_coords):
        #  find the line closest to origin
        min_x = np.min(face_vert_coords[:, 2])
        min_ind = np.array(np.where(face_vert_coords[:, 2] == min_x)[0])
        min_ind = min_ind[0]
        face_vert_coords = np.roll(face_vert_coords, -min_ind, axis=0)

        self.four(face_vert_coords[[1, 2, 3, 4], :])
        self.four(face_vert_coords[[1, 4, 5, 0], :])
        self.four(face_vert_coords[[0, 5, 6, 8], :])
        self.three(face_vert_coords[[8, 6, 7], :])

    def ten(self, face_vert_coords):
        #  find the line closest to origin
        min_x = np.min(face_vert_coords[:, 2])
        min_ind = np.array(np.where(face_vert_coords[:, 2] == min_x)[0])
        min_ind = min_ind[0]
        face_vert_coords = np.roll(face_vert_coords, -min_ind, axis=0)

        self.four(face_vert_coords[[0, 1, 3, 4], :])
        self.four(face_vert_coords[[0, 4, 5, 9], :])
        self.four(face_vert_coords[[9, 5, 6, 8], :])
        self.three(face_vert_coords[[1, 2, 3], :])
        self.three(face_vert_coords[[8, 6, 7], :])

    def all_forms(self, face_vert_coords):
        min_x = np.min(face_vert_coords[:, 2])
        min_ind = np.array(np.where(face_vert_coords[:, 2] == min_x)[0])
        min_ind = min_ind[0]
        face_vertices_coordinates = np.roll(face_vert_coords, -min_ind, axis=0)

        zero_s = list(np.around(face_vertices_coordinates[0], decimals=3))
        # adding face's edges and some diagonals to divide it on triangles
        for i, coordinate in enumerate(face_vertices_coordinates):
            if i > 0:
                # edg
                current = list(np.around(coordinate, decimals=5))
                previous = list(np.around(face_vertices_coordinates[i - 1], decimals=3))
                self.lines = np.append(self.lines, [[current, previous]], axis=0)
                if i > 1:
                    # diagonal (or closing line in case of the last edge)
                    if i == len(face_vertices_coordinates) - 1:
                        self.lines = np.append(self.lines, [[zero_s, current]], axis=0)
                    else:
                        self.lines_faces = np.append(self.lines_faces, [[zero_s, current]], axis=0)
                    # filling triangle with lines
                    step = 1 / self.divisor
                    position = 0
                    path_start = np.array(current)
                    path_end = np.array(previous)
                    path_length = np.linalg.norm(path_end - path_start)
                    vector_step = (path_end - path_start) * step / path_length
                    while position < path_length:
                        path_start += vector_step
                        path_start_round = list(np.around(path_start, decimals=3))
                        position += step
                        self.lines_faces = np.append(self.lines_faces, [[zero_s, path_start_round]], axis=0)

    def fill_plane(self, start_line_a, start_line_b, end_line_a, end_line_b):
        starts = np.linspace(start_line_a, end_line_a, self.divisor)
        ends = np.linspace(start_line_b, end_line_b, self.divisor)
        new_lines_faces = np.concatenate((starts, ends), 1)
        new_lines_faces = np.reshape(new_lines_faces, (len(new_lines_faces), 2, 3))
        return new_lines_faces

    def divide_vectors(self):
        divided_lines = np.array([[[0, 0, 0], [0, 0, 0]]])
        for vector in self.lines:
            vector_length = np.linalg.norm(vector[1] - vector[0])
            n_of_steps = int(vector_length / self.diff_jump)
            if n_of_steps > 1:
                new_vectors = np.linspace(vector[0], vector[1], n_of_steps)
                new_vector_sec = np.array([[start_point, new_vectors[index+1]] for index, start_point in
                                           enumerate(new_vectors) if index < len(new_vectors) - 1])
                divided_lines = np.append(divided_lines, new_vector_sec, axis=0)
        self.lines = divided_lines

        divided_lines = np.array([[[0, 0, 0], [0, 0, 0]]])
        for vector in self.lines_faces:
            vector_length = np.linalg.norm(vector[1] - vector[0])
            n_of_steps = int(vector_length / self.diff_jump)
            if n_of_steps > 1:
                new_vectors = np.linspace(vector[0], vector[1], n_of_steps)
                new_vector_sec = np.array([[start_point, new_vectors[index+1]] for index, start_point in
                                           enumerate(new_vectors) if index < len(new_vectors) - 1])
                divided_lines = np.append(divided_lines, new_vector_sec, axis=0)
        self.lines_faces = divided_lines

    def show_microstructure(self, n_cells_per_axis):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.ca_faces[2], self.ca_faces[1], self.ca_faces[0], color='darkgoldenrod', marker=',', s=1)
        ax.scatter(self.ca_edges[2], self.ca_edges[1], self.ca_edges[0], color='b', marker=',', s=5)
        ax.scatter(0, n_cells_per_axis/2, n_cells_per_axis/2, color='r', marker=',', s=50)

        ax.set_xlim3d(0, n_cells_per_axis)
        ax.set_ylim3d(0, n_cells_per_axis)
        ax.set_zlim3d(0, n_cells_per_axis)

        plt.show()
        # plt.close()

        # # _____Plot slices_______
        # div_step = int(n_cells_per_axis / 9)
        #
        # cut1 = np.nonzero(self.grain_boundaries[div_step, :, :])
        # x1 = cut1[0]
        # y1 = cut1[1]
        #
        # cut2 = np.nonzero(self.grain_boundaries[div_step * 2, :, :])
        # x2 = cut2[0]
        # y2 = cut2[1]
        #
        # cut3 = np.nonzero(self.grain_boundaries[div_step * 3, :, :])
        # x3 = cut3[0]
        # y3 = cut3[1]
        #
        # cut4 = np.nonzero(self.grain_boundaries[div_step * 4, :, :])
        # x4 = cut4[0]
        # y4 = cut4[1]
        #
        # cut5 = np.nonzero(self.grain_boundaries[div_step * 5, :, :])
        # x5 = cut5[0]
        # y5 = cut5[1]
        #
        # cut6 = np.nonzero(self.grain_boundaries[div_step * 6, :, :])
        # x6 = cut6[0]
        # y6 = cut6[1]
        #
        # cut7 = np.nonzero(self.grain_boundaries[div_step * 7, :, :])
        # x7 = cut7[0]
        # y7 = cut7[1]
        #
        # cut8 = np.nonzero(self.grain_boundaries[div_step * 8, :, :])
        # x8 = cut8[0]
        # y8 = cut8[1]
        #
        # cut9 = np.nonzero(self.grain_boundaries[div_step * 9, :, :])
        # x9 = cut9[0]
        # y9 = cut9[1]
        #
        # fig = plt.figure()
        # ax1 = fig.add_subplot(331)
        # ax2 = fig.add_subplot(332)
        # ax3 = fig.add_subplot(333)
        # ax4 = fig.add_subplot(334)
        # ax5 = fig.add_subplot(335)
        # ax6 = fig.add_subplot(336)
        # ax7 = fig.add_subplot(337)
        # ax8 = fig.add_subplot(338)
        # ax9 = fig.add_subplot(339)
        #
        # ax1.scatter(x1, y1, color='b', marker='s', s=10)
        # ax2.scatter(x2, y2, color='b', marker='s', s=10)
        # ax3.scatter(x3, y3, color='b', marker='s', s=10)
        # ax4.scatter(x4, y4, color='b', marker='s', s=10)
        # ax5.scatter(x5, y5, color='b', marker='s', s=10)
        # ax6.scatter(x6, y6, color='b', marker='s', s=10)
        # ax7.scatter(x7, y7, color='b', marker='s', s=10)
        # ax8.scatter(x8, y8, color='b', marker='s', s=10)
        # ax9.scatter(x9, y9, color='b', marker='s', s=10)
        #
        # plt.show()


# Some tests
if __name__ == "__main__":
    size = 300
    cells_size = 1
    edge_size = 5
    # import bresenham
    # _______Plot 3D______
    begin = time.time()
    micro = VoronoiMicrostructure(size)
    micro.generate_voronoi_3d_continious(50, seeds="own")
    # micro.show_microstructure(size)
    # micro.generate_voronoi_3d_continious(size, seeds="own")
    # cells_faces_3d = np.array(np.nonzero(cells_faces))
    # cells_edges_3d = np.array(np.nonzero(cells_edges))
    # x_edge = cells_edges_3d[2]
    # y_edge = cells_edges_3d[1]
    # z_edge = cells_edges_3d[0]
    # x_face = cells_faces_3d[2]
    # y_face = cells_faces_3d[1]
    # z_face = cells_faces_3d[0]
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(x_face, y_face, z_face, color='darkgoldenrod', marker=',', s=cells_size)
    # ax.scatter(x_edge, y_edge, z_edge, color='b', marker=',', s=edge_size)
    #
    # ax.set_xlim3d(0, size)
    # ax.set_ylim3d(0, size)
    # ax.set_zlim3d(0, size)
    #
    # # ax.scatter(0, 50, 50, color='r', marker=',', s=50)
    #
    # end = time.time()
    # elapsed_time = (end - begin)
    # print(f'elapsed time: {round(elapsed_time, 2)} s')
    # plt.show()
    # plt.close()
    #
    # # _____Plot slices_______
    # # begin = time.time()
    # # micro = VoronoiMicrostructure()
    # # cells_faces, cells_edges = micro.generate_voronoi_3d(size, 20)
    # # cells_faces += cells_edges
    # div_step = int(size / 9)
    #
    # cut1 = np.nonzero(cells_faces[div_step, :, :])
    # x1 = cut1[0]
    # y1 = cut1[1]
    #
    # cut2 = np.nonzero(cells_faces[div_step*2, :, :])
    # x2 = cut2[0]
    # y2 = cut2[1]
    #
    # cut3 = np.nonzero(cells_faces[div_step*3, :, :])
    # x3 = cut3[0]
    # y3 = cut3[1]
    #
    # cut4 = np.nonzero(cells_faces[div_step*4, :, :])
    # x4 = cut4[0]
    # y4 = cut4[1]
    #
    # cut5 = np.nonzero(cells_faces[div_step*5, :, :])
    # x5 = cut5[0]
    # y5 = cut5[1]
    #
    # cut6 = np.nonzero(cells_faces[div_step*6, :, :])
    # x6 = cut6[0]
    # y6 = cut6[1]
    #
    # cut7 = np.nonzero(cells_faces[div_step*7, :, :])
    # x7 = cut7[0]
    # y7 = cut7[1]
    #
    # cut8 = np.nonzero(cells_faces[div_step*8, :, :])
    # x8 = cut8[0]
    # y8 = cut8[1]
    #
    # cut9 = np.nonzero(cells_faces[div_step*9, :, :])
    # x9 = cut9[0]
    # y9 = cut9[1]
    #
    # fig = plt.figure()
    # ax1 = fig.add_subplot(331)
    # ax2 = fig.add_subplot(332)
    # ax3 = fig.add_subplot(333)
    # ax4 = fig.add_subplot(334)
    # ax5 = fig.add_subplot(335)
    # ax6 = fig.add_subplot(336)
    # ax7 = fig.add_subplot(337)
    # ax8 = fig.add_subplot(338)
    # ax9 = fig.add_subplot(339)
    #
    # ax1.scatter(x1, y1, color='b', marker='s', s=10)
    # ax2.scatter(x2, y2, color='b', marker='s', s=10)
    # ax3.scatter(x3, y3, color='b', marker='s', s=10)
    # ax4.scatter(x4, y4, color='b', marker='s', s=10)
    # ax5.scatter(x5, y5, color='b', marker='s', s=10)
    # ax6.scatter(x6, y6, color='b', marker='s', s=10)
    # ax7.scatter(x7, y7, color='b', marker='s', s=10)
    # ax8.scatter(x8, y8, color='b', marker='s', s=10)
    # ax9.scatter(x9, y9, color='b', marker='s', s=10)
    #
    # end = time.time()
    # elapsed_time = (end - begin)
    # print(f'Elapsed time: {round(elapsed_time, 2)} s')
    # plt.show()
