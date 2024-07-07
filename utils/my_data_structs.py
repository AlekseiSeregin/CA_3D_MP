import numpy as np


class MyBufferCoords:
    def __init__(self, reserve, dtype=np.short):
        self.buffer = np.zeros((3, reserve), dtype=dtype)
        self.last_in_buffer = 0

    def append_to_buffer(self, data_array):
        self.buffer[:, self.last_in_buffer:self.last_in_buffer + data_array.shape[1]] = data_array
        self.last_in_buffer += data_array.shape[1]

    def copy_to_buffer(self, data_array):
        self.reset_buffer()
        self.append_to_buffer(data_array)

    def get_elem_at_ind(self, indexes):
        return self.buffer[:, indexes]

    def get_elem_instead_ind(self, indexes):
        return np.delete(self.buffer[:, :self.last_in_buffer], indexes, axis=1)

    def update_buffer_at_axis(self, new_data, axis=2):
        self.buffer[axis, :self.last_in_buffer] = new_data

    def get_buffer(self):
        return self.buffer[:, :self.last_in_buffer]

    def reset_buffer(self):
        self.last_in_buffer = 0


class MyBufferSingle:
    def __init__(self, shape, dtype=np.uint32):
        self.buffer = np.zeros(shape, dtype=dtype)
        self.last_in_buffer = 0

    def copy_to_buffer(self, data_array):
        self.buffer[:data_array.shape[1]] = data_array
        self.last_in_buffer = data_array.shape[1]

    def update_buffer(self, new_data):
        self.buffer[:new_data.shape] = new_data
        self.last_in_buffer = new_data.shape

    def set_at_ind(self, pos, val):
        self.buffer[pos[0], pos[1]] = val
        self.last_in_buffer = pos[1][0] + 1

    def append(self, value):
        self.buffer[self.last_in_buffer] = value
        self.last_in_buffer += 1

    def fill_buffer(self, furthest_index, array_3d):
        data = np.array(np.sum(array_3d[:, :, np.arange(furthest_index + 1)]), dtype=np.uint32)
        self.update_buffer(data)

    def get_buffer(self):
        return self.buffer[:, :self.last_in_buffer]

    def reset_buffer(self):
        self.last_in_buffer = 0

    def use_whole_buffer(self):
        self.last_in_buffer = self.buffer.size - 1
