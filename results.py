from visualisation import *
import tkinter as tk
from tkinter import filedialog

# plot_kinetics(np.arange(100), with_kinetic=True)
layers = np.arange(0, 100, 1)
# layers = (3,)
# plot_kinetics(layers, with_kinetic=False)

root = tk.Tk()
root.withdraw()
database_name = filedialog.askopenfilename()
iter = 9999
visualise = Visualisation(database_name)
#
# visualise.animate_3d(animate_separate=False, const_cam_pos=False)

visualise.plot_3d(plot_separate=False, const_cam_pos=False)
# visualise.plot_3d(plot_separate=False, const_cam_pos=True, iteration=iter)

visualise.plot_2d(plot_separate=False)
# visualise.plot_2d(plot_separate=False, iteration=iter)

# for i in range(260, 301):
#     visualise.plot_2d(plot_separate=False, slice_pos=i)

visualise.plot_concentration(plot_separate=False, conc_type="atomic_tc", analytic_sol=False)
# visualise.plot_concentration(plot_separate=False, conc_type="cells", analytic_sol=True, iteration=iter)
# visualise.animate_concentration(conc_type="cells", analytic_sol=True)

# visualise.plot_h()

# for plane_ind in range(0, 61000, 100):
#     visualise.plot_3d(plot_separate=False, iteration=plane_ind, const_cam_pos=True)

# visualise.calculate_phase_size()


