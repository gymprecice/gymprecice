from __future__ import division, print_function
import os
import sys
import argparse
import numpy as np
import precice
from precice import action_write_initial_data, action_write_iteration_checkpoint, \
    action_read_iteration_checkpoint
import os
import time

grid_coords = [
    [ 4.30603865e-03, 4.98142036e-02, 5.00000000e-03],
    [ 4.20252675e-03, 4.98230433e-02, 5.00000000e-03],
    [ 4.09899670e-03, 4.98316678e-02, 5.00000000e-03],
    [ 3.99544900e-03, 4.98400773e-02, 5.00000000e-03],
    [ 3.89188405e-03, 4.98482715e-02, 5.00000000e-03],
    [ 3.78830225e-03, 4.98562505e-02, 5.00000000e-03],
    [ 3.68470410e-03, 4.98640143e-02, 5.00000000e-03],
    [ 3.58109005e-03, 4.98715629e-02, 5.00000000e-03],
    [ 3.47746055e-03, 4.98788961e-02, 5.00000000e-03],
    [ 3.37381605e-03, 4.98860140e-02, 5.00000000e-03],
    [ 3.27015695e-03, 4.98929165e-02, 5.00000000e-03],
    [ 3.16648375e-03, 4.98996036e-02, 5.00000000e-03],
    [ 3.06279690e-03, 4.99060752e-02, 5.00000000e-03],
    [ 2.95909680e-03, 4.99123314e-02, 5.00000000e-03],
    [ 2.85538390e-03, 4.99183722e-02, 5.00000000e-03],
    [ 2.75165870e-03, 4.99241975e-02, 5.00000000e-03],
    [ 2.64792160e-03, 4.99298072e-02, 5.00000000e-03],
    [ 2.54417305e-03, 4.99352013e-02, 5.00000000e-03],
    [ 2.44041355e-03, 4.99403799e-02, 5.00000000e-03],
    [ 2.33664355e-03, 4.99453428e-02, 5.00000000e-03],
    [ 2.23286345e-03, 4.99500901e-02, 5.00000000e-03],
    [ 2.12907365e-03, 4.99546218e-02, 5.00000000e-03],
    [ 2.02527470e-03, 4.99589378e-02, 5.00000000e-03],
    [ 1.92146700e-03, 4.99630381e-02, 5.00000000e-03],
    [ 1.81765100e-03, 4.99669227e-02, 5.00000000e-03],
    [ 1.71382720e-03, 4.99705916e-02, 5.00000000e-03],
    [ 1.60999595e-03, 4.99740448e-02, 5.00000000e-03],
    [ 1.50615775e-03, 4.99772822e-02, 5.00000000e-03],
    [ 1.40231305e-03, 4.99803038e-02, 5.00000000e-03],
    [ 1.29846230e-03, 4.99831097e-02, 5.00000000e-03],
    [ 1.19460595e-03, 4.99856998e-02, 5.00000000e-03],
    [ 1.09074445e-03, 4.99880741e-02, 5.00000000e-03],
    [ 9.86878250e-04, 4.99902325e-02, 5.00000000e-03],
    [ 8.83007750e-04, 4.99921752e-02, 5.00000000e-03],
    [ 7.79133450e-04, 4.99939020e-02, 5.00000000e-03],
    [ 6.75255800e-04, 4.99954130e-02, 5.00000000e-03],
    [ 5.71375250e-04, 4.99967081e-02, 5.00000000e-03],
    [ 4.67492200e-04, 4.99977874e-02, 5.00000000e-03],
    [ 3.63607100e-04, 4.99986509e-02, 5.00000000e-03],
    [ 2.59720450e-04, 4.99992985e-02, 5.00000000e-03],
    [ 1.55832700e-04, 4.99997302e-02, 5.00000000e-03],
    [ 5.19443000e-05, 4.99999460e-02, 5.00000000e-03],
    [-5.19443000e-05, 4.99999460e-02, 5.00000000e-03],
    [-1.55832700e-04, 4.99997302e-02, 5.00000000e-03],
    [-2.59720450e-04, 4.99992985e-02, 5.00000000e-03],
    [-3.63607100e-04, 4.99986509e-02, 5.00000000e-03],
    [-4.67492200e-04, 4.99977874e-02, 5.00000000e-03],
    [-5.71375250e-04, 4.99967081e-02, 5.00000000e-03],
    [-6.75255800e-04, 4.99954130e-02, 5.00000000e-03],
    [-7.79133450e-04, 4.99939020e-02, 5.00000000e-03],
    [-8.83007750e-04, 4.99921752e-02, 5.00000000e-03],
    [-9.86878250e-04, 4.99902325e-02, 5.00000000e-03],
    [-1.09074445e-03, 4.99880741e-02, 5.00000000e-03],
    [-1.19460595e-03, 4.99856998e-02, 5.00000000e-03],
    [-1.29846230e-03, 4.99831097e-02, 5.00000000e-03],
    [-1.40231305e-03, 4.99803038e-02, 5.00000000e-03],
    [-1.50615775e-03, 4.99772822e-02, 5.00000000e-03],
    [-1.60999595e-03, 4.99740448e-02, 5.00000000e-03],
    [-1.71382720e-03, 4.99705916e-02, 5.00000000e-03],
    [-1.81765100e-03, 4.99669227e-02, 5.00000000e-03],
    [-1.92146700e-03, 4.99630381e-02, 5.00000000e-03],
    [-2.02527470e-03, 4.99589378e-02, 5.00000000e-03],
    [-2.12907365e-03, 4.99546218e-02, 5.00000000e-03],
    [-2.23286345e-03, 4.99500901e-02, 5.00000000e-03],
    [-2.33664355e-03, 4.99453427e-02, 5.00000000e-03],
    [-2.44041355e-03, 4.99403799e-02, 5.00000000e-03],
    [-2.54417305e-03, 4.99352013e-02, 5.00000000e-03],
    [-2.64792160e-03, 4.99298072e-02, 5.00000000e-03],
    [-2.75165870e-03, 4.99241975e-02, 5.00000000e-03],
    [-2.85538390e-03, 4.99183722e-02, 5.00000000e-03],
    [-2.95909680e-03, 4.99123314e-02, 5.00000000e-03],
    [-3.06279690e-03, 4.99060752e-02, 5.00000000e-03],
    [-3.16648375e-03, 4.98996036e-02, 5.00000000e-03],
    [-3.27015695e-03, 4.98929165e-02, 5.00000000e-03],
    [-3.37381605e-03, 4.98860140e-02, 5.00000000e-03],
    [-3.47746055e-03, 4.98788961e-02, 5.00000000e-03],
    [-3.58109005e-03, 4.98715629e-02, 5.00000000e-03],
    [-3.68470410e-03, 4.98640143e-02, 5.00000000e-03],
    [-3.78830225e-03, 4.98562505e-02, 5.00000000e-03],
    [-3.89188405e-03, 4.98482715e-02, 5.00000000e-03],
    [-3.99544900e-03, 4.98400773e-02, 5.00000000e-03],
    [-4.09899670e-03, 4.98316678e-02, 5.00000000e-03],
    [-4.20252675e-03, 4.98230433e-02, 5.00000000e-03],
    [-4.30603865e-03, 4.98142036e-02, 5.00000000e-03]]


cwd = os.getcwd()
p_idx = int(cwd.split('_')[-1])

print(f"Starting Fluid_{p_idx} Solver")
t0 = time.time()
interface = precice.Interface(f"Fluid_{p_idx}", '../precice-config.xml', 0, 1)
print(f"Done Configure preCICE from Fluid_{p_idx} in {time.time()-t0} seconds")

dimensions = interface.get_dimensions()

mesh_name = f"Fluid-Mesh_{p_idx}"
write_var = f"Pressure_{p_idx}"
read_var = f"Velocity_{p_idx}"
meshID = interface.get_mesh_id(mesh_name)
pressureID = interface.get_data_id(write_var, meshID)
velocityID = interface.get_data_id(read_var, meshID)

grid_coords = np.array(grid_coords)
N = grid_coords.shape[0]
print(f"Fluid Solver number: {p_idx} grid size is {N}")
vertexIDs = interface.set_mesh_vertices(meshID, grid_coords)

t = 0

print(f"Fluid Solver_{p_idx}: init precice...")
# preCICE defines timestep size of solver via precice-config.xml
precice_dt = interface.initialize()

time_step = 0

pressure = (p_idx + 1) * 1000 * np.ones(N)
if interface.is_action_required(action_write_initial_data()):
    interface.write_block_scalar_data(pressureID, vertexIDs, pressure)
    interface.mark_action_fulfilled(action_write_initial_data())
    print(f"time: {time_step}, avg-{write_var} using {mesh_name} write = {pressure.mean():.4f}")

interface.initialize_data()

if interface.is_read_data_available():
    velocity = interface.read_block_vector_data(velocityID, vertexIDs)
    print(f"time: {time_step}, avg-{read_var} using {mesh_name} read = {velocity.mean():.4f}")

while interface.is_coupling_ongoing():
    # When an implicit coupling scheme is used, checkpointing is required
    if interface.is_action_required(action_write_iteration_checkpoint()):
        interface.mark_action_fulfilled(action_write_iteration_checkpoint())

    # advance the solver
    pressure += 1

    time_step += precice_dt
    # write, advance and then read following the tutorial elastic-tube-1d/fluid-python/FluidSolver.py
    # last read will not be executed
    interface.write_block_scalar_data(pressureID, vertexIDs, pressure)
    interface.advance(precice_dt)
    velocity = interface.read_block_vector_data(
        velocityID, vertexIDs)

    print(f"time: {time_step} (Fluid-{p_idx}), avg-{write_var} using {mesh_name} write = {pressure.mean():.4f}")
    print(f"time: {time_step} (Fluid-{p_idx}), avg-{read_var} using {mesh_name} read = {velocity.mean():.4f}")

    # i.e. not yet converged
    if interface.is_action_required(action_read_iteration_checkpoint()):
        interface.mark_action_fulfilled(action_read_iteration_checkpoint())

print(f"Exiting FluidSolver_{p_idx}")

interface.finalize()
