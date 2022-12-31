'''
test forward and rendering
'''
from utils.renderer import SimRenderer
import numpy as np
import redmax_py as redmax
import os
import argparse
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Test forward and rendering of simulation')
    parser.add_argument("--model", type = str, default = 'finger_torque') 
    parser.add_argument('--gradient', action = 'store_true')
    parser.add_argument("--record", action = "store_true")

    args = parser.parse_args()

    asset_folder = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../assets'))

    if args.model[-4:] == '.xml':
        model_path = os.path.join(asset_folder, args.model)
    else:
        model_path = os.path.join(asset_folder, args.model + '.xml')

    sim = redmax.Simulation(model_path)

    num_steps = 1000

    sim.reset(backward_flag = args.gradient) # reset the simulation to start a new trajectory

    ndof_u = sim.ndof_u # number of actions
    ndof_r = sim.ndof_r # number of degrees of freedom
    ndof_var = sim.ndof_var # number of auxiliary variables
    loss = 0.
    df_du = np.zeros(ndof_u * num_steps)
    df_dq = np.zeros(ndof_r * num_steps)

    t0 = time.time()
    for i in range(num_steps):
        sim.set_u(np.zeros(ndof_u))
        # sim.set_u((np.random.rand(ndof_u) - 0.5) * 2.)
        # sim.set_u(np.ones(ndof_u))
        # sim.set_u(np.ones(ndof_u) * np.sin(i / 100 * np.pi))
        # if i < 50:
        #     sim.set_u(np.ones(ndof_u) * -1)
        # else:
        #     sim.set_u(np.ones(ndof_u))
        sim.forward(1, verbose = False)
        q = sim.get_q()
        loss += np.sum(q)
        df_dq[ndof_u * i:ndof_u * (i + 1)] = 1.

    t1 = time.time()

    if args.gradient:
        sim.backward_info.set_flags(False, False, False, True)
        sim.backward_info.df_du = df_du
        sim.backward_info.df_dq = df_dq
        sim.backward()

    t2 = time.time()

    fps_forward_only = num_steps / (t1 - t0)
    fps_with_gradient = num_steps / (t2 - t0)

    print('FPS (forward only) = {:.1f}, FPS (with gradient) = {:.1f}'.format(fps_forward_only, fps_with_gradient))

    SimRenderer.replay(sim, record = args.record, record_path = os.path.join('simulation_record', '{}.mp4'.format(args.model))) # render the simulation replay video