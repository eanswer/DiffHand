'''
test forward and rendering
'''
from renderer import SimRenderer
import numpy as np
import redmax_py as redmax
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Test forward and rendering of simulation')
    parser.add_argument("--model", type = str, default = 'finger_torque') 
    
    args = parser.parse_args()

    asset_folder = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../assets'))

    if args.model[-4:] == '.xml':
        model_path = os.path.join(asset_folder, args.model)
    else:
        model_path = os.path.join(asset_folder, args.model + '.xml')

    sim = redmax.Simulation(model_path)

    num_steps = 1000

    sim.reset(backward_flag = False) # reset the simulation to start a new trajectory

    ndof_u = sim.ndof_u # number of actions
    ndof_r = sim.ndof_r # number of degrees of freedom
    ndof_var = sim.ndof_var # number of auxiliary variables

    for i in range(num_steps):
        sim.set_u(np.zeros(ndof_u))
        sim.forward(1, verbose = False)
        q = sim.get_q() # get state of the simulation, |q| = ndof_r
        qdot = sim.get_qdot() # get the velocity of the state, |qdot| = ndof_r
        variables = sim.get_variables() # get the auxiliary varibles
        
    SimRenderer.replay(sim, record = False) # render the simulation replay video