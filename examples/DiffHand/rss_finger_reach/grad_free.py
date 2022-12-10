'''
ES baselines for finger reach task
'''
import os
import sys

example_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(example_base_dir)
DiffHand_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
sys.path.append(DiffHand_dir)
working_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(working_dir)

from parameterization import Design as Design_np
from utils.renderer import SimRenderer
from utils.common import *
from common import *
import numpy as np
import redmax_py as redmax
import argparse
from grad_free_util import optimize_params
import argparse
import matplotlib.pyplot as plt

def forward(params):
    action = params[:ndof_u * num_ctrl_steps]
    u = np.tanh(action)

    if optimize_design_flag:
        cage_params = params[-ndof_cage:]
        design_params = design_np.parameterize(cage_params)
        sim.set_design_params(design_params)

    sim.reset()

    # objectives coefficients
    coef_u = 0.1
    coef_task = 10.

    f_u = 0.
    f_task = 0.

    f = 0.

    for i in range(num_ctrl_steps):
        traj_idx = i * sub_steps // num_task_steps
        target_pos = target_traj[traj_idx]

        sim.set_u(u[i * ndof_u:(i + 1) * ndof_u])
        sim.forward(sub_steps, verbose=False)

        variables = sim.get_variables()

        # compute objective f
        f_u_i = np.sum(u[i * ndof_u:(i + 1) * ndof_u] ** 2)
        f_task_i = np.linalg.norm(variables - target_pos)  # L2-norm
        f_u += f_u_i
        f_task += f_task_i
        f += coef_u * f_u_i + coef_task * f_task_i

    return f, {'f_u': f_u, 'f_task': f_task}

def env_loss(params):
    loss, _ = forward(params)
    return loss

'''call back function'''
def callback_func(params, render=False, record=False, record_path=None, log=True):
    f, info = forward(params)
    if render:
        print(f'f:{f}  info:{info}')
        if optimize_design_flag:
            cage_params = params[-ndof_cage:]
            design_params, meshes = design_np.parameterize(cage_params, True)
            Vs = []
            for i in range(len(meshes)):
                Vs.append(meshes[i].V)
            sim.set_rendering_mesh_vertices(Vs)
        SimRenderer.replay(sim, record=record, record_path=record_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument("--model", type=str, default='rss_finger_reach')
    parser.add_argument('--record', action='store_true')
    parser.add_argument('--save-dir', type=str, default='data')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--record-file-name', type=str, default='rss_finger_reach_grad_free')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--optim', '-o', choices=['TwoPointsDE', 'NGOpt',
                                                  'OnePlusOne', 'CMA', 'TBPSA',
                                                  'PSO', 'RandomSearch', 'DiagonalCMA', 'FCMA'],
                        default='OnePlusOne')
    parser.add_argument('--no-design-optim', action='store_true', help = 'whether control-only')
    parser.add_argument('--max-iters', type=int, default=5000)
    parser.add_argument('--popsize', type=int, default=None)
    parser.add_argument('--single_stage', action='store_true')
    parser.add_argument('--load-dir', type = str, default = None, help = 'load optimized parameters')
    parser.add_argument('--visualize', type=str, default='True', help = 'whether visualize the simulation')
    parser.add_argument('--verbose', default = False, action = 'store_true', help = 'verbose output')

    args = parser.parse_args()

    asset_folder = os.path.abspath(os.path.join(example_base_dir, '..', 'assets'))

    np.random.seed(args.seed)

    if args.model[-4:] == '.xml':
        model_path = os.path.join(asset_folder, args.model)
    else:
        model_path = os.path.join(asset_folder, args.model + '.xml')
    
    optimize_design_flag = not args.no_design_optim
    os.makedirs(args.save_dir, exist_ok = True)
    visualize = (args.visualize == 'True')
    play_mode = (args.load_dir is not None)

    '''init sim and task'''
    sim = redmax.Simulation(model_path, args.verbose)

    if args.verbose:
        sim.print_ctrl_info()
        sim.print_design_params_info()

    num_steps = 600

    ndof_u = sim.ndof_u
    ndof_r = sim.ndof_r
    ndof_var = sim.ndof_var
    ndof_p = sim.ndof_p

    # set up camera
    sim.viewer_options.camera_pos = np.array([2.5, -4, 1.8])

    # set task
    target_traj = [np.array([10, -5, -5]), np.array([10, -5, 5]), np.array([20, 10, 10]), np.array([20, 10, -10])]
    for i in range(len(target_traj)):
        name = "target point " + str(i)
        sim.update_virtual_object(name, target_traj[i])
    num_task_steps = num_steps // len(target_traj)
    
    design_np = Design_np()

    # init design params
    cage_params = np.ones(9)
    ndof_cage = len(cage_params)

    design_params, meshes = design_np.parameterize(cage_params, True)
    Vs = []
    for i in range(len(meshes)):
        Vs.append(meshes[i].V)
    sim.set_design_params(design_params)
    sim.set_rendering_mesh_vertices(Vs)

    # init control sequence
    sub_steps = 20
    assert (num_steps % sub_steps) == 0
    num_ctrl_steps = num_steps // sub_steps
    if args.seed == 0:
        action = np.zeros(ndof_u * num_ctrl_steps)
    else:
        np.random.seed(args.seed)
        action = np.random.uniform(-0.5, 0.5, ndof_u * num_ctrl_steps)

    if not optimize_design_flag:
        params = action
    else:
        params = np.zeros(ndof_u * num_ctrl_steps + ndof_cage)
        params[0:ndof_u * num_ctrl_steps] = action
        params[-ndof_cage:] = cage_params
    n_params = len(params)

    if play_mode:
        print(f'Loading from {args.load_dir}')
        with open(os.path.join(args.load_dir, 'params.npy'), 'rb') as fp:
            params = np.load(fp)
        with open(os.path.join(args.load_dir, 'logs.npy'), 'rb') as fp:
            f_log = np.load(fp)
    else:
        bounds = []
        for i in range(num_ctrl_steps * ndof_u):
            bounds.append((-1., 1.))
        if optimize_design_flag:
            for i in range(ndof_cage):
                bounds.append((0.5, 4.))
        bounds = np.array(bounds)
        params, losses = optimize_params(optim_name=args.optim,
                                         loss_func=env_loss,
                                         num_params=n_params,
                                         init_values=params,
                                         max_iters=args.max_iters,
                                         num_workers=args.num_workers,
                                         popsize=args.popsize,
                                         bounds=bounds)

        ''' save results '''
        with open(os.path.join(args.save_dir, 'params.npy'), 'wb') as fp:
            np.save(fp, params)
        with open(os.path.join(args.save_dir, 'logs.npy'), 'wb') as fp:
            np.save(fp, losses)

    if visualize:
        ax = plt.subplot()
        ax.set_xlabel('#sim')
        ax.set_ylabel('loss')
        ax.plot(losses[:, 0], losses[:, 1])
        plt.show()

        callback_func(params, render=True, record=args.record,
                    record_path=args.record_file_name + "_optimized.gif",
                    log=False)
        
