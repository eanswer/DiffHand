'''
description:
co-optimization for two finger assemble task
'''
import os
import sys

example_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(example_base_dir)
DiffHand_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
sys.path.append(DiffHand_dir)
working_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(working_dir)

from parameterization_torch import Design as Design
from parameterization import Design as Design_np

from utils.renderer import SimRenderer
from utils.common import *
from common import *
import numpy as np
import scipy.optimize
import redmax_py as redmax
import os
import argparse
import time
import torch
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.double)

optimize_design_flag = True
time_start = time.time()
if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument("--model", type = str, default = 'rss_two_finger_assemble')
    parser.add_argument('--record', action = 'store_true')
    parser.add_argument('--record-file-name', type = str, default = 'rss_two_finger_assemble')
    parser.add_argument('--seed', type = int, default = 0)
    parser.add_argument('--save-dir', type = str, default = './results/tmp/')
    parser.add_argument('--no-design-optim', action='store_true', help = 'whether control-only')
    parser.add_argument('--visualize', type=str, default='True', help = 'whether visualize the simulation')
    parser.add_argument('--load-dir', type = str, default = None, help = 'load optimized parameters')
    parser.add_argument('--verbose', default = False, action = 'store_true', help = 'verbose output')
    parser.add_argument('--continuation', default = False, action = 'store_true')
    parser.add_argument('--test-derivatives', default = False, action = 'store_true')

    asset_folder = os.path.abspath(os.path.join(example_base_dir, '..', 'assets'))

    args = parser.parse_args()

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

    q_init = sim.get_q_init().copy()
    q_init[-1] = np.pi / 6.
    sim.set_q_init(q_init)

    num_steps = 500

    ndof_u = sim.ndof_u
    ndof_r = sim.ndof_r
    ndof_var = sim.ndof_var
    ndof_p = sim.ndof_p

    # set up camera
    sim.viewer_options.camera_pos = np.array([2.5, -4, 1.8])
    sim.viewer_options.speed = 0.5

    # init design params
    design = Design()
    design_np = Design_np()
    cage_params = np.ones(17)
    ndof_cage = len(cage_params)

    design_params, meshes = design_np.parameterize(cage_params, True)
    sim.set_design_params(design_params)
    Vs = []
    for i in range(len(meshes)):
        Vs.append(meshes[i].V)
    sim.set_rendering_mesh_vertices(Vs)

    # init control sequence
    sub_steps = 5
    assert (num_steps % sub_steps) == 0
    num_ctrl_steps = num_steps // sub_steps
    if args.seed == 0:
        action = np.zeros(ndof_u * num_ctrl_steps)
    else:
        np.random.seed(args.seed)
        action = np.random.uniform(-0.5, 0.5, ndof_u * num_ctrl_steps)

    if visualize:
        print('ndof_p = ', ndof_p)
        print('ndof_u = ', len(action))
        print('ndof_cage = ', ndof_cage)

    if not optimize_design_flag:
        params = action
    else:
        params = np.zeros(ndof_u * num_ctrl_steps + ndof_cage)
        params[0:ndof_u * num_ctrl_steps] = action
        params[-ndof_cage:] = cage_params

    # init optimization history
    f_log = []
    global num_sim
    num_sim = 0

    coef_design = 0.

    '''compute the objectives by forward pass'''
    def forward(params, backward_flag = False):
        global num_sim
        num_sim += 1

        action = params[:ndof_u * num_ctrl_steps]
        u = np.tanh(action)
        # u = action
        
        if optimize_design_flag:
            cage_params = params[-ndof_cage:]
            design_params = design_np.parameterize(cage_params)
            sim.set_design_params(design_params)

        sim.reset(backward_flag = backward_flag, backward_design_params_flag = backward_flag and optimize_design_flag)

        # objectives coefficients
        coef_u = 0.
        coef_box0_pos = 15
        coef_task = 5
        coef_touch = 1
        coef_rot = 50.
        
        f_u = 0.
        f_box0_pos = 0.
        f_task = 0.
        f_touch = 0.
        f_rot = 0.
        f_design = 0.

        f = 0.

        if backward_flag:
            df_dq = np.zeros(ndof_r * num_steps)
            df_du = np.zeros(ndof_u * num_steps)
            df_dvar = np.zeros(ndof_var * num_steps)
            if optimize_design_flag:
                df_dp = np.zeros(ndof_p)
        
        if optimize_design_flag:
            f_design = np.sum(cage_params ** 2) * num_ctrl_steps
            f += coef_design * f_design

        for i in range(num_ctrl_steps):
            sim.set_u(u[i * ndof_u:(i + 1) * ndof_u])
            sim.forward(sub_steps, verbose = args.verbose)
            
            variables = sim.get_variables()
            q = sim.get_q()

            finger_pos0 = variables[0:3]
            finger_pos1 = variables[3:6]
            box0_center_pos = variables[6:9]
            box0_touch_pos = variables[9:12]
            box0_hole_pos = variables[12:15]
            box1_center_pos = variables[15:18]
            box1_touch_pos = variables[18:21]
            box0_rot = q[-4]
            box1_rot = q[-1]

            # compute objective f
            f_u_i = np.sum(u[i * ndof_u:(i + 1) * ndof_u] ** 2)
            f_touch_i = np.sum((finger_pos0 - box1_touch_pos) ** 2) + np.sum((finger_pos1 - box0_touch_pos) ** 2) - 6 ** 2 - 2.8 ** 2 # subtract the cube size
            f_box0_pos_i = np.sum(box0_center_pos[0:2] ** 2)
            f_task_i = np.sum((box0_hole_pos - box1_center_pos) ** 2)
            f_rot_i = (box0_rot - box1_rot) ** 2

            f_u += f_u_i * coef_u
            f_touch += f_touch_i * coef_touch
            f_box0_pos += f_box0_pos_i * coef_box0_pos
            f_task += f_task_i * coef_task
            f_rot += f_rot_i * coef_rot

            f += coef_u * f_u_i + coef_touch * f_touch_i + coef_box0_pos * f_box0_pos_i + coef_rot * f_rot_i + coef_task * f_task_i

            # backward info
            if backward_flag:
                ''' f_u '''
                df_du[i * sub_steps * ndof_u:(i * sub_steps + 1) * ndof_u] += \
                    coef_u * 2. * u[i * ndof_u:(i + 1) * ndof_u]

                ''' f_touch '''
                df_dvar[((i + 1) * sub_steps - 1) * ndof_var + 0:((i + 1) * sub_steps - 1) * ndof_var + 3] += \
                    coef_touch * 2. * (finger_pos0 - box1_touch_pos)
                df_dvar[((i + 1) * sub_steps - 1) * ndof_var + 18:((i + 1) * sub_steps - 1) * ndof_var + 21] += \
                    -coef_touch * 2. * (finger_pos0 - box1_touch_pos)
                df_dvar[((i + 1) * sub_steps - 1) * ndof_var + 3:((i + 1) * sub_steps - 1) * ndof_var + 6] += \
                    coef_touch * 2. * (finger_pos1 - box0_touch_pos)
                df_dvar[((i + 1) * sub_steps - 1) * ndof_var + 9:((i + 1) * sub_steps - 1) * ndof_var + 12] += \
                    -coef_touch * 2. * (finger_pos1 - box0_touch_pos)

                ''' f_box0_pos '''
                df_dvar[((i + 1) * sub_steps - 1) * ndof_var + 6:((i + 1) * sub_steps - 1) * ndof_var + 8] += \
                    coef_box0_pos * 2. * box0_center_pos[0:2]

                ''' f_task '''
                df_dvar[((i + 1) * sub_steps - 1) * ndof_var + 12:((i + 1) * sub_steps - 1) * ndof_var + 15] += \
                    coef_task * 2. * (box0_hole_pos - box1_center_pos)
                df_dvar[((i + 1) * sub_steps - 1) * ndof_var + 15:((i + 1) * sub_steps - 1) * ndof_var + 18] += \
                    -coef_task * 2. * (box0_hole_pos - box1_center_pos)

                ''' f_rot '''
                df_dq[(i + 1) * sub_steps * ndof_r - 4] += coef_rot * 2. * (box0_rot - box1_rot)
                df_dq[(i + 1) * sub_steps * ndof_r - 1] += -coef_rot * 2. * (box0_rot - box1_rot)

        if backward_flag:
            sim.backward_info.set_flags(False, False, optimize_design_flag, True)
            sim.backward_info.df_du = df_du
            sim.backward_info.df_dq = df_dq
            sim.backward_info.df_dvar = df_dvar
            if optimize_design_flag:
                sim.backward_info.df_dp = df_dp

        return f, {'f_touch': f_touch, 'f_box0_pos': f_box0_pos, 'f_task': f_task, 'f_rot': f_rot}

    '''compute loss and gradient'''
    def loss_and_grad(params):
        with torch.no_grad():
            f, _ = forward(params, backward_flag = True)
            sim.backward()

        grad = np.zeros(len(params))

        # gradient for control params
        action = params[:ndof_u * num_ctrl_steps]
        df_du_full = np.copy(sim.backward_results.df_du)
        grad[:num_ctrl_steps * ndof_u] = np.sum(df_du_full.reshape(num_ctrl_steps, sub_steps, ndof_u), axis = 1).reshape(-1)
        grad[:num_ctrl_steps * ndof_u] = grad[:num_ctrl_steps * ndof_u] * (1. - np.tanh(action) ** 2)

        # gradient for design params
        if optimize_design_flag:
            df_dp = torch.tensor(np.copy(sim.backward_results.df_dp))
            cage_params = torch.tensor(params[-ndof_cage:], dtype = torch.double, requires_grad = True)
            design_params = design.parameterize(cage_params)
            design_params.backward(df_dp)
            df_dcage = cage_params.grad.numpy()
            grad[-ndof_cage:] = df_dcage
            # grad[-ndof_cage:] += coef_design * 2. * params[-ndof_cage:] * num_ctrl_steps

        return f, grad

    '''call back function'''
    def callback_func(params, render = False, record = False, record_path = None, log = True):
        # print_info('in callback')

        f, info = forward(params, backward_flag = False)

        global f_log, stage, num_sim
        num_sim -= 1
        time_now = time.time()
        global time_start
        # print_info('iteration ', len(f_log), ', num_sim = ', num_sim, ', Objective = ', f, info)
        print_info('iteration ', len(f_log), ', num_sim = ', num_sim, ', Objective = ', f, info, ', time = ', time_now - time_start)
        # time_prev = time_now
        if log:
            f_log.append(np.array([num_sim, f]))

        if render:
            if optimize_design_flag:
                cage_params = params[-ndof_cage:]
                _, meshes = design_np.parameterize(cage_params, True)
                Vs = []
                for i in range(len(meshes)):
                    Vs.append(meshes[i].V)
                sim.set_rendering_mesh_vertices(Vs)
                
            sim.viewer_options.speed = 0.2
            SimRenderer.replay(sim, record = record, record_path = record_path)

    if not play_mode:
        ''' checking initial guess '''
        callback_func(params, render = False, log = True)
        if visualize:
            print_info('Press [Esc] to continue')
            callback_func(params, render = True, log = False, record = args.record, record_path = args.record_file_name + "_init.gif")

        t0 = time.time()

        ''' optimization by L-BFGS-B '''
        # set bounds for optimization variables
        bounds = []
        for i in range(num_ctrl_steps * ndof_u):
            bounds.append((-1., 1.))
        if optimize_design_flag:
            bounds.append((0.5, 6.))
            for i in range(ndof_cage - 1):
                bounds.append((0.5, 3.))

        if args.continuation:
            print_ok('start scale = 0.01')
            sim.set_contact_scale(0.01)
            res = scipy.optimize.minimize(loss_and_grad, np.copy(params), method = "L-BFGS-B", jac = True, callback = callback_func, bounds = bounds, options={'maxiter': 100})
            
            params = np.copy(res.x)
            
            print_ok('start scale = 0.1')
            sim.set_contact_scale(0.1)
            res = scipy.optimize.minimize(loss_and_grad, np.copy(params), method = "L-BFGS-B", jac = True, callback = callback_func, bounds = bounds, options={'maxiter': 100})
            
            params = np.copy(res.x)

            print_ok('start scale = 1')
            sim.set_contact_scale(1)
            res = scipy.optimize.minimize(loss_and_grad, np.copy(params), method = "L-BFGS-B", jac = True, callback = callback_func, bounds = bounds, options={'maxiter': 100})
        else:
            res = scipy.optimize.minimize(loss_and_grad, np.copy(params), method = "L-BFGS-B", jac = True, callback = callback_func, bounds = bounds, options={'maxiter': 100})

        params = np.copy(res.x)

        t1 = time.time()


        print('time = ', t1 - t0)

        ''' save results '''
        with open(os.path.join(args.save_dir, 'params.npy'), 'wb') as fp:
            np.save(fp, params)
        f_log = np.array(f_log)
        with open(os.path.join(args.save_dir, 'logs.npy'), 'wb') as fp:
            np.save(fp, f_log)
    else:
        with open(os.path.join(args.load_dir, 'params.npy'), 'rb') as fp:
            params = np.load(fp)
        with open(os.path.join(args.load_dir, 'logs.npy'), 'rb') as fp:
            f_log = np.load(fp)

    if visualize:
        if optimize_design_flag:
            print('design params = ', params[-ndof_cage:])

        # ''' visualize the optimized design and control '''
        print_info('Press [Esc] to continue')
        callback_func(params, render = True, record = args.record, record_path = args.record_file_name + "_optimized.gif", log = False)
        
        ax = plt.subplot()
        ax.set_xlabel('#sim')
        ax.set_ylabel('loss')
        ax.plot(f_log[:, 0], f_log[:, 1])
        plt.show()

        if args.test_derivatives:
            u = params[:ndof_u * num_ctrl_steps]
            print('min_u = ', np.min(u), ', max_u = ', np.max(u))

            ''' test gradient by finite difference '''
            f, grad = loss_and_grad(params)

            n_params = len(params)

            eps = 1e-5
            for _ in range(8):
                df_dparam_fd = np.zeros(n_params)
                for i in range(n_params):
                    params_pos = params.copy()
                    params_pos[i] += eps

                    f_pos, _ = forward(params_pos, backward_flag = False)

                    df_dparam_fd[i] = (f_pos - f) / eps

                print('eps = ', eps)
                abs_error = np.linalg.norm(df_dparam_fd - grad)
                rel_error = abs_error / (np.linalg.norm(grad) + 1e-7)
                print('df_dparam : error = {:10.6e}, rel_error = {:10.6e}'.format(abs_error, rel_error))  
                df_dparam_fd_normalized = df_dparam_fd / np.linalg.norm(df_dparam_fd)
                grad_normalized = grad / np.linalg.norm(grad)
                print('dot product: ', np.dot(df_dparam_fd_normalized, grad_normalized))

                eps /= 10.