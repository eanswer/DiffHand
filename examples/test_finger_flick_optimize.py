'''
optimize an open-loop action sequence for finger to push a box to a target location
'''
from utils.renderer import SimRenderer
import numpy as np
import scipy.optimize
import redmax_py as redmax

if __name__ == '__main__':
    sim = redmax.make_sim("TorqueFingerFlick-Demo", "BDF2")
    
    ndof_r, ndof_m, ndof_u = sim.ndof_r, sim.ndof_m, sim.ndof_u

    print('ndof_r = ', ndof_r)
    print('ndof_m = ', ndof_m)
    print('ndof_u = ', ndof_u)

    num_steps = 1000

    q0 = np.array([0., np.pi / 2., np.pi / 4., 0., 0., 0.])
    sim.set_q_init(q0)

    x_goal = 10.5

    q_goal = np.zeros(3)
    P_q = np.array([10., 2., 3.])

    sim.reset(False)

    # initial guess
    u = np.zeros(ndof_u * num_steps)
    for i in range(num_steps):
        q = sim.get_q()
        error = q_goal - q[:3]
        ui = error * P_q
        
        u[i * ndof_u:(i + 1) * ndof_u] = ui

        sim.set_u(ui)

        sim.forward(1, False)

    q = sim.get_q()
    print('q = ', q)

    SimRenderer.replay(sim, record = False, record_path = "./torque_finger_flick_init.gif")

    def loss_and_grad(u): # compute loss and gradient through diff redmax
        sim.reset(True)
        
        for i in range(num_steps):
            sim.set_u(u[ndof_u * i:ndof_u * (i + 1)])
            sim.forward(1)
        
        q = sim.get_q()

        f = (q[3] - x_goal) ** 2

        # set backward info
        sim.backward_info.set_flags(False, False, False, True) # specify which gradient we want, here we only want the gradients w.r.t. control actions
        
        # set terminal derivatives
        sim.backward_info.df_du = np.zeros(ndof_u * num_steps) # specify the partial derivative of the control actions in loss function
        df_dq = np.zeros(ndof_r * num_steps) 
        df_dq[ndof_r * (num_steps - 1) + 3] = 2. * (q[3] - x_goal) # specify the partial derivative of the state q in loss function
        sim.backward_info.df_dq = df_dq

        # backward
        sim.backward()

        grad = np.copy(sim.backward_results.df_du)
        
        return f, grad
    
    def callback_func(u, render = False):
        sim.reset(False)
        for i in range(num_steps):
            sim.set_u(u[ndof_u * i:ndof_u * (i + 1)])
            sim.forward(1)
        
        q = sim.get_q()

        f = (q[3] - x_goal) ** 2

        print('f = ', f)

        if render:
            print('q = ', q)
            SimRenderer.replay(sim, record = False, record_path = "./torque_finger_flick_optimized.gif")

    res = scipy.optimize.minimize(loss_and_grad, np.copy(u), method = "L-BFGS-B", jac = True, callback = callback_func)

    callback_func(u = res.x, render = True)