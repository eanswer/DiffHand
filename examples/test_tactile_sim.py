'''
test tactile simulation forward pass and rendering
'''
import os
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

from utils.renderer import SimRenderer
import numpy as np
import redmax_py as redmax
import os
import argparse
import cv2
import time

def visualize_tactile_image(tactile_array):
    resolution = 30
    T = len(tactile_array)
    nrows = tactile_array.shape[0]
    ncols = tactile_array.shape[1]

    imgs_tactile = np.zeros((nrows * resolution, ncols * resolution, 3), dtype = float)

    for row in range(nrows):
        for col in range(ncols):
            loc0_x = row * resolution + resolution // 2
            loc0_y = col * resolution + resolution // 2
            loc1_x = loc0_x + tactile_array[row, col][0] / 0.00015 * resolution
            loc1_y = loc0_y + tactile_array[row, col][1] / 0.00015 * resolution
            color = (0., max(0., 1. + tactile_array[row][col][2] / 0.0008), min(1., -tactile_array[row][col][2] / 0.0008))
            
            cv2.arrowedLine(imgs_tactile, (int(loc0_y), int(loc0_x)), (int(loc1_y), int(loc1_x)), color, 6, tipLength = 0.4)
    
    return imgs_tactile

def visualize_depth_image(tactile_forces):
    img = np.zeros((tactile_forces.shape[0], tactile_forces.shape[1]))
    for i in range(tactile_forces.shape[0]):
        for j in range(tactile_forces.shape[1]):
            img[i][j] = min(1., -tactile_forces[i][j][2] / 0.00120)
    return img

if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument("--model", type = str, default = 'tactile_pad')
    parser.add_argument("--render", action = "store_true")
    parser.add_argument("--record", action = "store_true")

    args = parser.parse_args()

    if args.record and not args.render:
        print('[Warning] Cannot record without rendering.')

    asset_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../assets')

    model_path = os.path.join(asset_folder, 'tactile_pad/tactile_pad.xml')

    # build simulation with the provided model path
    sim = redmax.Simulation(model_path)

    # modify the viewer options for better visualization
    sim.viewer_options.fps = 200
    sim.viewer_options.camera_pos = np.array([2. / 2.3, -2.5 / 2.3, 1.5 / 2.3 + 0.2])
    sim.viewer_options.camera_lookat = np.array([0., 0., 0.2])

    # generate control sequence, the pad is force/torque controlled with a 3-DoF translational joint
    action_array = [np.array([0., 0., 0.2]), np.array([0.1, 0., 0.2]), np.array([-0.2, 0., 0.2]), np.array([0., 0.1, 0.2]), np.array([0., -0.2, 0.2])]
    steps_array = [0, 100, 150, 200, 250, 350]
    actions = []
    for i in range(len(steps_array) - 1):
        for j in range(steps_array[i], steps_array[i + 1]):
            actions.append(action_array[i])
    
    # reset the simualtion, backward_flag means whether we want to get backward gradient later.
    sim.reset(backward_flag = False)

    ndof_u = sim.ndof_u # control dofs, which is 3 in this case
    ndof_r = sim.ndof_r # low-dim state dofs, which is 9 in this case, 3 for the pad, and 6 for the sphere ball (3 for translation, 3 for rotation in exponential coordinate representation).

    # image_pos maps each tactile marker into the arrangement location on the image (mainly for arrangement and visualization).
    image_pos = sim.get_tactile_image_pos(name = "pad")
    tactile_marker_rows, tactile_marker_cols = 0, 0
    for i in range(len(image_pos)):
        tactile_marker_rows = max(tactile_marker_rows, image_pos[i][0] + 1)
        tactile_marker_cols = max(tactile_marker_cols, image_pos[i][1] + 1)

    if args.record:
        export_folder = os.path.join("simulation_record", "tactile_pad")
        tactile_img_folder = os.path.join(export_folder, 'tactile_imgs')
        os.makedirs(tactile_img_folder, exist_ok = True)
        os.makedirs(os.path.join(tactile_img_folder, 'depth_map'), exist_ok = True)
        os.makedirs(os.path.join(tactile_img_folder, 'force_map'), exist_ok = True)
        tactile_cnt = 0

    # simulation
    t_start = time.time()
    tactile_frequency = 5
    for i in range(len(actions)):
        action = actions[i]
        sim.set_u(action)
        sim.forward(1, verbose = False, test_derivatives = False)
        
        if i % tactile_frequency == 0: # acquire tactile forces every 5 steps
            tactile_forces = sim.get_tactile_force_vector().copy() # query the tactile force, in shape (N * 3, )

            num_markers = tactile_forces.shape[0] // 3
            assert tactile_marker_rows * tactile_marker_cols == num_markers

            tactile_forces = tactile_forces.reshape((tactile_marker_rows, tactile_marker_cols, 3)) # reshape the tactile force to be arranged on an array
            
            if args.render:

                depth_image = visualize_depth_image(tactile_forces)
                tactile_image = visualize_tactile_image(tactile_forces[::10, ::10, :]) # downsample the marker points for better visualization

                cv2.imshow('tactile_depth', depth_image)
                cv2.imshow('tactile_force', tactile_image)
                cv2.waitKey(1)
                time.sleep(0.025)

                if args.record:
                    cv2.imwrite(os.path.join(os.path.join(tactile_img_folder, 'depth_map'), '{}.png'.format(tactile_cnt)), depth_image * 255)
                    cv2.imwrite(os.path.join(os.path.join(tactile_img_folder, 'force_map'), '{}.png'.format(tactile_cnt)), tactile_image * 255)
                    tactile_cnt += 1
        
    t_end = time.time()

    print('time elapsed = ', t_end - t_start, ', FPS = ', steps_array[-1] / (t_end - t_start))

    if args.render:
        SimRenderer.replay(sim, record = args.record, record_path = os.path.join('simulation_record', 'sim.mp4'))
       