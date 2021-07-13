'''
description:
define the parameterization for finger (new) with contact points
input: cage parameters
output: simulation related design parameters (joint transformation, body transformation, contact points, mass inertia, contact scale)

cage layout:
palm-k-j0-p0-j1-p1-j2-p2-tip
cage parameters:
0: j1 y scale
1: j2 y scale
2: p2-tip interface z scale
3: p2-tip interface y scale
4: tip end interface z scale
5: tip end interface y scale
6: p0 length scale
7: p1 length scale
8: p2 length scale
# 9: tip length scale
'''
import os
import sys

example_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(example_base_dir)

import torch
import numpy as np
from copy import deepcopy
import pyvista as pv

torch.set_default_dtype(torch.double)

asset_folder = os.path.abspath(os.path.join(example_base_dir, '..', 'assets/finger'))

def flatten_E(E):
    flat_E = torch.zeros(12)
    flat_E[0:9] = E[0:3, 0:3].flatten()
    flat_E[9:12] = E[0:3, 3]
    return flat_E

def compose_E(flat_E):
    E = torch.eye(4)
    E[0:3, 0:3] = flat_E[0:9].reshape(3, 3)
    E[0:3, 3] = flat_E[9:12]
    return E

def Einv(E):
    E_inv = torch.eye(4)
    E_inv[0:3, 0:3] = E[0:3, 0:3].T
    E_inv[0:3, 3] = -E[0:3, 0:3].T @ E[0:3, 3]
    return E_inv

class Interface:
    def __init__(self, width, height):
        self.width = width
        self.height = height

class TriMesh:
    def __init__(self, V, F):
        self.V = V
        self.F = F

class Cage:
    def __init__(self, width0, height0, width1, height1, length, is_joint, name_parent, name_child = None, joint_axis_origin = None):
        self.side_parent = Interface(width0, height0)
        self.side_child = Interface(width1, height1)
        self.length = length
        self.is_joint = is_joint
        self.side_parent_init = Interface(width0, height0)
        self.side_child_init = Interface(width1, height1)
        self.length_init = length
        if not self.is_joint:
            self.name = name_parent
            self.mesh, self.handles, self.lbs_mat, \
                self.contact_id, self.contact_lbs_mat = self.load(self.name)
        else:
            self.name_parent = name_parent
            self.name_child = name_child
            if self.name_parent:
                self.mesh_parent, self.handles_parent, self.lbs_mat_parent, \
                    self.contact_id_parent, self.contact_lbs_mat_parent = self.load(self.name_parent)
            self.mesh_child, self.handles_child, self.lbs_mat_child, \
                    self.contact_id_child, self.contact_lbs_mat_child = self.load(self.name_child)
            self.joint_axis_origin = joint_axis_origin

    def reset(self):
        self.side_parent = deepcopy(self.side_parent_init)
        self.side_child = deepcopy(self.side_child_init)
        self.length = self.length_init

    def load(self, name):
        mesh_path = os.path.join(asset_folder, 'meshes', name + '.obj')
        cage_path = os.path.join(asset_folder, 'cages', name + '.txt')
        weight_path = os.path.join(asset_folder, 'weights', name + '.npy')
        contact_path = os.path.join(asset_folder, 'contacts', name + '_id.npy')

        # load mesh
        mesh_pv = pv.read(mesh_path)
        V = torch.tensor(mesh_pv.points.T)
        F = torch.tensor(mesh_pv.faces.reshape(-1, 4).T[1:4, :])
        mesh = TriMesh(V, F)

        # load cage
        with open(cage_path, 'r') as fp:
            n = int(fp.readline())
            handles = torch.zeros((3, n))
            for i in range(n):
                data = fp.readline().split()
                x, y, z = float(data[0]), float(data[1]), float(data[2])
                handles[0, i], handles[1, i], handles[2, i] = x, y, z
            fp.close()

        # load weight
        lbs_mat = np.load(open(weight_path, 'rb'))
        lbs_mat = torch.tensor(lbs_mat)

        # load contact id
        with open(contact_path, 'rb') as fp:
            contact_id = np.load(fp)
            fp.close()
            contact_lbs_mat = deepcopy(lbs_mat[:, contact_id])

        return mesh, handles, lbs_mat, contact_id, contact_lbs_mat
    
    def get_handle_positions(self):
        handle_positions = torch.zeros((3, 14))
        handle_positions[0, 0], handle_positions[1, 0], handle_positions[2, 0] = 0, -self.side_parent.width / 2., -self.side_parent.height / 2.
        handle_positions[0, 1], handle_positions[1, 1], handle_positions[2, 1] = 0, -self.side_parent.width / 2., self.side_parent.height / 2.
        handle_positions[0, 2], handle_positions[1, 2], handle_positions[2, 2] = 0, self.side_parent.width / 2., -self.side_parent.height / 2.
        handle_positions[0, 3], handle_positions[1, 3], handle_positions[2, 3] = 0, self.side_parent.width / 2., self.side_parent.height / 2.
        handle_positions[0, 4], handle_positions[1, 4], handle_positions[2, 4] = self.length, -self.side_child.width / 2., -self.side_child.height / 2.
        handle_positions[0, 5], handle_positions[1, 5], handle_positions[2, 5] = self.length, -self.side_child.width / 2., self.side_child.height / 2.
        handle_positions[0, 6], handle_positions[1, 6], handle_positions[2, 6] = self.length, self.side_child.width / 2., -self.side_child.height / 2.
        handle_positions[0, 7], handle_positions[1, 7], handle_positions[2, 7] = self.length, self.side_child.width / 2., self.side_child.height / 2.
        handle_positions[:, 8] = (handle_positions[:, 1] + handle_positions[:, 5] + handle_positions[:, 7] + handle_positions[:, 3]) / 4.
        handle_positions[:, 9] = (handle_positions[:, 5] + handle_positions[:, 4] + handle_positions[:, 6] + handle_positions[:, 7]) / 4.
        handle_positions[:, 10] = (handle_positions[:, 3] + handle_positions[:, 7] + handle_positions[:, 6] + handle_positions[:, 2]) / 4.
        handle_positions[:, 11] = (handle_positions[:, 1] + handle_positions[:, 3] + handle_positions[:, 2] + handle_positions[:, 0]) / 4.
        handle_positions[:, 12] = (handle_positions[:, 0] + handle_positions[:, 2] + handle_positions[:, 6] + handle_positions[:, 4]) / 4.
        handle_positions[:, 13] = (handle_positions[:, 1] + handle_positions[:, 0] + handle_positions[:, 4] + handle_positions[:, 5]) / 4.

        return handle_positions

    def transform_mesh(self, mesh, handle_old_positions, lbs_mat, E_i_mesh):
        transformed_mesh = deepcopy(mesh)
        
        n_handles = handle_old_positions.shape[1]
        n_verts = mesh.V.shape[1]

        handle_positions = self.get_handle_positions()

        handle_transform = torch.zeros((3, 4 * n_handles))
        for i in range(n_handles):
            for j in range(3):
                handle_transform[j, i * 4 + j] = 1.
            handle_transform[0:3, i * 4 + 3] = handle_positions[:, i] - handle_old_positions[:, i]

        transformed_mesh.V = E_i_mesh[0:3, 0:3] @ handle_transform @ lbs_mat + E_i_mesh[0:3, 3:4]

        return transformed_mesh

    def transform_mesh_whole(self):
        return self.transform_mesh(self.mesh, self.handles, self.lbs_mat, self.E_i_mesh())
    
    def transform_mesh_parent(self):
        return self.transform_mesh(self.mesh_parent, self.handles_parent, self.lbs_mat_parent, self.E_i_mesh())

    def transform_mesh_child(self):
        return self.transform_mesh(self.mesh_child, self.handles_child, self.lbs_mat_child, self.joint_E_i_mesh())

    def transform_contacts(self, mesh, handle_old_positions, contact_lbs_mat, E_i_mesh):
        n_handles = handle_old_positions.shape[1]
        n_verts = mesh.V.shape[1]

        handle_positions = self.get_handle_positions()

        handle_transform = torch.zeros((3, 4 * n_handles))
        for i in range(n_handles):
            for j in range(3):
                handle_transform[j, i * 4 + j] = 1.
            handle_transform[0:3, i * 4 + 3] = handle_positions[:, i] - handle_old_positions[:, i]

        contact_points = E_i_mesh[0:3, 0:3] @ handle_transform @ contact_lbs_mat + E_i_mesh[0:3, 3:4]

        return contact_points.T

    def transform_contacts_whole(self):
        return self.transform_contacts(self.mesh, self.handles, self.contact_lbs_mat, self.E_i_mesh())

    def transform_contacts_parent(self):
        return self.transform_contacts(self.mesh_parent, self.handles_parent, self.contact_lbs_mat_parent, self.E_i_mesh())

    def transform_contacts_child(self):
        return self.transform_contacts(self.mesh_child, self.handles_child, self.contact_lbs_mat_child, self.joint_E_i_mesh())
        
    def scale_child_z(self, scale):
        self.side_child.height *= scale
    
    def scale_parent_z(self, scale):
        self.side_parent.height *= scale

    def scale_child_y(self, scale):
        self.side_child.width *= scale
    
    def scale_parent_y(self, scale):
        self.side_parent.width *= scale
    
    def scale_y(self, scale):
        self.scale_child_y(scale)
        self.scale_parent_y(scale)
    
    def scale_length(self, scale):
        self.length *= scale
    
    def E_jc(self):
        E = torch.eye(4)
        E[0, 3] = self.length
        return E
    
    def E_ji(self):
        E = torch.eye(4)
        E[0, 3] = self.length / 2.
        return E
    
    def E_i_mesh(self):
        E = Einv(self.E_ji())
        return E

    def joint_E_pj(self):
        E = torch.eye(4)
        E[0:3, 3] = self.joint_axis_origin
        return E

    def joint_E_jc(self):
        E = torch.eye(4)
        E[0, 3] = self.length
        E[0:3, 3] -= self.joint_axis_origin
        return E

    def joint_E_ji(self):
        E = torch.eye(4)
        E[0, 3] = self.length / 2.
        E[0:3, 3] -= self.joint_axis_origin
        return E

    def joint_E_i_mesh(self):
        E = Einv(self.E_ji())
        return E

    def endeffector_E_pj(self):
        h = (self.side_parent.height + self.side_child.height) / 2.
        E = torch.eye(4)
        E[0, 3] = self.length
        E[2, 3] = 0.
        return E

    def inertia(self):
        h = (self.side_parent.height + self.side_child.height) / 2.
        w = (self.side_parent.width + self.side_child.width) / 2.
        length = self.length
        mass = h * w * length
        I = torch.zeros(4)
        I[0] = mass
        I[1] = mass / 12. * (w * w + h * h)
        I[2] = mass / 12. * (h * h + length * length)
        I[3] = mass / 12. * (w * w + length * length)
        return I

    def contact_scale(self):
        old_S = self.side_parent_init.width * self.side_parent_init.height + self.side_child_init.width * self.side_child_init.height \
                + (self.side_parent_init.width + self.side_parent_init.height + self.side_child_init.width + self.side_child_init.height) * self.length_init
        new_S = self.side_parent.width * self.side_parent.height + self.side_child.width * self.side_child.height \
                + (self.side_parent.width + self.side_parent.height + self.side_child.width + self.side_child.height) * self.length
        return new_S / old_S
        
palm_cage = Cage(1.6, 3.24, 1.6, 3.24, 0.7, False, 'palm')
knuckle_cage = Cage(1.6, 3.24, 2.6, 2.6, 2.75, True, 'knuckle_parent', 'knuckle_child', joint_axis_origin = torch.tensor([1.15, 0., 0.]))
joint_cage = Cage(2.6, 2.6, 2.6, 2.6, 2.06, True, 'joint_parent', 'joint_child', joint_axis_origin = torch.tensor([1.08, 0., 0.]))
phalanx_cage = Cage(2.6, 2.6, 2.6, 2.6, 2.34, False, 'phalanx')
tip_cage = Cage(2.6, 2.6, 2.6, 2.6, 2.21, False, 'tip')

class Design:
    def __init__(self):
        self.structure = ['palm', 'k', 'j', 'p', 'j', 'p', 'j', 'p', 't']

        # build cages
        self.cages = []
        for symbol in self.structure:
            if (symbol == 'palm'):
                self.cages.append(deepcopy(palm_cage))
            elif (symbol == 'k'):
                self.cages.append(deepcopy(knuckle_cage))
            elif (symbol == 'j'):
                self.cages.append(deepcopy(joint_cage))
            elif (symbol == 'p'):
                self.cages.append(deepcopy(phalanx_cage))
            elif (symbol == 't'):
                self.cages.append(deepcopy(tip_cage))
        self.ndof_p3 = 0
        self.ndof_p6 = 0
        self.sub_ndof_p3 = []
        for i in range(len(self.cages)):
            symbol = self.structure[i]
            if symbol == 'j' or symbol == 'k':
                self.ndof_p3 += self.cages[i].contact_id_parent.shape[0] * 3
                self.sub_ndof_p3.append(self.cages[i].contact_id_parent.shape[0] * 3)
                self.ndof_p6 += 1
                self.ndof_p3 += self.cages[i].contact_id_child.shape[0] * 3
                self.sub_ndof_p3.append(self.cages[i].contact_id_child.shape[0] * 3)
                self.ndof_p6 += 1
            elif symbol == 'p' or symbol == 't' or symbol == 'palm':
                self.ndof_p3 += self.cages[i].contact_id.shape[0] * 3
                self.sub_ndof_p3.append(self.cages[i].contact_id.shape[0] * 3)
                self.ndof_p6 += 1
        
        self.n_link = 13                      # number of sublinks (a joint is composed from two sublinks)
        self.ndof_p1 = (self.n_link + 1) * 12 # one extra joint for two end effectors
        self.ndof_p2 = self.n_link * 12

    def parameterize(self, cage_parameters, generate_mesh = False):
        for i in range(len(self.cages)):
            self.cages[i].reset()

        n_link = self.n_link                   # number of parametric links
        ndof_p1 = (n_link + 1) * 12   # one extra joint for end effector
        ndof_p2 = n_link * 12
        ndof_p3 = self.ndof_p3
        ndof_p4 = n_link * 4
        ndof_p6 = self.ndof_p6
        ndof_p = ndof_p1 + ndof_p2 + ndof_p3 + ndof_p4 + ndof_p6

        design_params = torch.zeros(ndof_p)

        # apply cage parameters into cages
        parameter_idx = 0
        for i in range(len(self.cages)):
            symbol = self.structure[i]
            if (symbol == 'j' and self.structure[i + 1] == 'p' and self.structure[i - 1] == 'p'):
                self.cages[i - 1].scale_child_y(cage_parameters[parameter_idx])
                self.cages[i].scale_y(cage_parameters[parameter_idx])
                self.cages[i + 1].scale_parent_y(cage_parameters[parameter_idx])
                parameter_idx += 1
            elif (symbol == 'p' and self.structure[i + 1] != 'j' and self.structure[i + 1] != 'k'):
                self.cages[i].scale_child_z(cage_parameters[parameter_idx])
                self.cages[i + 1].scale_parent_z(cage_parameters[parameter_idx])
                parameter_idx += 1
                self.cages[i].scale_child_y(cage_parameters[parameter_idx])
                self.cages[i + 1].scale_parent_y(cage_parameters[parameter_idx])
                parameter_idx += 1
            elif (symbol == 't'):
                self.cages[i].scale_child_z(cage_parameters[parameter_idx])
                parameter_idx += 1
                self.cages[i].scale_child_y(cage_parameters[parameter_idx])
                parameter_idx += 1

        for i in range(len(self.cages)):
            symbol = self.structure[i]
            if (symbol == 'p'):
                self.cages[i].scale_length(cage_parameters[parameter_idx])
                parameter_idx += 1

        # convert from cages to design params
        # design params 1
        idx = 0
        for i in range(len(self.cages)):
            symbol = self.structure[i]
            if (symbol == 'j' or symbol == 'k'):
                # joint parent part
                if (self.structure[i - 1] == 'j' or self.structure[i - 1] == 'k'):
                    design_params[idx * 12:(idx + 1) * 12] = flatten_E(self.cages[i - 1].joint_E_jc())
                else:
                    design_params[idx * 12:(idx + 1) * 12] = flatten_E(self.cages[i - 1].E_jc())
                idx += 1
                # joint child part
                design_params[idx * 12:(idx + 1) * 12] = flatten_E(self.cages[i].joint_E_pj())
                idx += 1
            elif (symbol == 'p'):
                if (self.structure[i - 1] == 'j'):
                    design_params[idx * 12:(idx + 1) * 12] = flatten_E(self.cages[i - 1].joint_E_jc())
                else:
                    design_params[idx * 12:(idx + 1) * 12] = flatten_E(self.cages[i - 1].E_jc())
                idx += 1
            elif (symbol == 't'):
                if (self.structure[i - 1] == 'j'):
                    design_params[idx * 12:(idx + 1) * 12] = flatten_E(self.cages[i - 1].joint_E_jc())
                else:
                    design_params[idx * 12:(idx + 1) * 12] = flatten_E(self.cages[i - 1].E_jc())
                idx += 1
                design_params[idx * 12:(idx + 1) * 12] = flatten_E(self.cages[i].endeffector_E_pj())
                idx += 1
            elif (symbol == 'palm'):
                E = torch.eye(4)
                E[1, 1] = -1
                E[2, 2] = -1
                E[2, 3] = 4
                design_params[idx * 12:(idx + 1) * 12] = flatten_E(E)
                idx += 1

        # design param 2
        idx = 0
        for i in range(len(self.cages)):
            symbol = self.structure[i]
            if (symbol == 'p' or symbol == 't' or symbol == 'palm'):
                design_params[ndof_p1 + idx * 12:ndof_p1 + (idx + 1) * 12] = flatten_E(self.cages[i].E_ji())
                idx += 1
            elif (symbol == 'j' or symbol == 'k'):
                # joint parent part
                design_params[ndof_p1 + idx * 12:ndof_p1 + (idx + 1) * 12] = flatten_E(self.cages[i].E_ji())
                idx += 1
                # joint child part
                design_params[ndof_p1 + idx * 12:ndof_p1 + (idx + 1) * 12] = flatten_E(self.cages[i].joint_E_ji())
                idx += 1
        
        # design param 3
        param_id = ndof_p1 + ndof_p2
        for i in range(len(self.cages)):
            symbol = self.structure[i]
            if (symbol == 'p' or symbol == 't' or symbol == 'palm'):
                design_params[param_id:param_id + self.cages[i].contact_id.shape[0] * 3] = self.cages[i].transform_contacts_whole().flatten()
                param_id += self.cages[i].contact_id.shape[0] * 3
            elif (symbol == 'j' or symbol == 'k'):
                # joint parent part
                design_params[param_id:param_id + self.cages[i].contact_id_parent.shape[0] * 3] = self.cages[i].transform_contacts_parent().flatten()
                param_id += self.cages[i].contact_id_parent.shape[0] * 3
                # joint child part
                design_params[param_id:param_id + self.cages[i].contact_id_child.shape[0] * 3] = self.cages[i].transform_contacts_child().flatten()
                param_id += self.cages[i].contact_id_child.shape[0] * 3

        # design param 4
        idx = 0
        for i in range(len(self.cages)):
            symbol = self.structure[i]
            if (symbol == 'p' or symbol == 't' or symbol == 'palm'):
                design_params[ndof_p1 + ndof_p2 + ndof_p3 + idx * 4:ndof_p1 + ndof_p2 + ndof_p3 + (idx + 1) * 4] = self.cages[i].inertia()
                idx += 1
            elif (symbol == 'j' or symbol == 'k'):
                # joint parent part
                design_params[ndof_p1 + ndof_p2 + ndof_p3 + idx * 4:ndof_p1 + ndof_p2 + ndof_p3 + (idx + 1) * 4] = self.cages[i].inertia()
                idx += 1
                # joint child part
                design_params[ndof_p1 + ndof_p2 + ndof_p3 + idx * 4:ndof_p1 + ndof_p2 + ndof_p3 + (idx + 1) * 4] = self.cages[i].inertia()
                idx += 1

        # design param 6
        param_id = ndof_p1 + ndof_p2 + ndof_p3 + ndof_p4
        for i in range(len(self.cages)):
            symbol = self.structure[i]
            if (symbol == 'p' or symbol == 't' or symbol == 'palm'):
                design_params[param_id] = self.cages[i].contact_scale()
                param_id += 1
            elif (symbol == 'j' or symbol == 'k'):
                design_params[param_id] = self.cages[i].contact_scale()
                param_id += 1
                design_params[param_id] = self.cages[i].contact_scale()
                param_id += 1

        if generate_mesh:
            meshes = []
            for i in range(len(self.cages)):
                symbol = self.structure[i]
                if (symbol == 'p' or symbol == 't' or symbol == 'palm'):
                    meshes.append(self.cages[i].transform_mesh_whole())
                elif (symbol == 'j' or symbol == 'k'):
                    # joint parent part
                    meshes.append(self.cages[i].transform_mesh_parent())
                    # joint child part
                    meshes.append(self.cages[i].transform_mesh_child())
            return design_params, meshes
        else:
            return design_params