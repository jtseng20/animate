import torch
import numpy as np
from .quaternion import qmul, qrot
import torch.nn as nn
from scipy.spatial.transform import Rotation as R

amass_offsets = [
    [0.0, 0.0, 0.0],

    [0.058581, -0.082280, -0.017664],
    [0.043451, -0.386469, 0.008037],
    [-0.014790, -0.426874, -0.037428],
    [0.041054, -0.060286, 0.122042],
    [0.0, 0.0, 0.0],

    [-0.060310, -0.090513, -0.013543],
    [-0.043257, -0.383688, -0.004843],
    [0.019056, -0.420046, -0.034562],
    [-0.034840, -0.062106, 0.130323],
    [0.0, 0.0, 0.0],

    [0.004439, 0.124404, -0.038385],
    [0.004488, 0.137956, 0.026820],
    [-0.002265, 0.056032, 0.002855],
    [-0.013390, 0.211636, -0.033468],
    [0.010113, 0.088937, 0.050410],
    [0.0, 0.0, 0.0],

    [0.071702, 0.114000, -0.018898],
    [0.122921, 0.045205, -0.019046],
    [0.255332, -0.015649, -0.022946],
    [0.265709, 0.012698, -0.007375],
    [0.0, 0.0, 0.0],

    [-0.082954, 0.112472, -0.023707],
    [-0.113228, 0.046853, -0.008472],
    [-0.260127, -0.014369, -0.031269],
    [-0.269108, 0.006794, -0.006027],
    [0.0, 0.0, 0.0]
]

sk_offsets = [
    [-42.198200, 91.614723, -40.067841],

    [0.103456, 1.857829, 10.548506],
    [43.499992, -0.000038, -0.000002],
    [42.372192, 0.000015, -0.000007],
    [17.299999, -0.000002, 0.000003],
    [0.000000, 0.000000, 0.000000],

    [0.103457, 1.857829, -10.548503],
    [43.500042, -0.000027, 0.000008],
    [42.372257, -0.000008, 0.000014],
    [17.299992, -0.000005, 0.000004],
    [0.000000, 0.000000, 0.000000],

    [6.901968, -2.603733, -0.000001],
    [12.588099, 0.000002, 0.000000],
    [12.343206, 0.000000, -0.000001],
    [25.832886, -0.000004, 0.000003],
    [11.766620, 0.000005, -0.000001],
    [0.000000, 0.000000, 0.000000],

    [19.745899, -1.480370, 6.000108],
    [11.284125, -0.000009, -0.000018],
    [33.000050, 0.000004, 0.000032],
    [25.200008, 0.000015, 0.000008],
    [0.000000, 0.000000, 0.000000],

    [19.746099, -1.480375, -6.000073],
    [11.284138, -0.000015, -0.000012],
    [33.000092, 0.000017, 0.000013],
    [25.199780, 0.000135, 0.000422],
    [0.000000, 0.000000, 0.000000],
]

sk_parents = [
    -1,
    0,
    1,
    2,
    3,
    4,
    0,
    6,
    7,
    8,
    9,
    0,
    11,
    12,
    13,
    14,
    15,
    13,
    17,
    18,
    19,
    20,
    13,
    22,
    23,
    24,
    25,
]

sk_joints_to_remove = [5, 10, 16, 21, 26]

joint_names = [
    "Hips", # 0
    "LeftUpLeg", # 1
    "LeftLeg", # 2
    "LeftFoot", # 3
    "LeftToe", # 4
    "RightUpLeg", # 6
    "RightLeg", # 7
    "RightFoot", # 8
    "RightToe", # 9
    "Spine", # 11
    "Spine1", # 12
    "Spine2", # 13
    "Neck", # 14
    "Head", # 15
    "LeftShoulder", # 17
    "LeftArm", # 18
    "LeftForeArm", # 19
    "LeftHand", # 20
    "RightShoulder", # 22
    "RightArm", # 23
    "RightForeArm", # 24
    "RightHand", # 25
]

def w_last_batch(x):
    return np.concatenate((x[:,:,1:], x[:,:,0:1]), -1)

def w_first_batch(x):
    return np.concatenate((x[:,:,-1:], x[:,:,:-1]), -1)

class Skeleton:
    def __init__(
        self,
        offsets,
        parents,
        joints_left=None,
        joints_right=None,
        bone_length=None,
        device=None,
    ):
        assert len(offsets) == len(parents)

        self._offsets = torch.Tensor(offsets).to(device)
        self._parents = np.array(parents)
        self._joints_left = joints_left
        self._joints_right = joints_right
        self._compute_metadata()

    def num_joints(self):
        return self._offsets.shape[0]

    def offsets(self):
        return self._offsets

    def parents(self):
        return self._parents

    def has_children(self):
        return self._has_children

    def children(self):
        return self._children

    def convert_to_global_pos(self, unit_vec_rerp):
        """
        Convert the unit offset matrix to global position.
        First row(root) will have absolute position value in global coordinates.
        """
        bone_length = self.get_bone_length_weight()
        batch_size = unit_vec_rerp.size(0)
        seq_len = unit_vec_rerp.size(1)
        unit_vec_table = unit_vec_rerp.reshape(batch_size, seq_len, 22, 3)
        global_position = torch.zeros_like(unit_vec_table, device=unit_vec_table.device)

        for i, parent in enumerate(self._parents):
            if parent == -1:  # if root
                global_position[:, :, i] = unit_vec_table[:, :, i]

            else:
                global_position[:, :, i] = global_position[:, :, parent] + (
                    nn.functional.normalize(unit_vec_table[:, :, i], p=2.0, dim=-1)
                    * bone_length[i]
                )

        return global_position

    def convert_to_unit_offset_mat(self, global_position):
        """
        Convert the global position of the skeleton to a unit offset matrix.
        First row(root) will have absolute position value in global coordinates.
        """

        bone_length = self.get_bone_length_weight()
        unit_offset_mat = torch.zeros_like(
            global_position, device=global_position.device
        )

        for i, parent in enumerate(self._parents):

            if parent == -1:  # if root
                unit_offset_mat[:, :, i] = global_position[:, :, i]
            else:
                unit_offset_mat[:, :, i] = (
                    global_position[:, :, i] - global_position[:, :, parent]
                ) / bone_length[i]

        return unit_offset_mat

    def remove_joints(self, joints_to_remove):
        """
        Remove the joints specified in 'joints_to_remove', both from the
        skeleton definition and from the dataset (which is modified in place).
        The rotations of removed joints are propagated along the kinematic chain.
        """
        valid_joints = []
        for joint in range(len(self._parents)):
            if joint not in joints_to_remove:
                valid_joints.append(joint)

        index_offsets = np.zeros(len(self._parents), dtype=int)
        new_parents = []
        for i, parent in enumerate(self._parents):
            if i not in joints_to_remove:
                new_parents.append(parent - index_offsets[parent])
            else:
                index_offsets[i:] += 1
        self._parents = np.array(new_parents)

        self._offsets = self._offsets[valid_joints]
        self._compute_metadata()

    def forward_kinematics(self, rotations, root_positions, names=None):
        """
        Perform forward kinematics using the given trajectory and local rotations.
        Arguments (where N = batch size, L = sequence length, J = number of joints):
         -- rotations: (N, L, J, 4) tensor of unit quaternions describing the local rotations of each joint.
         -- root_positions: (N, L, 3) tensor describing the root joint positions.
        """
        assert len(rotations.shape) == 4
        assert rotations.shape[-1] == 4

        positions_world = []
        rotations_world = []

        expanded_offsets = self._offsets.expand(
            rotations.shape[0],
            rotations.shape[1],
            self._offsets.shape[0],
            self._offsets.shape[1],
        )

        # Parallelize along the batch and time dimensions
        for i in range(self._offsets.shape[0]):
            if self._parents[i] == -1:
                positions_world.append(root_positions)
                rotations_world.append(rotations[:, :, 0])
                print(f"{names[i]} is at {positions_world[-1].cpu().numpy()}")
            else:
                positions_world.append(
                    qrot(rotations_world[self._parents[i]], expanded_offsets[:, :, i])
                    + positions_world[self._parents[i]]
                )
                print(f"{names[i]} is at {positions_world[-1].cpu().numpy()} = {rotations_world[self._parents[i]].cpu().numpy()} * {expanded_offsets[:, :, i].cpu().numpy()} + {positions_world[self._parents[i]].cpu().numpy()}")
                if self._has_children[i]:
                    rotations_world.append(
                        qmul(rotations_world[self._parents[i]], rotations[:, :, i])
                    )
                    print(f"{rotations_world[-1].cpu().numpy()} = {rotations_world[self._parents[i]].cpu().numpy()} * {rotations[:, :, i].cpu().numpy()}")
                else:
                    # This joint is a terminal node -> it would be useless to compute the transformation
                    rotations_world.append(None)

        return torch.stack(positions_world, dim=3).permute(0, 1, 3, 2)
    
    def identity_like(self, x):
        out = torch.zeros_like(x)
        assert out.shape[-1] == 4
        out[..., 0] = 1
        return out
    
    def forward_kinematics_with_rotation(self, rotations, root_positions, root_space=False):
        """
        Perform forward kinematics using the given trajectory and local rotations.
        Arguments (where N = batch size, L = sequence length, J = number of joints):
         -- rotations: (N, L, J, 4) tensor of unit quaternions describing the local rotations of each joint.
         -- root_positions: (N, L, 3) tensor describing the root joint positions.
        """
        assert len(rotations.shape) == 4
        assert rotations.shape[-1] == 4

        positions_world = []
        rotations_world = []

        expanded_offsets = self._offsets.expand(
            rotations.shape[0],
            rotations.shape[1],
            self._offsets.shape[0],
            self._offsets.shape[1],
        ).to(rotations.device)

        # Parallelize along the batch and time dimensions
        for i in range(self._offsets.shape[0]):
            if self._parents[i] == -1:
                positions_world.append(torch.zeros_like(root_positions) if root_space else root_positions)
                rotations_world.append(self.identity_like(rotations[:,:,0]) if root_space else rotations[:, :, 0])
            else:
                positions_world.append(
                    qrot(rotations_world[self._parents[i]], expanded_offsets[:, :, i])
                    + positions_world[self._parents[i]]
                )
                if self._has_children[i]:
                    rotations_world.append(
                        qmul(rotations_world[self._parents[i]], rotations[:, :, i])
                    )
                else:
                    # This joint is a terminal node -> it would be useless to compute the transformation
                    rotations_world.append(
                        torch.Tensor([1, 0, 0, 0])
                        .expand(rotations.shape[0], rotations.shape[1], 4)
                        .to(rotations.device)
                    )

        return torch.stack(positions_world, dim=3).permute(0, 1, 3, 2), torch.stack(
            rotations_world, dim=3
        ).permute(0, 1, 3, 2)

    def get_bone_length_weight(self):
        bone_length = []
        for i, parent in enumerate(self._parents):
            if parent == -1:
                bone_length.append(1)
            else:
                bone_length.append(
                    torch.linalg.norm(self._offsets[i : i + 1], ord="fro").item()
                )
        return torch.Tensor(bone_length)

    def get_root_orientation(self, global_q, reduce=False):
        # batch, sequence, joints, 4
        # get hip rotation
        hips = global_q[:,:,0,:]
        assert len(hips.shape) == 3
        # convert to euler, drop the x, y rotations to project to ground
        flat = hips.reshape((-1, 4))
        euler = R.from_quat(flat).as_euler("xyz")
        if reduce:
            euler[:,:2] = 0
        vec = R.from_euler("xyz", euler).as_rotvec()
        out = vec.reshape((hips.shape[0], hips.shape[1], -1))
        if reduce:
            assert out[:,:,:2].sum() == 0
            return out[:, :, -1] # only one dimension
        else:
            return out
    
    def get_root_w(self, global_q, reduce=False):
        root_orn = self.get_root_orientation(global_q, reduce)
        # batch, sequence
        b, *_ = root_orn.shape
        # finite difference
        diff = root_orn[:, 1:] - root_orn[:, :-1]
        return diff
    
    def joints_left(self):
        return self._joints_left

    def joints_right(self):
        return self._joints_right

    def _compute_metadata(self):
        self._has_children = np.zeros(len(self._parents)).astype(bool)
        for i, parent in enumerate(self._parents):
            if parent != -1:
                self._has_children[parent] = True

        self._children = []
        for i, parent in enumerate(self._parents):
            self._children.append([])
        for i, parent in enumerate(self._parents):
            if parent != -1:
                self._children[parent].append(i)
    
    def global_to_local(self, q):
        assert len(q.shape) == 3
        # q is seq x joints x 4 in global space
        q = w_last_batch(q)
        local_rot = []
        parents = self.parents()
        children = self.has_children()
        for i, (parent, has_child) in enumerate(zip(parents, children)):
            if parent == -1:
                # root is unchanged
                local_rot.append(q[:, i])
            else:
                if has_child:
                    # the local rotation of this link is the difference between the global rotation of
                    # its parent and its own global rotation
                    parent_inv = R.from_quat(q[:,parent]).inv()
                    my_rot = R.from_quat(q[:,i])
                    my_local_rot = (parent_inv * my_rot).as_quat()
                    local_rot.append(my_local_rot)
                else:
                    local_rot.append(q[:,i])
        return w_first_batch(np.array(local_rot).transpose(1,0,2))