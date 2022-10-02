import torch
import torch.nn as nn
import pickle
import os
from sklearn.preprocessing import LabelEncoder
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

class Normalizer:
    def __init__(self, data):
        flat = data.view(-1, data.shape[-1])
        self.maxs = flat.max(0)[0]
        self.mins = flat.min(0)[0]
        if True:
            eps = 1
            for i in range(len(self.mins)):
                if self.mins[i] == self.maxs[i]:
                    print(f'''
                        [ utils/normalization ] Constant data in dimension {i} | '''
                        f'''max = min = {self.maxs[i]}'''
                    )
                    self.mins -= eps
                    self.maxs += eps
        
    def normalize(self, x):
        batch, seq, ch = x.shape
        x = x.view(-1, ch)
        ## [ 0, 1 ]
        x = (x - self.mins) / (self.maxs - self.mins)
        ## [ -1, 1 ]
        x = 2 * x - 1
        x = x.view(batch, seq, ch)
        return x

    def unnormalize(self, x, eps=1e-4):
        '''
            x : [ -1, 1 ]
        '''
        batch, seq, ch = x.shape
        x = x.reshape(-1, ch)
        x = torch.clip(x, -1, 1)

        ## [ -1, 1 ] --> [ 0, 1 ]
        x = (x + 1) / 2.
        return (x * (self.maxs - self.mins) + self.mins).reshape(batch, seq, ch)
    
def replace_constant(minibatch_pose_input, mask_start_frame):

    seq_len = minibatch_pose_input.size(1)
    interpolated = (
        torch.ones_like(minibatch_pose_input, device=minibatch_pose_input.device) * 0.1
    )

    if mask_start_frame == 0 or mask_start_frame == (seq_len - 1):
        interpolate_start = minibatch_pose_input[:, 0, :]
        interpolate_end = minibatch_pose_input[:, seq_len - 1, :]

        interpolated[:, 0, :] = interpolate_start
        interpolated[:, seq_len - 1, :] = interpolate_end

        assert torch.allclose(interpolated[:, 0, :], interpolate_start)
        assert torch.allclose(interpolated[:, seq_len - 1, :], interpolate_end)

    else:
        interpolate_start1 = minibatch_pose_input[:, 0, :]
        interpolate_end1 = minibatch_pose_input[:, mask_start_frame, :]

        interpolate_start2 = minibatch_pose_input[:, mask_start_frame, :]
        interpolate_end2 = minibatch_pose_input[:, seq_len - 1, :]

        interpolated[:, 0, :] = interpolate_start1
        interpolated[:, mask_start_frame, :] = interpolate_end1

        interpolated[:, mask_start_frame, :] = interpolate_start2
        interpolated[:, seq_len - 1, :] = interpolate_end2

        assert torch.allclose(interpolated[:, 0, :], interpolate_start1)
        assert torch.allclose(interpolated[:, mask_start_frame, :], interpolate_end1)

        assert torch.allclose(interpolated[:, mask_start_frame, :], interpolate_start2)
        assert torch.allclose(interpolated[:, seq_len - 1, :], interpolate_end2)
    return interpolated


def slerp(x, y, a):
    """
    Perfroms spherical linear interpolation (SLERP) between x and y, with proportion a

    :param x: quaternion tensor
    :param y: quaternion tensor
    :param a: indicator (between 0 and 1) of completion of the interpolation.
    :return: tensor of interpolation results
    """
    device = x.device
    len = torch.sum(x * y, dim=-1)

    neg = len < 0.0
    len[neg] = -len[neg]
    y[neg] = -y[neg]

    a = torch.zeros_like(x[..., 0]) + a
    amount0 = torch.zeros(a.shape, device=device)
    amount1 = torch.zeros(a.shape, device=device)

    linear = (1.0 - len) < 0.01
    omegas = torch.arccos(len[~linear])
    sinoms = torch.sin(omegas)

    amount0[linear] = 1.0 - a[linear]
    amount0[~linear] = torch.sin((1.0 - a[~linear]) * omegas) / sinoms

    amount1[linear] = a[linear]
    amount1[~linear] = torch.sin(a[~linear] * omegas) / sinoms
    # res = amount0[..., np.newaxis] * x + amount1[..., np.newaxis] * y
    res = amount0.unsqueeze(3) * x + amount1.unsqueeze(3) * y

    return res


def slerp_input_repr(minibatch_pose_input, mask_start_frame):
    seq_len = minibatch_pose_input.size(1)
    minibatch_pose_input = minibatch_pose_input.reshape(
        minibatch_pose_input.size(0), seq_len, -1, 4
    )
    interpolated = torch.zeros_like(
        minibatch_pose_input, device=minibatch_pose_input.device
    )

    if mask_start_frame == 0 or mask_start_frame == (seq_len - 1):
        interpolate_start = minibatch_pose_input[:, 0:1]
        interpolate_end = minibatch_pose_input[:, seq_len - 1 :]

        for i in range(seq_len):
            dt = 1 / (seq_len - 1)
            interpolated[:, i : i + 1, :] = slerp(
                interpolate_start, interpolate_end, dt * i
            )

        assert torch.allclose(interpolated[:, 0:1], interpolate_start)
        assert torch.allclose(interpolated[:, seq_len - 1 :], interpolate_end)
    else:
        interpolate_start1 = minibatch_pose_input[:, 0:1]
        interpolate_end1 = minibatch_pose_input[
            :, mask_start_frame : mask_start_frame + 1
        ]

        interpolate_start2 = minibatch_pose_input[
            :, mask_start_frame : mask_start_frame + 1
        ]
        interpolate_end2 = minibatch_pose_input[:, seq_len - 1 :]

        for i in range(mask_start_frame + 1):
            dt = 1 / mask_start_frame
            interpolated[:, i : i + 1, :] = slerp(
                interpolate_start1, interpolate_end1, dt * i
            )

        assert torch.allclose(interpolated[:, 0:1], interpolate_start1)
        assert torch.allclose(
            interpolated[:, mask_start_frame : mask_start_frame + 1], interpolate_end1
        )

        for i in range(mask_start_frame, seq_len):
            dt = 1 / (seq_len - mask_start_frame - 1)
            interpolated[:, i : i + 1, :] = slerp(
                interpolate_start2, interpolate_end2, dt * (i - mask_start_frame)
            )

        assert torch.allclose(
            interpolated[:, mask_start_frame : mask_start_frame + 1], interpolate_start2
        )
        assert torch.allclose(interpolated[:, seq_len - 1 :], interpolate_end2)

    interpolated = torch.nn.functional.normalize(interpolated, p=2.0, dim=3)
    return interpolated.reshape(minibatch_pose_input.size(0), seq_len, -1)


def lerp_input_repr(minibatch_pose_input, mask_start_frame):
    seq_len = minibatch_pose_input.size(1)
    interpolated = torch.zeros_like(
        minibatch_pose_input, device=minibatch_pose_input.device
    )

    if mask_start_frame == 0 or mask_start_frame == (seq_len - 1):
        interpolate_start = minibatch_pose_input[:, 0, :]
        interpolate_end = minibatch_pose_input[:, seq_len - 1, :]

        for i in range(seq_len):
            dt = 1 / (seq_len - 1)
            interpolated[:, i, :] = torch.lerp(
                interpolate_start, interpolate_end, dt * i
            )

        assert torch.allclose(interpolated[:, 0, :], interpolate_start)
        assert torch.allclose(interpolated[:, seq_len - 1, :], interpolate_end)
    else:
        interpolate_start1 = minibatch_pose_input[:, 0, :]
        interpolate_end1 = minibatch_pose_input[:, mask_start_frame, :]

        interpolate_start2 = minibatch_pose_input[:, mask_start_frame, :]
        interpolate_end2 = minibatch_pose_input[:, -1, :]

        for i in range(mask_start_frame + 1):
            dt = 1 / mask_start_frame
            interpolated[:, i, :] = torch.lerp(
                interpolate_start1, interpolate_end1, dt * i
            )

        assert torch.allclose(interpolated[:, 0, :], interpolate_start1)
        assert torch.allclose(interpolated[:, mask_start_frame, :], interpolate_end1)

        for i in range(mask_start_frame, seq_len):
            dt = 1 / (seq_len - mask_start_frame - 1)
            interpolated[:, i, :] = torch.lerp(
                interpolate_start2, interpolate_end2, dt * (i - mask_start_frame)
            )

        assert torch.allclose(interpolated[:, mask_start_frame, :], interpolate_start2)
        assert torch.allclose(interpolated[:, -1, :], interpolate_end2)
    return interpolated


def vectorize_representation(global_position, global_rotation):

    batch_size = global_position.shape[0]
    seq_len = global_position.shape[1]

    global_pos_vec = global_position.reshape(batch_size, seq_len, -1).contiguous()
    global_rot_vec = global_rotation.reshape(batch_size, seq_len, -1).contiguous()

    global_pose_vec_gt = torch.cat([global_pos_vec, global_rot_vec], dim=2)
    return global_pose_vec_gt

def vectorize_many(data):
    # given a list of batch x seqlen x joints? x channels, flatten all to batch x seqlen x -1, concatenate
    batch_size = data[0].shape[0]
    seq_len = data[0].shape[1]

    out = [x.reshape(batch_size, seq_len, -1).contiguous() for x in data]

    global_pose_vec_gt = torch.cat(out, dim=2)
    return global_pose_vec_gt

def get_first_last_mask(posq_batch, start_width=1, end_width=1):
    # an array in batch x seq_len x (3+4)*joint_num format
    # return a mask that is ones in the first and last row (or first/last WIDTH rows) in the sequence direction
    mask = torch.zeros_like(posq_batch)
    mask[:,:start_width,:] = 1
    mask[:,-end_width:,:] = 1
    return mask

def get_root_mask(posq_batch):
    # an array in batch x seq_len x (3+4)*joint_num format
    # return a mask that is ones in the first and last row (or first/last WIDTH rows) in the sequence direction
    mask = torch.zeros_like(posq_batch)
    mask[:,:,get_root_idx()] = 1
    return mask

def get_head_mask(posq_batch):
    # an array in batch x seq_len x (3+4)*joint_num format
    # return a mask that is ones in the first and last row (or first/last WIDTH rows) in the sequence direction
    mask = torch.zeros_like(posq_batch)
    mask[:,:,get_head_idx()] = 1
    return mask

def get_first_mask(posq_batch, width=1):
    # an array in batch x seq_len x (3+4)*joint_num format
    # return a mask that is ones in the first and last row in the sequence direction
    mask = torch.zeros_like(posq_batch)
    mask[:,:width,:] = 1
    return mask

def get_first_last_middle_mask(posq_batch, end_width=10, middle_width=10):
    # an array in batch x seq_len x (3+4)*joint_num format
    # return a mask that is ones in the first and last row (or first/last WIDTH rows) in the sequence direction
    mask = torch.zeros_like(posq_batch)
    mask[:,:end_width,:] = 1
    mask[:,-end_width:,:] = 1
    # middle part
    total_width = mask.shape[1]
    middle_end_width = (total_width - 2 * end_width - middle_width) // 2
    mask[:,end_width + middle_end_width:-end_width-middle_end_width,:] = 1
    return mask

def get_upper_idx():
    out = list(range(3*9, 66)) + list(range(66 + 4*9, 66 + 88)) + list(range(0,3)) + list(range(66, 66+4))
    return np.array(out)

def get_root_idx():
    out = list(range(0,3)) + list(range(66, 66+4))
    return np.array(out)

def get_head_idx():
    out = list(range(4 + 13 * 3, 4 + 13 * 3 + 3)) + list(range(4 + 66 + 13 * 4, 4 + 66 + 13 * 4 + 4))
    return np.array(out)
    
def get_upperbody_mask(posq_batch):
    # an array in batch x seq_len x (3+4)*joint_num format
    # return a mask that is ones in the first and last row (or first/last WIDTH rows) in the sequence direction
    mask = torch.zeros_like(posq_batch)
    mask[:,:,get_upper_idx()] = 1
    return mask

def get_phase_mask(posq_batch):
    # an array in batch x seq_len x (3+4)*joint_num format
    # return a mask that is ones in the first and last row (or first/last WIDTH rows) in the sequence direction
    mask = torch.zeros_like(posq_batch)
    mask[:,:,-20:] = 1
    return mask

def get_lower_idx():
    out = list(range(0, 3*9)) + list(range(66, 66 + 4*9))
    return np.array(out)
    
def get_lowerbody_mask(posq_batch):
    # an array in batch x seq_len x (3+4)*joint_num format
    # return a mask that is ones in the first and last row (or first/last WIDTH rows) in the sequence direction
    mask = torch.zeros_like(posq_batch)
    mask[:,:,get_lower_idx()] = 1
    return mask

def get_dataset(lafan_dataset, from_idx, target_idx, device, skeleton_mocap, wdir, save_dir, train, phase=False):
    horizon = target_idx - from_idx
    print(f"Horizon: {horizon}")
    
    root_pos = torch.Tensor(lafan_dataset.data['root_p'][:, from_idx:target_idx+1]).to(device)
    local_q = torch.Tensor(lafan_dataset.data['local_q'][:, from_idx:target_idx+1]).to(device)
    root_v = torch.Tensor(lafan_dataset.data['root_v'][:, from_idx:target_idx+1]).to(device)
    # contact data
    contact_data = torch.Tensor(lafan_dataset.data['contact'][:, from_idx:target_idx+1]).to(device)
    local_q_normalized = nn.functional.normalize(local_q, p=2.0, dim=-1)

    global_pos, global_q = skeleton_mocap.forward_kinematics_with_rotation(local_q_normalized, root_pos)
    b, s, c = root_v.shape
    
    global_pos = global_pos[:, :horizon, :, :]
    global_q = global_q[:, :horizon, :, :]
    
    # now, flatten everything into: batch x sequence x [...]
    l = [contact_data, global_pos, global_q]
    if phase:
        # phase data
        phase_data = torch.Tensor(lafan_dataset.data['phase'][:, from_idx:target_idx+1]).to(device)
        l.append(phase_data)
    global_pose_vec_gt = vectorize_many(l).float()
    global_pose_vec_input = global_pose_vec_gt.clone().detach()
    
    # normalize the data. Both train and test need the same normalizer.
    if train == "train":
        normalizer = Normalizer(global_pose_vec_input)
        pickle.dump(normalizer, open(os.path.join("./", f"{train}_normalizer.pkl"), "wb"))
    else:
        normalizer = pickle.load(open(os.path.join("./", f"train_normalizer.pkl"), "rb"))
    global_pose_vec_input = normalizer.normalize(global_pose_vec_input)
    # create conditioning channel. For now, it's always first and last full pose.
    cond_mask = get_first_last_mask(global_pose_vec_input)
    
    # get bone lengths in normalized space
    normalized_bone_lengths = get_bone_lengths(global_pose_vec_input[:,:,:66], skeleton_mocap)
    
    
    assert not torch.isnan(global_pose_vec_input).any()
    # batch x seq_len x features
    
    seq_categories = [x[:-1] for x in lafan_dataset.data['seq_names']]
    
    # need same labelencoder across train/test too
    if train == "train":
        le = LabelEncoder()
        # encode y to int labels
        # add a "unconditional" label for classifer-free guidance
        seq_categories_with_unc = seq_categories + ["unconditional"]
        # fit with extra label
        le.fit(seq_categories_with_unc)
        pickle.dump(le, open(os.path.join("./", f"{train}_le.pkl"), "wb"))
    else:
        le = pickle.load(open(os.path.join("./", f"train_le.pkl"), "rb"))
    
    UNC_LABEL = le.transform(["unconditional"])[0]
    print(global_pose_vec_input.shape)
    
    # transform the original set without the extra label
    le_np = le.transform(seq_categories)
    seq_labels = torch.Tensor(le_np).type(torch.int64).unsqueeze(1).to(device)
    np.save(f'{save_dir}/{train}_le_classes_.npy', le.classes_)
    num_labels = len(le.classes_)

    tensor_dataset = TensorDataset(global_pose_vec_input, seq_labels, cond_mask, normalized_bone_lengths)
    return tensor_dataset, UNC_LABEL, le, normalizer, horizon, num_labels