import numpy as np
from typing import List

import torch

class Skeleton:
    """Base skeleton and forward kinematics."""
    def __init__(self):
        self.topology = []
        self.offsets = []

    def forward_kinematics(self, root_pos, orientations, to_smpl=True, to_mvnx=True):   
        r"""
        Forward kinematic for a batch of orientations, in rotation matrix.
        :param orientations: global orientation in smpl frame, predicted by the network,
                             must transform back first.
        :param to_smpl: transform bvh positions to smpl frame, default to True.
        :return: positions in smpl frame.
        """
        device = orientations.device
        if to_mvnx:
            G = torch.tensor([[0, 1, 0], [0, 0, 1], [1, 0, 0]], \
                dtype=torch.float32).repeat(orientations.shape[0], 1, 1).unsqueeze(1).to(device)             
            orientations = G.transpose(2, 3).matmul(orientations).matmul(G)
                
        # positions (num_segments, N, 3)
        positions = torch.zeros([len(self.offsets), orientations.shape[0], 3], dtype=torch.float32).to(device)
        positions[0] = root_pos
        if not to_smpl and not to_mvnx:
            abs_orientations = torch.zeros_like(orientations) # N, num_segments, 3, 3
            abs_orientations[:, 0] = orientations[:, 0]
        
        topology = self.topology[:orientations.shape[1]]
        for i, parent_indices in enumerate(topology):
            if parent_indices == -1:
                continue
            else:            
                x_B = self.offsets[i].to(device) # (3,)the offset for this bone
                x_B = x_B.view(1, -1, 1).repeat(orientations.shape[0], 1, 1) # (N, 3, 1) so that it can be broadcasted to (N, 3, 3)

                R_GB = orientations[:, parent_indices] # N, 3, 3
                
                # start from the parent's position, and apply the rotation to the bone to get the bone position
                positions[i] = (positions[parent_indices].to(device) + R_GB.bmm(x_B).squeeze(2))
                if not to_smpl and not to_mvnx:
                    abs_orientations[:, i] = torch.bmm(abs_orientations[:, parent_indices], R_GB)
        if to_smpl:
            R = torch.tensor([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=torch.float32).to(device)
            positions = R.matmul(positions.permute(1, 2, 0).contiguous()).transpose(1, 2)
            return positions
        else:
            if not to_mvnx:
                return positions.permute(1, 0, 2), abs_orientations
            return positions.permute(1, 0, 2) # (N, num_segments, 3)
            
            
class XsensSkeleton(Skeleton):
    # Reference: https://github.com/dx118/dynaip/blob/ddfff5b88b77e124744bc74a792967cb8bad6735/utils/skeleton.py#L12
    """Skeleton defination and forward kinematics for Xsens."""

    def __init__(self, offsets=None):
        """
        Initialize the skeleton, using segment lengths from Andy dataset,
        extracted from bvh files.
        
        segments = ["Pelvis", "L5", "L3", "T12", "T8", "Neck", "Head",
                    "RightShoulder", "RightUpperArm", "RightForeArm", "RightHand",
                    "LeftShoulder", "LeftUpperArm", "LeftForeArm", "LeftHand",
                    "RightUpperLeg", "RightLowerLeg", "RightFoot", "RightToe",
                    "LeftUpperLeg", "LeftLowerLeg", "LeftFoot", "LeftToe"]
                    
        parents = [None, "Pelvis", "L5", "L3", "T12", "T8", "Neck",
                   "T8", "RightShoulder", "RightUpperArm", "RightForeArm",
                   "T8", "LeftShoulder", "LeftUpperArm", "LeftForeArm",
                   "Pelvis", "RightUpperLeg", "RightLowerLeg", "RightFoot",
                   "Pelvis", "LeftUpperLeg", "LeftLowerLeg", "LeftFoot"]    
        """

        self.offsets =  torch.tensor([[ 0.0000e+00,  0.0000e+00,  0.0000e+00],
                                        [-1.1913e-04,  0.0000e+00,  1.0651e-01],
                                        [ 2.1639e-04,  0.0000e+00,  1.0562e-01],
                                        [ 0.0000e+00,  0.0000e+00,  9.5423e-02],
                                        [ 0.0000e+00,  0.0000e+00,  9.5423e-02],
                                        [ 0.0000e+00,  0.0000e+00,  1.3356e-01],
                                        [ 6.4870e-04,  0.0000e+00,  8.7248e-02],
                                        [ 0.0000e+00, -2.9223e-02,  7.4656e-02],
                                        [ 0.0000e+00, -1.4086e-01,  0.0000e+00],
                                        [ 0.0000e+00, -2.9541e-01,  0.0000e+00],
                                        [ 4.4812e-05, -2.4263e-01,  0.0000e+00], # 10 RightHand
                                        [ 0.0000e+00,  2.9223e-02,  7.4656e-02],
                                        [ 0.0000e+00,  1.4086e-01,  0.0000e+00],
                                        [ 0.0000e+00,  2.9541e-01,  0.0000e+00],
                                        [ 4.4812e-05,  2.4263e-01,  0.0000e+00], # 14 LeftHand
                                        [ 5.9564e-05, -8.7448e-02,  5.8373e-04],
                                        [ 4.6875e-05,  0.0000e+00, -4.6706e-01],
                                        [-1.3225e-04,  0.0000e+00, -4.1930e-01], # 17 RightLowerLeg
                                        [ 1.6630e-01,  0.0000e+00, -1.0138e-01], # 18 RightFoot
                                        [ 5.9564e-05,  8.7448e-02,  5.8373e-04],
                                        [ 4.6875e-05,  0.0000e+00, -4.6706e-01],
                                        [-1.3225e-04,  0.0000e+00, -4.1930e-01], # 21 LeftLowerLeg
                                        [ 1.6630e-01,  0.0000e+00, -1.0138e-01]]) # 22 LeftFoot
        
        
        if offsets is not None:
            assert offsets.shape == self.offsets.shape
            self.offsets = offsets
        
        self.topology = XSensConstants.kintree_parents
        self.connections = []
        for i, pi in enumerate(self.topology):
            if pi >= 0:
                self.connections.append([pi, i])             
                                               
    def forward_kinematics(self, root_pos, orientation, to_smpl=True, to_mvnx=True):
        positions = super().forward_kinematics(root_pos, orientation, to_smpl, to_mvnx)
        return positions

class XSensConstants:
    """
    \brief Transformations segment_tXYZ and segment_qWXYZ are defined as
           from XSens segment/part coordinates to XSens world coordinates.
           See XSens manual for details.
    """

    num_parts: int = 23
    upper_body_num_parts: int = 15
    num_bones: int = 22
    num_sensors: int = 17
    k_timestamps_us: str = "timestamps_us"
    k_frame_count: str = "frameCount"
    k_framerate: str = "frameRate"
    k_part_tXYZ: str = "segment_tXYZ"
    k_part_qWXYZ: str = "segment_qWXYZ"
    k_ipose_part_tXYZ: str = "identity_segment_tXYZ"
    k_ipose_part_qWXYZ: str = "identity_segment_qWXYZ"
    k_tpose_part_tXYZ: str = "tpose_segment_tXYZ"
    k_tpose_part_qWXYZ: str = "tpose_segment_qWXYZ"
    k_foot_contacts: str = "foot_contacts"
    k_sensor_tXYZ: str = "sensor_tXYZ"
    k_sensor_qWXYZ: str = "sensor_qWXYZ"
    part_names = [
        "Pelvis",
        "L5",
        "L3",
        "T12",
        "T8",
        "Neck",
        "Head",
        "R_Shoulder",
        "R_UpperArm",
        "R_Forearm",
        "R_Hand",
        "L_Shoulder",
        "L_UpperArm",
        "L_Forearm",
        "L_Hand",
        "R_UpperLeg",
        "R_LowerLeg",
        "R_Foot",
        "R_Toe",
        "L_UpperLeg",
        "L_LowerLeg",
        "L_Foot",
        "L_Toe",
    ]  # num = 23
    kintree_parents: List[int] = [
        -1,
        0,
        1,
        2,
        3,
        4,
        5,
        4,
        7,
        8,
        9,
        4,
        11,
        12,
        13,
        0,
        15,
        16,
        17,
        0,
        19,
        20,
        21,
    ]  # num = 23
    sensor_names: List[int] = [
        "Pelvis",
        "T8",
        "Head",
        "RightShoulder",
        "RightUpperArm",
        "RightForeArm",
        "RightHand",
        "LeftShoulder",
        "LeftUpperArm",
        "LeftForeArm",
        "LeftHand",
        "RightUpperLeg",
        "RightLowerLeg",
        "RightFoot",
        "LeftUpperLeg",
        "LeftLowerLeg",
        "LeftFoot",
    ]  # num = 17
    color_skeleton = np.array(
        [
            [0, 0, 0],
            [127, 0, 255],
            [105, 34, 254],
            [81, 71, 252],
            [59, 103, 249],
            [35, 136, 244],
            [11, 167, 238],
            [10, 191, 232],
            [34, 214, 223],
            [58, 232, 214],
            [80, 244, 204],
            [104, 252, 192],
            [128, 254, 179],
            [150, 252, 167],
            [174, 244, 152],
            [196, 232, 138],
            [220, 214, 122],
            [244, 191, 105],
            [255, 167, 89],
            [255, 136, 71],
            [255, 103, 53],
            [255, 71, 36],
            [255, 34, 17],
        ]
    )
    leaf_parts=[
        "Pelvis",
        "Head",
        "R_Hand",
        "L_Hand",
    ]
    leaf_indices = [
        0,
        6,
        10,
        14,
    ]
    intermediate_parts=[
        "L5",
        "L3",
        "T12",
        "T8",
        "Neck",
        "R_Shoulder",
        "R_UpperArm",
        "R_Forearm",
        "L_Shoulder",
        "L_UpperArm",
        "L_Forearm",
    ]
    intermediate_indices = [
        1,
        2,
        3,
        4,
        5,
        7,
        8,
        9,
        11,
        12,
        13
    ]

DEFAULT_GOAL_BODY_PARTS = ["Pelvis", "Head", "R_Hand", "L_Hand"]
GOAL_BODY_PART_COLORS = {"Pelvis": "red", "Head": "green", "R_Hand": "blue", "L_Hand": "yellow"}