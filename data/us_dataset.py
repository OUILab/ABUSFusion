import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from torchvision import transforms
from scipy.spatial.transform import Rotation


class UnifiedUltrasoundDataset(Dataset):
    def __init__(self, file_paths, sequence_length=10, downsample_factor=3):
        self.file_paths = [file_paths] if isinstance(file_paths, str) else file_paths
        self.sequence_length = sequence_length
        self.downsample_factor = downsample_factor

        self.dfs = [pd.read_hdf(file_path) for file_path in self.file_paths]
        self.frame_shape = self.dfs[0]["frame"].iloc[0].shape

        self.indices = self._create_indices()

        self.resize = transforms.Resize(
            (
                self.frame_shape[0] // self.downsample_factor,
                self.frame_shape[1] // self.downsample_factor,
            )
        )

    def _create_indices(self):
        indices = []
        for session, df in enumerate(self.dfs):
            session_length = len(df)
            indices.extend(
                [(session, i) for i in range(session_length - self.sequence_length)]
            )
        return indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        session, start_idx = self.indices[idx]
        df = self.dfs[session]
        sequence = df.iloc[start_idx : start_idx + self.sequence_length]

        frames = torch.tensor(np.stack(sequence["frame"].values)).float()
        frames = frames.mean(dim=-1, keepdim=True)  # Convert to grayscale
        frames = frames.permute(
            0, 3, 1, 2
        )  # (sequence_length, channels, height, width)
        frames = torch.stack([self.resize(frame) for frame in frames])

        imu_columns = [
            "imu_acc_x",
            "imu_acc_y",
            "imu_acc_z",
            "imu_orientation_x",
            "imu_orientation_y",
            "imu_orientation_z",
        ]
        imu_data = torch.tensor(sequence[imu_columns].values).float()

        # Get the transformation for the last frame relative to the first frame
        first_frame = sequence.iloc[0]
        last_frame = sequence.iloc[-1]
        target = self._get_relative_transform(first_frame, last_frame)

        return frames, imu_data, target

    def _get_relative_transform(self, first_frame, last_frame):
        def get_transform_matrix(frame):
            pos = [frame["ot_pos_x"], frame["ot_pos_y"], frame["ot_pos_z"]]
            quat = [frame["ot_qw"], frame["ot_qx"], frame["ot_qy"], frame["ot_qz"]]
            rot = Rotation.from_quat(quat).as_matrix()
            transform = np.eye(4)
            transform[:3, :3] = rot
            transform[:3, 3] = pos
            return transform

        T_first = get_transform_matrix(first_frame)
        T_last = get_transform_matrix(last_frame)

        T_relative = np.linalg.inv(T_first) @ T_last

        # Extract translation
        translation = T_relative[:3, 3]

        # Extract rotation and convert to Euler angles
        rotation = Rotation.from_matrix(T_relative[:3, :3]).as_euler("xyz")

        # Combine translation and rotation into a 6-DoF vector
        transform = np.concatenate([translation, rotation])

        return torch.tensor(transform).float()
