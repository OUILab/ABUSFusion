import torch
import torch.nn as nn
from mamba_ssm import Mamba
from torchvision.models.optical_flow import (Raft_Large_Weights,
                                             Raft_Small_Weights, raft_large,
                                             raft_small)


class RAFTMamba(nn.Module):
    def __init__(
        self, input_channels=1, input_height=1000, input_width=657, use_small_raft=False
    ):
        super(RAFTMamba, self).__init__()

        # Initialize RAFT model from torchvision
        if use_small_raft:
            self.raft = raft_small(weights=Raft_Small_Weights.DEFAULT)
        else:
            self.raft = raft_large(weights=Raft_Large_Weights.DEFAULT)

        # Freeze RAFT parameters
        for param in self.raft.parameters():
            param.requires_grad = False

        self.mamba = Mamba(d_model=128, d_state=16, d_conv=4, expand=2)

        raft_output_size = 256  # This may need to be adjusted based on RAFT output
        imu_feature_size = 6  # 6 IMU features

        self.fc_raft = nn.Linear(raft_output_size, 64)
        self.fc_imu = nn.Linear(imu_feature_size, 64)
        self.fc_out = nn.Linear(128, 6)  # Output: tx, ty, tz, ex, ey, ez

    def forward(self, frames, imu_data):
        batch_size, seq_len, channels, height, width = frames.shape

        # Process frame pairs with RAFT
        raft_features = []
        for i in range(seq_len - 1):
            # RAFT expects inputs in [0, 255] range
            frame1 = frames[:, i] * 255
            frame2 = frames[:, i + 1] * 255

            # Ensure frames have 3 channels
            if channels == 1:
                frame1 = frame1.repeat(1, 3, 1, 1)
                frame2 = frame2.repeat(1, 3, 1, 1)

            flow = self.raft(frame1, frame2)[
                -1
            ]  # Get the last (finest) flow prediction
            raft_features.append(flow.view(batch_size, -1))
        raft_features = torch.stack(raft_features, dim=1)

        # Process RAFT features
        raft_features = self.fc_raft(raft_features)

        # Process IMU data
        imu_features = self.fc_imu(
            imu_data[:, :-1]
        )  # Exclude the last IMU data point to match RAFT features

        # Combine RAFT and IMU features
        combined_features = torch.cat([raft_features, imu_features], dim=-1)

        # Process with Mamba
        mamba_out = self.mamba(combined_features)

        # Final prediction
        transformations = self.fc_out(mamba_out)

        return transformations
