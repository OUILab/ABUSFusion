# %% [markdown]
# # Ultrasound Reconstruction with IMU and Optical Tracker Data
#

# %% [markdown]
# ## 1. Import Libraries
#

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import DataLoader, Dataset
from tqdm.notebook import tqdm

# %% [markdown]
# ## 2. Load and Preprocess Data
#

# %%
file_path = "/home/varun/xia_lab/repos/ABUSFusion/scans/20240826/wrist_data.h5"
df = pd.read_hdf(file_path)

# %% [markdown]
# ## 3. Create Dataset and DataLoader
#


# %%
class QuaternionToEulerTransform:
    def __call__(self, sample):
        frames, imu_data, ot_data = sample

        ot_pos = ot_data[..., :3]
        ot_quat = ot_data[..., 3:]

        w, x, y, z = ot_quat.unbind(-1)

        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x * x + y * y)
        ex = torch.atan2(t0, t1)

        t2 = 2.0 * (w * y - z * x)
        t2 = torch.clamp(t2, -1.0, 1.0)
        ey = torch.asin(t2)

        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (y * y + z * z)
        ez = torch.atan2(t3, t4)

        ot_euler = torch.stack([ex, ey, ez], dim=-1)
        target = torch.cat([ot_pos, ot_euler], dim=-1)

        return frames, imu_data, target


# %%
class UltrasoundSequenceDataset(Dataset):
    def __init__(self, df, sequence_length=10, transform=None):
        self.df = df
        self.sequence_length = sequence_length
        self.transform = transform

    def __len__(self):
        return len(self.df) - self.sequence_length + 1

    def __getitem__(self, idx):
        sequence = self.df.iloc[idx : idx + self.sequence_length]

        frames = torch.tensor(np.stack(sequence["frame"].values)).float()
        frames = frames.mean(dim=-1, keepdim=True)  # Convert RGB to grayscale
        frames = frames.permute(
            0, 3, 1, 2
        )  # Change to (sequence_length, channels, height, width)

        imu_data = torch.tensor(
            sequence[
                [
                    "imu_acc_x",
                    "imu_acc_y",
                    "imu_acc_z",
                    "imu_orientation_x",
                    "imu_orientation_y",
                    "imu_orientation_z",
                ]
            ].values
        ).float()
        ot_data = torch.tensor(
            sequence[
                ["ot_pos_x", "ot_pos_y", "ot_pos_z", "ot_qw", "ot_qx", "ot_qy", "ot_qz"]
            ].values
        ).float()

        sample = (frames, imu_data, ot_data)

        if self.transform:
            sample = self.transform(sample)

        return sample


# %%
class UltrasoundPairDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df) - 1

    def __getitem__(self, idx):
        pair = self.df.iloc[idx : idx + 2]

        frames = torch.tensor(np.stack(pair["frame"].values)).float()
        frames = frames.mean(dim=-1, keepdim=True)  # Convert RGB to grayscale
        frames = frames.permute(0, 3, 1, 2)  # Change to (2, channels, height, width)

        imu_columns = [
            "imu_acc_x",
            "imu_acc_y",
            "imu_acc_z",
            "imu_orientation_x",
            "imu_orientation_y",
            "imu_orientation_z",
        ]
        imu_data = torch.tensor(pair.iloc[1][imu_columns].astype(float).values).float()

        ot_columns = [
            "ot_pos_x",
            "ot_pos_y",
            "ot_pos_z",
            "ot_qw",
            "ot_qx",
            "ot_qy",
            "ot_qz",
        ]
        ot_data = torch.tensor(pair.iloc[1][ot_columns].astype(float).values).float()

        sample = (frames, imu_data, ot_data)

        if self.transform:
            sample = self.transform(sample)

        return sample


# %%
# Create datasets and dataloaders
transform = QuaternionToEulerTransform()
# sequence_dataset = UltrasoundSequenceDataset(
#     df, sequence_length=10, transform=transform
# )
# sequence_loader = DataLoader(sequence_dataset, batch_size=1, shuffle=False)


pair_dataset = UltrasoundPairDataset(df, transform=transform)
pair_loader = DataLoader(pair_dataset, batch_size=1, shuffle=False)

# %%
sample = pair_dataset[0]
print(f"Frames shape: {sample[0].shape}")
print(f"IMU data shape: {sample[1].shape}")
print(f"OT data shape: {sample[2].shape}")

# %% [markdown]
# ## 4. Define Model Architecture
#


# %%
class UltrasoundSequenceModel(nn.Module):
    def __init__(
        self, sequence_length, input_channels=1, input_height=1000, input_width=657
    ):
        super(UltrasoundSequenceModel, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Calculate the size of the feature map after convolutions and pooling
        conv_output_height = (
            input_height // 4
        )  # Two pooling operations, each reducing size by half
        conv_output_width = input_width // 4

        self.lstm = nn.LSTM(
            64 * conv_output_height * conv_output_width + 6, 128, batch_first=True
        )
        self.fc = nn.Linear(128, 6)  # Output: tx, ty, tz, ex, ey, ez

    def forward(self, frames, imu_data):
        batch_size, seq_len, _, height, width = frames.shape

        # Process each frame independently
        frame_features = []
        for i in range(seq_len):
            x = self.pool(torch.relu(self.conv1(frames[:, i, :, :, :])))
            x = self.pool(torch.relu(self.conv2(x)))
            frame_features.append(x.view(batch_size, -1))

        # Combine frame features with IMU data
        combined_features = torch.cat(
            [torch.stack(frame_features, dim=1), imu_data], dim=2
        )

        # Process sequence with LSTM
        lstm_out, _ = self.lstm(combined_features)

        # Predict transformation for each frame in the sequence
        transformations = self.fc(lstm_out)

        return transformations


# %%
class UltrasoundPairModel(nn.Module):
    def __init__(self, input_channels=1, input_height=1000, input_width=657):
        super(UltrasoundPairModel, self).__init__()
        self.conv1 = nn.Conv2d(input_channels * 2, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        conv_output_height = input_height // 4
        conv_output_width = input_width // 4

        self.fc1 = nn.Linear(64 * conv_output_height * conv_output_width + 6, 128)
        self.fc2 = nn.Linear(128, 6)  # Output: tx, ty, tz, ex, ey, ez

    def forward(self, frames, imu_data):
        x = torch.cat([frames[:, 0, :, :, :], frames[:, 1, :, :, :]], dim=1)
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)

        combined = torch.cat([x, imu_data], dim=1)
        x = torch.relu(self.fc1(combined))
        transformations = self.fc2(x)

        return transformations


# %%
# sequence_model = UltrasoundSequenceModel(sequence_length=10, input_channels=1, input_height=1000, input_width=657)
pair_model = UltrasoundPairModel(input_channels=1, input_height=1000, input_width=657)

# %% [markdown]
# ## 5. Train Model
#

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# sequence_model = sequence_model.to(device)
pair_model = pair_model.to(device)
criterion = nn.MSELoss()
# sequence_optimizer = optim.Adam(sequence_model.parameters(), lr=0.001)
pair_optimizer = optim.Adam(pair_model.parameters(), lr=0.001)


# %%
def train_sequence_model(model, dataloader, optimizer, num_epochs=100):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for frames, imu_data, targets in dataloader:
            frames, imu_data, targets = (
                frames.to(device),
                imu_data.to(device),
                targets.to(device),
            )

            optimizer.zero_grad()
            outputs = model(frames, imu_data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(
            f"Sequence Model - Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}"
        )


# %%
def train_pair_model(model, dataloader, optimizer, num_epochs=100):
    for epoch in tqdm(range(num_epochs), desc="Epochs", position=0):
        model.train()
        total_loss = 0
        epoch_progress = tqdm(
            dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", position=1, leave=False
        )

        for frames, imu_data, targets in epoch_progress:
            frames, imu_data, targets = (
                frames.to(device),
                imu_data.to(device),
                targets.to(device),
            )

            optimizer.zero_grad()
            outputs = model(frames, imu_data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            avg_loss = total_loss / (epoch_progress.n + 1)  # Current average loss

            # Update the progress bar with the current average loss
            epoch_progress.set_postfix({"Avg Loss": f"{avg_loss:.4f}"})

        # Update the epoch progress bar with the final average loss
        tqdm.write(
            f"Pair Model - Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}"
        )


# %%
# Train both models
# train_sequence_model(sequence_model, sequence_loader, sequence_optimizer)
train_pair_model(pair_model, pair_loader, pair_optimizer)

# %% [markdown]
# ## 6. Apply Transform and Reconstruct Volume
#


# %%
def apply_transform(frame, tx, ty, tz, ex, ey, ez):
    # Create transformation matrix
    R_x = np.array(
        [[1, 0, 0], [0, np.cos(ex), -np.sin(ex)], [0, np.sin(ex), np.cos(ex)]]
    )

    R_y = np.array(
        [[np.cos(ey), 0, np.sin(ey)], [0, 1, 0], [-np.sin(ey), 0, np.cos(ey)]]
    )

    R_z = np.array(
        [[np.cos(ez), -np.sin(ez), 0], [np.sin(ez), np.cos(ez), 0], [0, 0, 1]]
    )

    R = np.dot(R_z, np.dot(R_y, R_x))

    T = np.array(
        [
            [R[0, 0], R[0, 1], R[0, 2], tx],
            [R[1, 0], R[1, 1], R[1, 2], ty],
            [R[2, 0], R[2, 1], R[2, 2], tz],
            [0, 0, 0, 1],
        ]
    )

    # Apply transformation to each pixel
    h, w = frame.shape
    transformed_frame = np.zeros_like(frame)
    for i in range(h):
        for j in range(w):
            p = np.array([i, j, 0, 1])
            p_transformed = np.dot(T, p)
            x, y = int(p_transformed[0]), int(p_transformed[1])
            if 0 <= x < h and 0 <= y < w:
                transformed_frame[x, y] = frame[i, j]

    return transformed_frame


# %%
def euler_to_rotation_matrix(ex, ey, ez):
    Rx = np.array(
        [[1, 0, 0], [0, np.cos(ex), -np.sin(ex)], [0, np.sin(ex), np.cos(ex)]]
    )

    Ry = np.array(
        [[np.cos(ey), 0, np.sin(ey)], [0, 1, 0], [-np.sin(ey), 0, np.cos(ey)]]
    )

    Rz = np.array(
        [[np.cos(ez), -np.sin(ez), 0], [np.sin(ez), np.cos(ez), 0], [0, 0, 1]]
    )

    return np.dot(Rz, np.dot(Ry, Rx))


# %%
def reconstruct_volume(frames, transformations):
    volume = np.zeros((1000, 657, len(frames)))  # Adjust size as needed
    cumulative_transform = np.eye(4)

    for i, (frame, transform) in enumerate(zip(frames, transformations)):
        transform = transform.cpu().numpy()

        t_matrix = np.eye(4)
        t_matrix[:3, 3] = transform[:3]  # translation
        t_matrix[:3, :3] = euler_to_rotation_matrix(
            transform[3], transform[4], transform[5]
        )

        cumulative_transform = np.dot(cumulative_transform, t_matrix)

        transformed_frame = apply_transform(frame, cumulative_transform)

        volume[:, :, i] = transformed_frame

    return volume


# %% [markdown]
# ## 7. Visualize Reconstructed Volume
#

# %%
# Perform inference and reconstruction
# sequence_model.eval()
pair_model.eval()

with torch.no_grad():
    # # Sequence model inference
    # for frames, imu_data, _ in sequence_loader:
    #     frames, imu_data = frames.to(device), imu_data.to(device)
    #     sequence_transforms = sequence_model(frames, imu_data)
    #     sequence_volume = reconstruct_volume(frames[0], sequence_transforms[0])
    #     break  # Just process one sequence for this example

    # Pair model inference
    reconstructed_frames = []
    pair_transforms = []
    for frames, imu_data, _ in pair_loader:
        frames, imu_data = frames.to(device), imu_data.to(device)
        pair_transform = pair_model(frames, imu_data)
        reconstructed_frames.extend(frames[:, 1].cpu().numpy())
        pair_transforms.extend(pair_transform.cpu().numpy())
    pair_volume = reconstruct_volume(reconstructed_frames, pair_transforms)
