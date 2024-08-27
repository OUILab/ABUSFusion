# %% [markdown]
# # 3D Ultrasound Volume Reconstruction
#
# This notebook demonstrates how to load ultrasound data from an HDF5 file, train a model to predict transformation parameters, and visualize the 3D volume reconstruction.

# %%
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from freehand.loss import PointDistance
# Import custom modules
from freehand.network import build_model
from freehand.transform import (LabelTransform, PredictionTransform,
                                TransformAccumulation)
from freehand.utils import pair_samples, reference_image_points, type_dim
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import DataLoader, Dataset
from torchvision.models import efficientnet_b1

# %% [markdown]
# ## Load and Preprocess Data


# %%
class UltrasoundDataset(Dataset):
    def __init__(self, hdf5_file, num_samples=10, sample_range=10):
        self.data = pd.read_hdf(hdf5_file)
        self.num_samples = num_samples
        self.sample_range = sample_range

    def __len__(self):
        return len(self.data) - self.sample_range + 1

    def __getitem__(self, idx):
        sample = self.data.iloc[idx : idx + self.sample_range]
        frames = torch.tensor(
            np.stack(sample["ultrasound_frame"].values), dtype=torch.float32
        )
        imu_data = torch.tensor(
            sample[
                [
                    "acceleration_x",
                    "acceleration_y",
                    "acceleration_z",
                    "orientation_x",
                    "orientation_y",
                    "orientation_z",
                ]
            ].values,
            dtype=torch.float32,
        )
        ot_data = torch.tensor(
            sample[
                [
                    "ot_position_x",
                    "ot_position_y",
                    "ot_position_z",
                    "ot_orientation_x",
                    "ot_orientation_y",
                    "ot_orientation_z",
                ]
            ].values,
            dtype=torch.float32,
        )

        return frames, imu_data, ot_data


# Load the dataset
dataset = UltrasoundDataset("path/to/your/hdf5_file.h5")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# %% [markdown]
# ## Define Model and Training Parameters

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model parameters
NUM_SAMPLES = 10
NUM_PRED = 9
PRED_TYPE = "parameter"
LABEL_TYPE = "point"

# Training parameters
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100

# Create model
frame_size = dataset.data["ultrasound_frame"].iloc[0].shape
image_points = reference_image_points(frame_size, 2).to(device)
data_pairs = pair_samples(NUM_SAMPLES, NUM_PRED).to(device)
pred_dim = type_dim(PRED_TYPE, image_points.shape[1], data_pairs.shape[0])

model = build_model(efficientnet_b1, in_frames=NUM_SAMPLES, out_dim=pred_dim).to(device)

# Define loss and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# %% [markdown]
# ## Train the Model


# %%
def train_model(model, dataloader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (frames, imu_data, ot_data) in enumerate(dataloader):
            frames, imu_data, ot_data = (
                frames.to(device),
                imu_data.to(device),
                ot_data.to(device),
            )

            optimizer.zero_grad()
            outputs = model(frames)

            # Compute loss (you may need to adjust this based on your specific requirements)
            loss = criterion(
                outputs, ot_data[:, -1, :]
            )  # Predict the last frame's OT data

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader)}")


train_model(model, dataloader, criterion, optimizer, NUM_EPOCHS)

# %% [markdown]
# ## Visualize 3D Volume Reconstruction


# %%
def visualize_3d_volume(predicted_params, frame_size):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Convert predicted parameters to 3D points (this is a placeholder, adjust based on your transformation method)
    x = predicted_params[:, 0]
    y = predicted_params[:, 1]
    z = predicted_params[:, 2]

    # Plot the points
    ax.scatter(x, y, z, c=range(len(x)), cmap="viridis")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Reconstructed 3D Ultrasound Volume")

    plt.show()


# Perform inference on a sample
model.eval()
with torch.no_grad():
    sample_frames, _, _ = next(iter(dataloader))
    sample_frames = sample_frames.to(device)
    predicted_params = model(sample_frames)

visualize_3d_volume(predicted_params.cpu().numpy(), frame_size)

# %% [markdown]
# This visualization is a basic representation of the 3D volume using the predicted transformation parameters. You may need to adjust the visualization based on your specific transformation method and requirements.
