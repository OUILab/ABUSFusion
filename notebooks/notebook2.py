# %% [markdown]
# # Ultrasound Reconstruction with IMU and Optical Tracker Data

# %% [markdown]
# ## 1. Import Libraries

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# %%
import torch
import torch.nn as nn
import torch.optim as optim
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from tqdm import tqdm

# %% [markdown]
# ## 2. Load and Preprocess Data


# %%
def load_data(file_path):
    return pd.read_hdf(file_path)


file_path = "path/to/your/data.h5"
df = load_data(file_path)

# %% [markdown]
# ## 3. Create Dataset and DataLoader


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


# Create datasets and dataloaders
transform = QuaternionToEulerTransform()
dataset = UltrasoundSequenceDataset(df, sequence_length=10, transform=transform)

# Split the dataset
train_indices, val_indices = train_test_split(
    range(len(dataset)), test_size=0.2, random_state=42
)

train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

train_loader = DataLoader(dataset, batch_size=32, sampler=train_sampler)
val_loader = DataLoader(dataset, batch_size=32, sampler=val_sampler)

# %% [markdown]
# ## 4. Define Model Architecture


# %%
class UltrasoundGRUModel(nn.Module):
    def __init__(
        self, sequence_length, input_channels=1, input_height=1000, input_width=657
    ):
        super(UltrasoundGRUModel, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        conv_output_height = input_height // 4
        conv_output_width = input_width // 4

        self.gru = nn.GRU(
            64 * conv_output_height * conv_output_width + 6, 128, batch_first=True
        )
        self.fc = nn.Linear(128, 6)  # Output: tx, ty, tz, ex, ey, ez

    def forward(self, frames, imu_data):
        batch_size, seq_len, _, height, width = frames.shape

        frame_features = []
        for i in range(seq_len):
            x = self.pool(torch.relu(self.conv1(frames[:, i, :, :, :])))
            x = self.pool(torch.relu(self.conv2(x)))
            frame_features.append(x.view(batch_size, -1))

        combined_features = torch.cat(
            [torch.stack(frame_features, dim=1), imu_data], dim=2
        )

        gru_out, _ = self.gru(combined_features)

        transformations = self.fc(gru_out)

        return transformations


# Initialize model
model = UltrasoundGRUModel(
    sequence_length=10, input_channels=1, input_height=1000, input_width=657
)

# %% [markdown]
# ## 5. Train Model

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(
    optimizer, mode="min", factor=0.1, patience=5, verbose=True
)


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    num_epochs=100,
    patience=10,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    best_val_loss = float("inf")
    best_model_weights = None
    epochs_no_improve = 0

    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        model.train()
        train_loss = 0.0

        for frames, imu_data, targets in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False
        ):
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

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for frames, imu_data, targets in val_loader:
                frames, imu_data, targets = (
                    frames.to(device),
                    imu_data.to(device),
                    targets.to(device),
                )
                outputs = model(frames, imu_data)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        scheduler.step(val_loss)

        tqdm.write(
            f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                tqdm.write("Early stopping triggered")
                break

    model.load_state_dict(best_model_weights)
    return model


# Train the model
trained_model = train_model(
    model, train_loader, val_loader, criterion, optimizer, scheduler
)

# Save the best model
torch.save(trained_model.state_dict(), "best_model.pth")

# %% [markdown]
# ## 6. Evaluate Model


# %%
def evaluate_model(model, data_loader):
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for frames, imu_data, targets in data_loader:
            frames, imu_data, targets = (
                frames.to(device),
                imu_data.to(device),
                targets.to(device),
            )
            outputs = model(frames, imu_data)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)

    return avg_loss, all_predictions, all_targets


# Evaluate the model
val_loss, val_predictions, val_targets = evaluate_model(trained_model, val_loader)
print(f"Validation Loss: {val_loss:.4f}")

# %% [markdown]
# ## 7. Visualize Results


# %%
def visualize_predictions(predictions, targets):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    titles = [
        "Translation X",
        "Translation Y",
        "Translation Z",
        "Rotation X",
        "Rotation Y",
        "Rotation Z",
    ]

    for i in range(6):
        ax = axes[i // 3, i % 3]
        ax.plot(predictions[:100, i], label="Predicted")
        ax.plot(targets[:100, i], label="Ground Truth")
        ax.set_title(titles[i])
        ax.legend()

    plt.tight_layout()
    plt.show()


visualize_predictions(val_predictions, val_targets)

# %% [markdown]
# ## 8. Reconstruct 3D Volume


# %%
def reconstruct_volume(frames, transformations):
    # Implement your volume reconstruction logic here
    # This is a placeholder function
    volume = np.zeros((100, 100, 100))  # Example size, adjust as needed
    return volume


# Example usage:
sample_frames, sample_imu, _ = next(iter(val_loader))
sample_transformations = trained_model(sample_frames.to(device), sample_imu.to(device))
reconstructed_volume = reconstruct_volume(
    sample_frames.numpy(), sample_transformations.cpu().numpy()
)

# Visualize a slice of the reconstructed volume
plt.imshow(reconstructed_volume[50, :, :], cmap="gray")
plt.title("Slice of Reconstructed Volume")
plt.colorbar()
plt.show()
