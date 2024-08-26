import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from raft import RAFT  # You'll need to import RAFT from the appropriate library
from mamba_ssm import Mamba  # You'll need to import Mamba from the appropriate library


class SingleSessionUltrasoundDataset(Dataset):
    def __init__(self, df, sequence_length=10, transform=None):
        self.df = df
        self.sequence_length = sequence_length
        self.transform = transform
        self.indices = self._create_indices()

    def _create_indices(self):
        return list(range(len(self.df) - self.sequence_length + 1))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start_idx = self.indices[idx]
        sequence = self.df.iloc[start_idx : start_idx + self.sequence_length]

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


class UltrasoundRAFTMambaModel(nn.Module):
    def __init__(
        self, sequence_length, input_channels=1, input_height=1000, input_width=657
    ):
        super(UltrasoundRAFTMambaModel, self).__init__()
        self.raft = RAFT()  # Initialize RAFT model
        self.mamba = Mamba(
            d_model=128, d_state=16, d_conv=4, expand=2
        )  # Initialize Mamba model

        # Adjust these sizes based on RAFT output and your specific requirements
        raft_output_size = 128  # This should match the output size of your RAFT model
        imu_feature_size = 6  # 6 IMU features

        self.fc_raft = nn.Linear(raft_output_size, 64)
        self.fc_imu = nn.Linear(imu_feature_size, 64)
        self.fc_out = nn.Linear(128, 6)  # Output: tx, ty, tz, ex, ey, ez

    def forward(self, frames, imu_data):
        batch_size, seq_len, _, height, width = frames.shape

        # Process frame pairs with RAFT
        raft_features = []
        for i in range(seq_len - 1):
            flow = self.raft(frames[:, i], frames[:, i + 1])
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


# Usage example with a single dataframe
df = your_single_dataframe  # Your single dataframe
dataset = SingleSessionUltrasoundDataset(
    df, sequence_length=10, transform=QuaternionToEulerTransform()
)

# Split at sample level for single dataframe
train_indices, val_indices = train_test_split(
    range(len(dataset)), test_size=0.2, random_state=42
)

train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

train_loader = DataLoader(dataset, batch_size=32, sampler=train_sampler)
val_loader = DataLoader(dataset, batch_size=32, sampler=val_sampler)

# Initialize model, criterion, optimizer, and scheduler
model = UltrasoundRAFTMambaModel(
    sequence_length=10, input_channels=1, input_height=1000, input_width=657
)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(
    optimizer, mode="min", factor=0.1, patience=5, verbose=True
)

# Train the model
trained_model = train_model(
    model, train_loader, val_loader, criterion, optimizer, scheduler
)

# Save the best model
torch.save(trained_model.state_dict(), "best_model_raft_mamba.pth")
