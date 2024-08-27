import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from tqdm import tqdm


class MultiSessionUltrasoundDataset(Dataset):
    def __init__(self, dataframes, sequence_length=10, transform=None):
        self.dataframes = dataframes
        self.sequence_length = sequence_length
        self.transform = transform
        self.session_indices = self._create_session_indices()

    def _create_session_indices(self):
        session_indices = []
        start_idx = 0
        for i, df in enumerate(self.dataframes):
            session_indices.extend(
                [(i, j) for j in range(len(df) - self.sequence_length + 1)]
            )
            start_idx += len(df)
        return session_indices

    def __len__(self):
        return len(self.session_indices)

    def __getitem__(self, idx):
        session_idx, start_idx = self.session_indices[idx]
        df = self.dataframes[session_idx]
        sequence = df.iloc[start_idx : start_idx + self.sequence_length]

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


# Usage example
dataframes = [df1, df2, df3]  # Your multiple scan session dataframes
dataset = MultiSessionUltrasoundDataset(
    dataframes, sequence_length=10, transform=QuaternionToEulerTransform()
)

# Split at session level
n_sessions = len(dataframes)
train_indices, val_indices = train_test_split(
    range(n_sessions), test_size=0.2, random_state=42
)

train_sampler = SubsetRandomSampler(
    [i for i in range(len(dataset)) if dataset.session_indices[i][0] in train_indices]
)
val_sampler = SubsetRandomSampler(
    [i for i in range(len(dataset)) if dataset.session_indices[i][0] in val_indices]
)

train_loader = DataLoader(dataset, batch_size=32, sampler=train_sampler)
val_loader = DataLoader(dataset, batch_size=32, sampler=val_sampler)

# Initialize model, criterion, optimizer, and scheduler
model = UltrasoundGRUModel(
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
torch.save(trained_model.state_dict(), "best_model.pth")
