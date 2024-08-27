import argparse

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.us_dataset import QuaternionToEulerTransform, SequentialDataset
from models.gru.grunet import GruNet
from utils.losses.mse_correlation_loss import MSECorrelationLoss


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train(config):
    device = torch.device(config["device"])

    # Load data
    df = pd.read_hdf(config["data_file"])

    # Create dataset
    transform = QuaternionToEulerTransform()
    dataset = SequentialDataset(
        df,
        sequence_length=config["sequence_length"],
        transform=transform,
        downsample_factor=config["downsample_factor"],
    )

    # Split dataset
    train_indices, val_indices = train_test_split(
        range(len(dataset)), test_size=config["val_split"], random_state=42
    )

    # Create data loaders
    train_loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        sampler=torch.utils.data.SubsetRandomSampler(train_indices),
    )
    val_loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        sampler=torch.utils.data.SubsetRandomSampler(val_indices),
    )

    # Initialize model
    model = GruNet(
        input_channels=1,
        input_height=1000 // config["downsample_factor"],
        input_width=657 // config["downsample_factor"],
    ).to(device)

    # Loss and optimizer
    criterion = MSECorrelationLoss(lambda_corr=config["lambda_corr"])
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=5, verbose=True
    )

    # Training loop
    for epoch in range(config["num_epochs"]):
        model.train()
        train_loss = 0.0
        for frames, imu_data, targets in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}"
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

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        print(
            f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )

        scheduler.step(val_loss)

    # Save the model
    torch.save(model.state_dict(), config["model_save_path"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Ultrasound GRU Model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    train(config)
