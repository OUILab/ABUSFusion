import argparse

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from data.us_dataset import QuaternionToEulerTransform, SequentialDataset
from models.gru.grunet import GruNet
from utils.reconstruction import reconstruct_volume


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_inference(config, data_file, output_file):
    device = torch.device(config["device"])

    # Load data
    df = pd.read_hdf(data_file)

    # Create dataset and dataloader
    transform = QuaternionToEulerTransform()
    dataset = SequentialDataset(
        df,
        sequence_length=config["sequence_length"],
        transform=transform,
        downsample_factor=config["downsample_factor"],
    )
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)

    # Load model
    model = GruNet(
        input_channels=1,
        input_height=1000 // config["downsample_factor"],
        input_width=657 // config["downsample_factor"],
    ).to(device)
    model.load_state_dict(torch.load(config["model_save_path"]))
    model.eval()

    # Run inference
    all_frames = []
    all_predictions = []

    with torch.no_grad():
        for frames, imu_data, _ in dataloader:
            frames = frames.to(device)
            imu_data = imu_data.to(device)
            outputs = model(frames, imu_data)

            all_frames.append(frames.cpu().numpy())
            all_predictions.append(outputs.cpu().numpy())

    all_frames = np.concatenate(all_frames, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)

    # Reconstruct volume
    probe_specs = config["probe_specs"]
    reconstructed_volume, volume_origin, voxel_size = reconstruct_volume(
        all_frames[:, 0],  # Use only the first frame of each sequence
        all_predictions[:, -1],  # Use the last prediction of each sequence
        probe_specs,
        voxel_size=config["reconstruction_voxel_size"],
    )

    # Save reconstructed volume
    np.save(output_file, reconstructed_volume)
    print(f"Reconstructed volume saved to {output_file}")

    # Save metadata
    metadata = {
        "volume_origin": volume_origin.tolist(),
        "voxel_size": voxel_size,
        "probe_specs": probe_specs,
    }
    metadata_file = output_file.rsplit(".", 1)[0] + "_metadata.npy"
    np.save(metadata_file, metadata)
    print(f"Metadata saved to {metadata_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference and reconstruct 3D ultrasound volume"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--data", type=str, required=True, help="Path to input data file (HDF5)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output file for reconstructed volume",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run_inference(config, args.data, args.output)
