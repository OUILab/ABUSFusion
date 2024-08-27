# main.py

import argparse
import os

import numpy as np
import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms

from data.us_dataset import USDataset
from models.dclnet.dclnet import DCLNet
from utils.losses.mse_correlation_loss import MSECorrelationLoss
from utils.visualization.plot_results import visualize_3d_volume


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train(
    model, train_loader, val_loader, criterion, optimizer, scheduler, config, device
):
    for epoch in range(config["num_epochs"]):
        model.train()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            if batch_idx % config["log_interval"] == 0:
                print(
                    f"Train Epoch: {epoch} [{batch_idx * len(inputs)}/{len(train_loader.dataset)} "
                    f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
                )

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()

        val_loss /= len(val_loader)
        print(f"Validation loss: {val_loss:.6f}")

        scheduler.step()

    # Save the model
    torch.save(model.state_dict(), config["model_save_path"])


def inference(model, inference_loader, config, device, vis_framework):
    model.eval()
    reconstructed_positions = []

    with torch.no_grad():
        for inputs, start_indices in inference_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            reconstructed_positions.extend(outputs.cpu().numpy())

    reconstructed_positions = np.array(reconstructed_positions)
    visualize_3d_volume(reconstructed_positions, framework=vis_framework)


def main(args):
    config = load_config(args.config)
    device = torch.device(config["device"])

    # Data loading
    transform = transforms.Compose(
        [
            transforms.Resize((config["image_size"], config["image_size"])),
            transforms.ToTensor(),
        ]
    )

    if args.mode in ["train", "val"]:
        train_dataset = USDataset(
            h5_file=config["train_h5_file"],
            image_root_dir=config["image_root_dir"],
            num_frames=config["num_input_frames"],
            transform=transform,
        )
        val_dataset = USDataset(
            h5_file=config["val_h5_file"],
            image_root_dir=config["image_root_dir"],
            num_frames=config["num_input_frames"],
            transform=transform,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config["num_workers"],
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
        )

    if args.mode == "inference":
        inference_dataset = USDataset(
            h5_file=config["inference_h5_file"],
            image_root_dir=config["image_root_dir"],
            num_frames=config["num_input_frames"],
            transform=transform,
            inference_mode=True,
        )
        inference_loader = DataLoader(
            inference_dataset,
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
        )

    # Model
    model = DCLNet(num_input_frames=config["num_input_frames"], num_classes=7).to(
        device
    )

    if args.mode in ["val", "inference"]:
        model.load_state_dict(torch.load(config["model_save_path"]))

    if args.mode in ["train", "val"]:
        criterion = MSECorrelationLoss(lambda_corr=config["lambda_corr"])
        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=config["lr_step_size"], gamma=config["lr_gamma"]
        )

        train(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            scheduler,
            config,
            device,
        )

    if args.mode == "inference":
        inference(model, inference_loader, config, device, args.vis_framework)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DCL-Net for 3D Ultrasound Reconstruction"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "val", "inference"],
        required=True,
        help="Mode of operation",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--vis_framework",
        type=str,
        choices=["matplotlib", "vispy", "rapids"],
        default="matplotlib",
        help="Visualization framework",
    )
    args = parser.parse_args()

    main(args)
