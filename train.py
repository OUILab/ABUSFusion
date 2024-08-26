import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from models.dclnet.dclnet import DCLNet
from data.us_dataset import USDataset
from utils.losses.mse_correlation_loss import MSECorrelationLoss
import yaml
import os


def train(config):
    # Set device
    device = torch.device(config["device"])

    # Data loading
    transform = transforms.Compose(
        [
            transforms.Resize((config["image_size"], config["image_size"])),
            transforms.ToTensor(),
        ]
    )

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
        val_dataset, batch_size=config["batch_size"], num_workers=config["num_workers"]
    )

    # Model
    model = DCLNet(num_input_frames=config["num_input_frames"], num_classes=7).to(
        device
    )

    # Loss and optimizer
    criterion = MSECorrelationLoss(lambda_corr=config["lambda_corr"])
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=config["lr_step_size"], gamma=config["lr_gamma"]
    )

    # Training loop
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


if __name__ == "__main__":
    with open("configs/default_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    train(config)
