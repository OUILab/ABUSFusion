# data/us_dataset.py

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
import os


class USDataset(Dataset):
    def __init__(
        self,
        h5_file,
        image_root_dir,
        num_frames=5,
        transform=None,
        inference_mode=False,
    ):
        self.data = pd.read_hdf(h5_file)
        self.image_root_dir = image_root_dir
        self.num_frames = num_frames
        self.transform = transform
        self.inference_mode = inference_mode
        self.samples = self._prepare_samples()

    def _prepare_samples(self):
        samples = []
        for i in range(len(self.data) - self.num_frames + 1):
            samples.append(i)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        start_idx = self.samples[idx]

        frames = []
        for i in range(self.num_frames):
            row = self.data.iloc[start_idx + i]

            # Load image
            img_path = os.path.join(self.image_root_dir, row["image_path"])
            frame = Image.open(img_path).convert("L")  # Convert to grayscale
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)

        frames_tensor = torch.stack(frames)

        if self.inference_mode:
            return frames_tensor, start_idx

        # Extract OT data
        ot_data = []
        for i in range(self.num_frames):
            row = self.data.iloc[start_idx + i]
            ot = row["OT_columns"]
            ot_data.append(
                [ot["qw"], ot["qx"], ot["qy"], ot["qz"], ot["x"], ot["y"], ot["z"]]
            )

        ot_tensor = torch.tensor(ot_data, dtype=torch.float32)
        ot_mean = torch.mean(ot_tensor[:-1], dim=0)

        return frames_tensor, ot_mean
