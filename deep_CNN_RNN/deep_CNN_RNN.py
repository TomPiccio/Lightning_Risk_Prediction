import torch
from torch.utils.data import Dataset
import pandas as pd
import pytz
import numpy as np
import os
import sys
import glob
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
from datetime import datetime
import matplotlib.pyplot as plt
import re
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import ConcatDataset, TensorDataset

chosen_stations = ['S104', 'S107', 'S109', 'S115', 'S116', 'S43', 'S50']
pixel_coords = [(4, 11, 'S109'),
 (2, 7, 'S50'),
 (1, 16, 'S107'),
 (2, 13, 'S43'),
 (0, 0, 'S115'),
 (0, 6, 'S116'),
 (8, 8, 'S104')]
def tabular_to_image(data: pd.DataFrame, pixel_coords, image_shape=(9, 18)):
    feature_types = ['rainfall', 'air_temperature', 'wind_speed', 'relative_humidity', 'wind_direction']
    H, W = image_shape
    T = data.shape[0]
    image = np.full((T, H, W, len(feature_types)), np.nan, dtype=np.float32)

    feature_to_channel = {feat: i for i, feat in enumerate(feature_types)}

    for y, x, station_id in pixel_coords:
        for feat in feature_types:
            col_name = f"{feat}_{station_id}"
            if col_name in data.columns:
                channel = feature_to_channel[feat]
                image[:, y, x, channel] = data[col_name].values

    return image if T > 1 else image[0]

class deep_CNN_RNN_Dataset(Dataset):
    def __init__(self, compiled_df, pixel_coords = pixel_coords, image_shape=(9, 18), timezone_str="Asia/Singapore", reject_zeros=True, interpolate=True):
        self.compiled_df = compiled_df.copy()
        self.pixel_coords = pixel_coords
        self.image_shape = image_shape
        self.timezone = pytz.timezone(timezone_str)
        self.samples = []
        self.reject_zeros = reject_zeros
        self.rejected_samples = []
        self.interpolate = interpolate

        self._prepare_dataset()

    def _prepare_dataset(self):
        # Ensure datetime index
        self.compiled_df["Timestamp"] = pd.to_datetime(self.compiled_df["Timestamp"])
        if not isinstance(self.compiled_df.index, pd.DatetimeIndex):
            self.compiled_df.set_index("Timestamp", inplace=True)
        self.compiled_df.index = self.compiled_df.index.tz_localize(None)

        # Drop target for input features
        input_df = self.compiled_df.drop(columns=["Lightning_Risk"])

        # Valid 2-hour timestamps
        min_ts = self.compiled_df.index.min().ceil("2h") + pd.Timedelta(hours=2)
        max_ts = self.compiled_df.index.max().floor("2h")
        valid_ts = self.compiled_df.loc[
            (self.compiled_df.index >= min_ts) &
            (self.compiled_df.index <= max_ts) &
            (self.compiled_df.index.hour % 2 == 0) &
            (self.compiled_df.index.minute == 0)
        ].index

        for timestamp in valid_ts:
            try:
                # Input time windows (past 5)
                input_times = [timestamp - pd.Timedelta(minutes=delta) for delta in [120, 90, 60, 30, 0]]
                input_slices = self.compiled_df.loc[input_times]
                input_images = tabular_to_image(input_slices, self.pixel_coords, self.image_shape)  # (5, H, W, C)

                # Rearrange to (C, T, H, W) if needed
                input_tensor = np.transpose(input_images, (3, 0, 1, 2))  # (C, T, H, W)

                if self.interpolate:
                    input_tensor = self.interpolate_nan(input_tensor)

                # Output time windows (future 5)
                output_times = [timestamp + pd.Timedelta(minutes=delta) for delta in [0, 30, 60, 90, 120]]
                output_data = self.compiled_df.loc[output_times, "Lightning_Risk"].astype(int).values.flatten()

                if self.reject_zeros and not (output_data == 1).any():
                    input_tensor = torch.tensor(input_tensor, dtype=torch.float32)
                    output_data = torch.tensor(output_data, dtype=torch.float32)
                    self.rejected_samples.append((input_tensor, output_data))
                    continue

                self.samples.append((torch.tensor(input_tensor, dtype=torch.float32), torch.tensor(output_data, dtype=torch.float32)))

            except KeyError:
                continue  # Skip if any timestamps are missing

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        x = x.permute(1, 0, 2, 3)  # (C, T, H, W) → (T, C, H, W)
        return x, y

    def get_positive_ratio(self):
        all_labels = np.array([sample[1] for sample in self.samples])  # shape (N, 5)
        total = all_labels.size
        positives = (all_labels == 1).sum()
        return positives / total

    def get_rejected_samples(self):
        inputs, outputs = zip(*self.rejected_samples)
        input_tensor = torch.stack(inputs)  # Could be (N, C, T, H, W)
        output_tensor = torch.stack(outputs)

        input_tensor = input_tensor.permute(0, 2, 1, 3, 4)

        return torch.utils.data.TensorDataset(input_tensor, output_tensor)

    def interpolate_nan(self, x):
        # x: (C, T, H, W)
        C, T, H, W = x.shape
        for t in range(T):
            for c in range(C):
                frame = x[c, t]
                nan_mask = ~np.isnan(frame)
                frame = np.nan_to_num(frame, nan=0.0)

                frame_tensor = torch.tensor(frame, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                mask_tensor = torch.tensor(nan_mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

                kernel = torch.ones(1, 1, 5, 5) / 25.0
                blurred = F.conv2d(frame_tensor, kernel, padding=2)
                weight = F.conv2d(mask_tensor, kernel, padding=2)

                interpolated = blurred / (weight + 1e-6)
                x[c, t] = interpolated.squeeze().cpu().numpy()

        return x

    @classmethod
    def from_saved(cls, saved_path, compiled_df, pixel_coords, image_shape=(9, 18), timezone_str="Asia/Singapore"):
        # Create empty instance without triggering __init__
        self = cls.__new__(cls)

        # Manually set attributes
        self.compiled_df = compiled_df
        self.pixel_coords = pixel_coords
        self.image_shape = image_shape
        self.timezone = pytz.timezone(timezone_str)
        self.reject_zeros = True  # or False, based on what was used when saving
        self.interpolate = True   # same here

        # Load samples
        saved = torch.load(saved_path)
        self.samples = saved["samples"]
        self.rejected_samples = saved["rejected_samples"]

        return self
    
class deep_CNN_RNN_Module(nn.Module):
    def __init__(self, num_channels=5, num_future_steps=5, hidden_size=256):
        super(deep_CNN_RNN_Module, self).__init__()

        # --- Improved CNN Feature Extractor ---
        self.cnn = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Downsample to 4x9

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Downsample to 2x4

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # Final output: (batch, 128, 1, 1)
            nn.Dropout(0.3)
        )

        self.flatten = nn.Flatten()

        # Dynamically determine CNN output size
        with torch.no_grad():
            dummy_input = torch.zeros(1, num_channels, 9, 18)
            self.feature_size = self.flatten(self.cnn(dummy_input)).shape[1]  # Should be 128

        # --- RNN ---
        self.rnn = nn.RNN(
            input_size=self.feature_size,
            hidden_size=hidden_size,
            batch_first=True,
        )

        # --- Output Layer ---
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_future_steps)
        )

        # --- Learnable initial hidden state ---
        self.initial_hidden_state = nn.Parameter(torch.randn(1, 1, hidden_size))

    def forward(self, x):
        # x: (batch_size, seq_len, channels, height, width)
        batch_size, seq_len, c, h, w = x.shape

        cnn_features = []
        for t in range(seq_len):
            x_t = x[:, t]  # (batch, channels, height, width)
            x_t = torch.nan_to_num(x_t, nan=0.0)

            out = self.cnn(x_t)
            out = self.flatten(out)  # (batch, feature_size)

            cnn_features.append(out)

        cnn_features = torch.stack(cnn_features, dim=1)  # (batch_size, seq_len, feature_size)

        h0 = self.initial_hidden_state.expand(1, batch_size, -1).contiguous()
        rnn_out, _ = self.rnn(cnn_features, h0)

        final_hidden = rnn_out[:, -1, :]  # (batch_size, hidden_size)

        predictions = self.fc(final_hidden)  # For BCEWithLogitsLoss, remove sigmoid
        return predictions
