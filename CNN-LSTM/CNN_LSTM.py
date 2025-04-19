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

class CNN_LSTM_Dataset(Dataset):
    def __init__(self, compiled_df, pixel_coords, image_shape=(9, 18), timezone_str="Asia/Singapore", reject_zeros=True, interpolate=True):
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
    
class CNN_LSTM_Module(nn.Module):
    def __init__(self, num_channels=5, num_future_steps=5, hidden_size=256):
        super(CNN_LSTM_Module, self).__init__()

        # --- CNN Feature Extractor ---
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=32, kernel_size=5, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()

        # Dynamically determine CNN output size
        with torch.no_grad():
            dummy_input = torch.zeros(1, num_channels, 9, 18)
            x = F.relu(self.conv1(dummy_input))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            self.feature_size = self.flatten(x).shape[1]

        # --- LSTM ---
        self.lstm = nn.LSTM(input_size=self.feature_size, hidden_size=hidden_size, batch_first=True)

        # --- Output Layer ---
        self.fc = nn.Linear(hidden_size, num_future_steps)

        # --- Learnable initial hidden state ---
        self.initial_hidden_state = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.initial_cell_state = nn.Parameter(torch.randn(1, 1, hidden_size))  # LSTM also has a cell state

    def forward(self, x):
        # x: (batch_size, seq_len=5, channels=5, height, width)
        batch_size, seq_len, c, h, w = x.shape

        cnn_features = []
        for t in range(seq_len):
            x_t = x[:, t]  # (batch, channels, height, width)
            x_t = torch.nan_to_num(x_t, nan=0.0)

            out = F.relu(self.conv1(x_t))
            out = F.relu(self.conv2(out))
            out = F.relu(self.conv3(out))
            out = self.flatten(out)

            cnn_features.append(out)

        cnn_features = torch.stack(cnn_features, dim=1)  # (batch_size, seq_len, feature_size)

        # Expand learnable hidden state and cell state for batch
        h0 = self.initial_hidden_state.expand(1, batch_size, -1).contiguous()
        c0 = self.initial_cell_state.expand(1, batch_size, -1).contiguous()

        # LSTM
        lstm_out, _ = self.lstm(cnn_features, (h0, c0))

        # Final hidden state (from the last time step)
        final_hidden = lstm_out[:, -1, :]  # (batch_size, hidden_size)

        # Risk prediction without sigmoid (for BCEWithLogitsLoss)
        predictions = self.fc(final_hidden)  # (batch_size, num_future_steps)

        return predictions