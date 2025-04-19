# Matplotlib
import matplotlib.pyplot as plt
# Numpy
import numpy as np
# Pandas
import pandas as pd
# Torch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytz
import os
import sys
import glob
from datetime import datetime
from sklearn.metrics import f1_score, accuracy_score

class LSTM_Dataset(Dataset):
    def __init__(self, compiled_df, timezone_str="Asia/Singapore", hours_lookback=6):
        self.compiled_df = compiled_df.copy()
        self.timezone = pytz.timezone(timezone_str)
        self.samples = []
        self.hours_lookback = hours_lookback

        self._prepare_dataset()

    def _prepare_dataset(self):
        # Ensure datetime index
        self.compiled_df["Timestamp"] = pd.to_datetime(self.compiled_df["Timestamp"])
        if not isinstance(self.compiled_df.index, pd.DatetimeIndex):
            self.compiled_df.set_index("Timestamp", inplace=True)
        self.compiled_df.index = self.compiled_df.index.tz_localize(None)

        ## reindex with the min and max
        full_index = pd.date_range(start=self.compiled_df.index.min(), end=self.compiled_df.index.max(), freq='5min')
        self.compiled_df = self.compiled_df.reindex(full_index)

        nan_mask = self.compiled_df.isna() # find values that are NaN
        gap_id = nan_mask.ne(nan_mask.shift()).cumsum() # find the indices that mark the gaps beween a value and a NaN

        for col in self.compiled_df.columns:
            col_nan_mask = nan_mask[col] 
            col_gap_ids = gap_id[col]
            gap_sizes = col_nan_mask.groupby(col_gap_ids).transform('sum') # find the sizes of the gaps

            to_interpolate = (col_nan_mask) & (gap_sizes <= 24)
            self.compiled_df.loc[~to_interpolate & col_nan_mask, col] = -999999 ## set the NaNs in the large gaps to -999999 to avoid interpolation
            self.compiled_df[col] = self.compiled_df[col].interpolate(limit_direction='both')
            self.compiled_df.loc[self.compiled_df[col] == -999999, col] = np.nan
        
        # Prepare input features and drop target
        input_df = self.compiled_df.drop(columns=["Lightning_Risk"])
        input_columns = input_df.columns.values.tolist()

        # Get valid timestamps
        min_ts = self.compiled_df.index.min().ceil(f"{self.hours_lookback}h") + pd.Timedelta(hours=self.hours_lookback)
        max_ts = self.compiled_df.index.max().floor("2h")
        valid_ts = self.compiled_df.loc[
            (self.compiled_df.index >= min_ts) &
            (self.compiled_df.index <= max_ts) &
            (self.compiled_df.index.hour % 2 == 0) &
            (self.compiled_df.index.minute == 0)
        ].index

        for timestamp in valid_ts:
            try:
                # lookback of 6 hours
                input_data = input_df.loc[timestamp - pd.Timedelta(hours=self.hours_lookback):timestamp - pd.Timedelta(minutes=5)]

                # Output time windows (future)
                output_times = [timestamp + pd.Timedelta(minutes=delta) for delta in [0, 30, 60, 90, 120]]
                output_data = self.compiled_df.loc[output_times, "Lightning_Risk"]

                if input_data.isna().any().any() or output_data.isna().any().any():
                    continue

                input_data = input_data.values.flatten()
                input_data = input_data.reshape(-1, len(input_df.columns))
                # print(input_data.shape)
                output_data = output_data.astype(int).values.flatten()
                
                self.samples.append((input_data, output_data))
            except KeyError:
                continue  # Skip if any timestamps are missing

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_out):
        # lstm_out: [B, T, H]
        weights = self.attn(lstm_out)  # [B, T, 1]
        weights = torch.softmax(weights, dim=1)  # attention over time
        context = torch.sum(weights * lstm_out, dim=1)  # [B, H]
        return context

class LightningLSTMAttn(nn.Module):
    def __init__(self, hours_lookback=6, input_dim=35, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.hours_lookback = hours_lookback
        self.input_dim = input_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=False)
        self.attn = Attention(hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 5)  # 5 future time points
        )

    def forward(self, x):
        x = x.view(x.size(0), self.hours_lookback*12, self.input_dim)  # [B, T, F]
        lstm_out, _ = self.lstm(x)     # [B, T, H]
        context = self.attn(lstm_out)  # [B, H]
        logits = self.fc(context)      # [B, 5]
        return logits  # raw logits → use sigmoid during evaluation


class LSTM_Module(nn.Module):
    def __init__(self, hours_lookback=6, input_dim=35, hidden_dim=128, num_layers=1, dropout=0.2):
        super(LSTM_Module, self).__init__()
        self.hours_lookback=hours_lookback
        self.input_dim = input_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 5)  # 5 future time points
        )

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim] (32, 72, 40)
        x = x.view(x.size(0), self.hours_lookback*12, self.input_dim)  # [B, T, F]
        _, (hn, _) = self.lstm(x)  # Take the last hidden state
        hn = hn[-1]  # [batch_size, hidden_dim] from last LSTM layer
        out = self.fc(hn)  # [batch_size, 5]
        # return torch.sigmoid(out)  # Use Sigmoid if you want probabilities
        return out ## focal loss expects raw logits
    

def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for test_inputs, test_targets in test_loader:
            test_inputs, test_targets = test_inputs.to(device), test_targets.to(device)
            test_outputs = model(test_inputs)
            test_outputs = torch.sigmoid(test_outputs) ## for focal loss since modified model returns raw logits
            predicted = (test_outputs > 0.5).float()

            all_preds.append(predicted.cpu())
            all_targets.append(test_targets.cpu())

        # Stack all batches
        all_preds = torch.cat(all_preds).numpy()
        all_targets = torch.cat(all_targets).numpy()

        test_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
        test_acc = accuracy_score(all_targets, all_preds)
        per_class_f1 = f1_score(all_targets, all_preds, average=None, zero_division=0)

    print(f"macro f1: {test_f1:.2f}, test accuracy: {test_acc:.2f}, per class macro {per_class_f1}")

if "__name__" == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Define the directory and base filename pattern
    file_path_test = "../data/test_data/cleaned_compiled_data_normalized.csv"

    # Load and concatenate all parts
    test_data = pd.read_csv(file_path_test)

    test_dataset = LSTM_Dataset(test_data, hours_lookback=2)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # test_model = LightningLSTMAttn(input_dim=35, hours_lookback=2)
    test_model = LSTM_Module(input_dim=35, hours_lookback=2)
    test_model.load_state_dict(torch.load(f"models/SeqToLabels_2025_04_19_22_33_0.3563_best.pth", weights_only=True))
    test_model.to(device)
    evaluate_model(test_model, test_loader)