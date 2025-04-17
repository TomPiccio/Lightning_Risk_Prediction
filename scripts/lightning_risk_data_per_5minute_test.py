import pandas as pd
from tqdm import tqdm

input_data = "data/lightning_risk_data_test/lightning_risk_data_mins.csv"
output_base = "data/lightning_risk_data_test/lightning_risk_data_per_5min"

# Read data into DataFrame
df = pd.read_csv(input_data, parse_dates=['DateTime'])

# Define min and max timestamps rounded up to next 5-min
min_time = df['DateTime'].min().ceil('5min')
max_time = df['DateTime'].max().ceil('5min')

# Generate 5-minute interval datetime range
date_range = pd.date_range(start=min_time, end=max_time, freq='5min')
print(min_time, max_time)

# Create new DataFrame with 5-min timestamps
df_new = pd.DataFrame({'DateTime': date_range})
df_new['Date'] = df_new['DateTime'].dt.date.astype(str)
df_new['Time'] = df_new['DateTime'].dt.strftime('%H:%M:%S')

# Initialize new DataFrame with False for each target column
df_new = df_new.assign(**{col: False for col in df.columns[3:]})

# Fill values based on closest previous row
for i in tqdm(range(len(df_new)), total=len(df_new), desc="Processing rows"):
    x = df_new.at[i, 'DateTime']
    df_past = df[df['DateTime'] <= x]

    if df_past.empty:
        continue  # No previous data available

    y = df_past.iloc[-1]  # Get the closest previous row
    time_diff = (x - y['DateTime']).total_seconds() / 60

    for col in df.columns[3:]:
        df_new.at[i, col] = time_diff < y[col]  # Assign boolean value

# Estimate row size and determine max rows per file
max_file_size = 90 * 1024 * 1024  # 90 MB
sample_size = min(1000, len(df_new))
sample_csv = df_new.iloc[:sample_size].to_csv(index=False)
avg_row_size = len(sample_csv) / sample_size
max_rows = int(max_file_size / avg_row_size)

# Split and Save in Parts
num_parts = (len(df_new) // max_rows) + 1
for i in range(num_parts):
    start_idx = i * max_rows
    end_idx = min((i + 1) * max_rows, len(df_new))

    if start_idx >= len(df_new):
        break

    output_file = f"{output_base}_{i+1}.csv"
    df_new.iloc[start_idx:end_idx].to_csv(output_file, index=False)
    print(f"Saved {output_file} with rows {start_idx} to {end_idx}")
