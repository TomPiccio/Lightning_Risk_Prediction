import os
import json
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import boxcox
import joblib
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import boxcox
def normalize(df: pd.DataFrame, df_name: str):
    # Define save path
    save_path = os.path.join(os.path.join(os.getcwd(),".."), "data/params")
    
    # Ensure the directory exists
    os.makedirs(save_path, exist_ok=True)
    
    # Drop Timestamp column and reset index for processing
    df_values = df.drop(columns=["Timestamp"], errors="ignore").melt().value.dropna()

    # Remove zeros for Box-Cox
    df_values_non_zero = df_values[df_values > 0]
    
    # Ensure that there are positive values
    if len(df_values_non_zero) == 0:
        raise ValueError(f"No positive values in the dataset for Box-Cox transformation.")
    non_zero_min = df_values_non_zero.min()

    zero_replace = non_zero_min / 1e5
    print(f"Zero Replace: {zero_replace}")
    df_values.replace(0, zero_replace, inplace=True)
    
    # Apply Box-Cox transformation
    transformed, lmbda = boxcox(df_values.values.flatten())
    
    # Min-Max Scaling
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(transformed.reshape(-1, 1)).flatten()
    
    # Save parameters for later use
    param_dict = {
        "zero_replace": zero_replace,
        "lmbda": lmbda,
        "min": scaler.data_min_[0],
        "max": scaler.data_max_[0]
    }
    
    # Save parameters to a JSON file
    with open(os.path.join(save_path, f"{df_name}_boxcox_params.json"), "w") as f:
        json.dump(param_dict, f, indent=4)
    
    # Save the scaler object
    joblib.dump(scaler, os.path.join(save_path, f"{df_name}_scaler.pkl"))
    print("Done saving scaler object and param dict.")
def normalize_new_data(df: pd.DataFrame, df_name: str):
    load_path = os.path.join(os.path.join(os.getcwd(), ".."), "data/params")

    # Separate Timestamp column if it exists
    timestamp_col = None
    if "Timestamp" in df.columns:
        timestamp_col = df["Timestamp"]
    
    # Drop Timestamp and copy data for normalization
    df_clean = df.drop(columns=["Timestamp"], errors="ignore").copy()
    
    # Save original zero locations
    zero_mask = (df_clean == 0)

    # Load parameters from saved JSON file
    with open(os.path.join(load_path, f"{df_name}_boxcox_params.json"), "r") as f:
        param_dict = json.load(f)

    # Replace 0s in the new data with the stored value
    df_clean.replace(0, param_dict["zero_replace"], inplace=True)

    # Apply stored Box-Cox transformation with the saved lambda
    lmbda = param_dict["lmbda"]
    transformed = boxcox(df_clean.values.flatten(), lmbda=lmbda)

    # Load the scaler and apply it
    scaler = joblib.load(os.path.join(load_path, f"{df_name}_scaler.pkl"))
    scaled = scaler.transform(transformed.reshape(-1, 1)).flatten()

    # Reshape back to the original dataframe shape
    norm_df = pd.DataFrame(scaled.reshape(df_clean.shape), columns=df_clean.columns, index=df_clean.index)

    # Restore zeros using the mask
    norm_df[zero_mask] = 0

    # Set Timestamp as index if it existed
    if timestamp_col is not None:
        norm_df.index = pd.to_datetime(timestamp_col)

    print("Done normalizing")
    return norm_df
