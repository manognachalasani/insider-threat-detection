import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 3)
        )

        self.decoder = nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def normalize_scores(scores):
    scores = np.clip(scores, 0, None)

    min_val = scores.min()
    max_val = scores.max()

    if max_val - min_val == 0:
        return np.zeros_like(scores)

    normalized = (scores - min_val) / (max_val - min_val)

    return np.round(normalized, 6)


def run_autoencoder(data):

    print("\nRunning Autoencoder...\n")

    #handle user column

    possible_user_cols = [
        "user_id",
        "user",
        "userid",
        "employee_id",
        "employee",
        "id"
    ]

    user_col = None

    for col in possible_user_cols:
        if col in data.columns:
            user_col = col
            break

    if user_col is None:
        raise ValueError(
            f"No valid user column found.\nColumns found:\n{list(data.columns)}"
        )

    data = data.rename(columns={user_col: "user_id"})

    print(f"Using '{user_col}' as user column")

    #keep only numeric features for autoencoder

    feature_cols = data.columns.drop("user_id")

    numeric_data = data[feature_cols].select_dtypes(
        include=[np.number]
    ).copy()

    if numeric_data.shape[1] == 0:
        raise ValueError("No numeric columns found in dataset")

    print(f"\nUsing {numeric_data.shape[1]} numeric features")

    #scaling

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(numeric_data)

    user_ids = data["user_id"].values

    #train on first 90% of data, test on all data to get reconstruction errors for everyone

    n_normal = int(0.9 * len(X_scaled))

    X_train = X_scaled[:n_normal]
    X_test = X_scaled

    print(f"Training on first {n_normal} rows")

    #model

    model = Autoencoder(X_train.shape[1])

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    X_train_tensor = torch.tensor(
        X_train,
        dtype=torch.float32
    )

    #training

    print("\nTraining model...\n")

    for epoch in range(50):

        optimizer.zero_grad()

        reconstructed = model(X_train_tensor)

        loss = criterion(reconstructed, X_train_tensor)

        loss.backward()

        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.6f}")

    #reconstruction errors

    X_test_tensor = torch.tensor(
        X_test,
        dtype=torch.float32
    )

    with torch.no_grad():

        reconstructed = model(X_test_tensor)

        feature_errors = (
            (X_test_tensor - reconstructed) ** 2
        ).numpy()

    #feature groups

    cols = numeric_data.columns

    cols_lower = [c.lower() for c in cols]

    login_cols = [
        cols[i]
        for i, c in enumerate(cols_lower)
        if (
            "login" in c or
            "auth" in c or
            "access" in c
        )
    ]

    volume_cols = [
        cols[i]
        for i, c in enumerate(cols_lower)
        if (
            "upload" in c or
            "download" in c or
            "file" in c or
            "volume" in c
        )
    ]

    network_cols = [
        cols[i]
        for i, c in enumerate(cols_lower)
        if (
            "network" in c or
            "ip" in c or
            "traffic" in c
        )
    ]

    usb_cols = [
        cols[i]
        for i, c in enumerate(cols_lower)
        if (
            "usb" in c or
            "device" in c
        )
    ]

    print("\nDetected Feature Groups:\n")

    print("Login Features:")
    print(login_cols)

    print("\nVolume Features:")
    print(volume_cols)

    print("\nNetwork Features:")
    print(network_cols)

    print("\nUSB Features:")
    print(usb_cols)

    
    #get column indices for each group
    
    def get_indices(column_group):
        return [
            cols.get_loc(col)
            for col in column_group
            if col in cols
        ]

    login_idx = get_indices(login_cols)
    volume_idx = get_indices(volume_cols)
    network_idx = get_indices(network_cols)
    usb_idx = get_indices(usb_cols)

    
    #group errors
    
    def calculate_group_error(indices):

        if len(indices) == 0:
            return np.zeros(len(feature_errors))

        return feature_errors[:, indices].mean(axis=1)

    login_error = calculate_group_error(login_idx)

    volume_error = calculate_group_error(volume_idx)

    network_error = calculate_group_error(network_idx)

    usb_error = calculate_group_error(usb_idx)


    # normalize scores to 0-1 range
    
    login_score = normalize_scores(login_error)

    volume_score = normalize_scores(volume_error)

    network_score = normalize_scores(network_error)

    usb_score = normalize_scores(usb_error)

    # combined autoencoder risk score
    autoencoder_risk = (
        login_score * 0.25 +
        volume_score * 0.25 +
        network_score * 0.25 +
        usb_score * 0.25
    )

    
    # op table
    
    output_df = pd.DataFrame({
    "user_id": user_ids,
    "login_anomaly": login_score,
    "file_anomaly": volume_score,
    "network_anomaly": network_score,
    "system_anomaly": usb_score,
    "autoencoder_risk": autoencoder_risk
})

    # keep highest anomaly score per user
    output_df = output_df.groupby(
        "user_id",
        as_index=False
    ).max()

    
    #sample op
    
    print("\nSample Output:\n")

    print(output_df.head(10))

    print("\nAnomaly Score Summary:\n")

    print(output_df.describe())

    #save op
    output_file = "autoencoder_output.xlsx"

    output_df.to_excel(
        output_file,
        index=False
    )

    print(f"\nSaved output to '{output_file}'")

    print("\nAutoencoder completed successfully.\n")

    return output_df


#main
if __name__ == "__main__":

    try:

        # reads your Excel dataset directly
        data = pd.read_excel(
            "behavior_dataset.xlsx"
        )

        results = run_autoencoder(data)

    except Exception as e:

        print("\nERROR:\n")

        print(str(e))