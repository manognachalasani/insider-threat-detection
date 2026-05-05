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
        return self.decoder(self.encoder(x))


def run_autoencoder(data):
    print("\nRunning Autoencoder...\n")

    if "user_id" not in data.columns:
        raise ValueError("Dataset must contain 'user_id' column")

    #preprocessing
    feature_cols = data.columns.drop("user_id")
    numeric_data = data[feature_cols].select_dtypes(include=[np.number])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric_data)

    users = data["user"].values

    n_normal = int(0.9 * len(X_scaled))
    X_train = X_scaled[:n_normal]
    X_test = X_scaled

    print(f"Using first {n_normal} rows as normal data")

    #model
    model = Autoencoder(X_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)

    # training
    for epoch in range(50):
        optimizer.zero_grad()
        output = model(X_train_tensor)
        loss = criterion(output, X_train_tensor)
        loss.backward()
        optimizer.step()

    # reconstruction error
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    with torch.no_grad():
        reconstructed = model(X_test_tensor)
        feature_errors = (X_test_tensor - reconstructed) ** 2
        feature_errors = feature_errors.numpy()

    # feature grouping based on column names
    cols = numeric_data.columns
    cols_lower = [c.lower() for c in cols]

    login_cols = [cols[i] for i, c in enumerate(cols_lower) if "login" in c]
    volume_cols = [cols[i] for i, c in enumerate(cols_lower) if "upload" in c or "download" in c]
    network_cols = [cols[i] for i, c in enumerate(cols_lower) if "network" in c]
    usb_cols = [cols[i] for i, c in enumerate(cols_lower) if "usb" in c]

    print("\nFeature Groups:")
    print("Login:", login_cols)
    print("Volume:", volume_cols)
    print("Network:", network_cols)
    print("USB:", usb_cols)

    def group_error(indices):
        if len(indices) == 0:
            return np.zeros(len(feature_errors))
        return feature_errors[:, indices].mean(axis=1)

    login_idx = [cols.get_loc(c) for c in login_cols]
    volume_idx = [cols.get_loc(c) for c in volume_cols]
    network_idx = [cols.get_loc(c) for c in network_cols]
    usb_idx = [cols.get_loc(c) for c in usb_cols]

    login_error = group_error(login_idx)
    volume_error = group_error(volume_idx)
    network_error = group_error(network_idx)
    usb_error = group_error(usb_idx)

    # thresholds based on normal data
    def get_threshold(err):
        return err[:n_normal].mean() + 2 * err[:n_normal].std()

    login_thr = get_threshold(login_error)
    volume_thr = get_threshold(volume_error)
    network_thr = get_threshold(network_error)
    usb_thr = get_threshold(usb_error)

    # flags based on thresholds
    login_flag = (login_error > login_thr).astype(int)
    volume_flag = (volume_error > volume_thr).astype(int)
    network_flag = (network_error > network_thr).astype(int)
    usb_flag = (usb_error > usb_thr).astype(int)

    # output table
    output_df = pd.DataFrame({
        "user_id": user_ids,
        "login": login_flag,
        "volume": volume_flag,
        "network": network_flag,
        "usb": usb_flag
    })

    # IMPORTANT: aggregate per user
    output_df = output_df.groupby("user_id").max().reset_index()

    print("\nSample Output:\n")
    print(output_df.head())

    # save file
    output_df.to_excel("autoencoder_output.xlsx", index=False)

    print("\nSaved to autoencoder_output.xlsx\n")

    return output_df


# test run
if __name__ == "__main__":
    data = pd.read_csv("behavior_dataset.csv")
    run_autoencoder(data)