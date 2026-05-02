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

    #check requied column
    if "user_id" not in data.columns:
        raise ValueError("Dataset must contain 'user_id' column")

    #handle non numeric data
    feature_cols = data.columns.drop("user_id")

    # keep only numeric columns
    numeric_data = data[feature_cols].select_dtypes(include=[np.number])

    if numeric_data.shape[1] == 0:
        raise ValueError("No numeric features found in dataset")

    #scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric_data)

    user_ids = data["user_id"].values

    #define normal data as first 90% of samples
    # assume majority is normal
    n_normal = int(0.9 * len(X_scaled))

    X_train = X_scaled[:n_normal]
    X_test = X_scaled

    print(f"Using first {n_normal} samples as normal data")

    #model setup
    model = Autoencoder(X_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)

    #training loop
    for epoch in range(50):
        optimizer.zero_grad()
        output = model(X_train_tensor)
        loss = criterion(output, X_train_tensor)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    #anomaly detection
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    with torch.no_grad():
        reconstructed = model(X_test_tensor)
        errors = torch.mean((X_test_tensor - reconstructed)**2, dim=1).numpy()

    threshold = errors[:n_normal].mean() + 2 * errors[:n_normal].std()
    predictions = (errors > threshold).astype(int)

    print(f"\nThreshold: {threshold:.4f}")
    print(f"Total anomalies detected: {predictions.sum()}")

    #feature level error analysis
    feature_errors = (X_test_tensor - reconstructed)**2
    feature_errors = feature_errors.numpy()

    print("\nSample anomalies:\n")

    count = 0
    for i in range(len(predictions)):
        if predictions[i] == 1:
            feature = numeric_data.columns[np.argmax(feature_errors[i])]
            print(f"Index {i} → {feature} spike (Error: {errors[i]:.2f})")
            count += 1
            if count >= 10:
                break

    #user-level risk scoring
    results = pd.DataFrame({
        "user_id": user_ids,
        "error": errors,
        "is_anomaly": predictions
    })

    user_risk = results.groupby("user_id").agg({
        "error": "mean",
        "is_anomaly": "sum"
    }).reset_index()

    user_risk.columns = ["user_id", "avg_error", "num_anomalies"]

    user_risk["risk_score"] = (
        user_risk["avg_error"] * 0.7 +
        user_risk["num_anomalies"] * 0.3
    )

    print("\nTop risky users:\n")
    print(user_risk.sort_values(by="risk_score", ascending=False).head(5))

    print("\nDone.\n")

    return predictions, errors, user_risk


#optional test run
if __name__ == "__main__":
    data = pd.read_csv("behavior_dataset.csv")
    run_autoencoder(data)