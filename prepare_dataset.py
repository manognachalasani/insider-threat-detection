import pandas as pd

# Load datasets
logon = pd.read_csv("logon.csv")
http = pd.read_csv("http.csv", header=None)
http.columns = ["id", "date", "user", "pc", "url"]
device = pd.read_csv("device.csv")

# -----------------------------
# LOGIN FEATURES
# -----------------------------
logon_logins = logon[logon["activity"] == "Logon"]
login_counts = logon_logins.groupby("user").size().reset_index(name="logins")

# -----------------------------
# WEB FEATURES
# -----------------------------
web_counts = http.groupby("user").size().reset_index(name="web_visits")

# -----------------------------
# USB FEATURES
# -----------------------------
device_connect = device[device["activity"] == "Connect"]
usb_counts = device_connect.groupby("user").size().reset_index(name="usb_usage")

# -----------------------------
# MERGE DATA
# -----------------------------
dataset = login_counts.merge(web_counts, on="user", how="outer")
dataset = dataset.merge(usb_counts, on="user", how="outer")

dataset = dataset.fillna(0)

# -----------------------------
# SAVE DATASET
# -----------------------------
dataset.to_csv("behavior_dataset.csv", index=False)

print("✅ Dataset Created Successfully!")
print(dataset.head())