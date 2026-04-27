import pandas as pd
import numpy as np

# =========================
# LOAD DATASETS
# =========================
logon = pd.read_csv("logon.csv")
http = pd.read_csv("http.csv", header=None)
http.columns = ["id", "date", "user", "pc", "url"]
device = pd.read_csv("device.csv")

# =========================
# CLEANING
# =========================
logon.columns = logon.columns.str.strip()
device.columns = device.columns.str.strip()

# =========================
# LOGIN FEATURES
# =========================
logon_logins = logon[logon["activity"] == "Logon"]

login_counts = logon_logins.groupby("user").size().reset_index(name="logins")

# Extract login hour (REAL FEATURE)
logon_logins["date"] = pd.to_datetime(logon_logins["date"])
logon_logins["login_hour"] = logon_logins["date"].dt.hour

login_hour_avg = logon_logins.groupby("user")["login_hour"].mean().reset_index()

# =========================
# WEB FEATURES
# =========================
web_counts = http.groupby("user").size().reset_index(name="web_visits")

# Simulate file access from web activity
web_counts["file_access_count"] = (web_counts["web_visits"] * np.random.uniform(0.2, 0.5)).astype(int)

# Simulate network usage
web_counts["network_usage"] = web_counts["web_visits"] * np.random.uniform(0.5, 1.5)

# =========================
# USB FEATURES
# =========================
device_connect = device[device["activity"] == "Connect"]
usb_counts = device_connect.groupby("user").size().reset_index(name="usb_usage")

# =========================
# MERGE ALL DATA
# =========================
dataset = login_counts.merge(web_counts, on="user", how="outer")
dataset = dataset.merge(usb_counts, on="user", how="outer")
dataset = dataset.merge(login_hour_avg, on="user", how="outer")

# Fill missing values
dataset = dataset.fillna(0)

# =========================
# ROLE ASSIGNMENT
# =========================
roles = ["Admin", "HR", "Finance", "Engineer", "Intern"]
dataset["role"] = np.random.choice(roles, size=len(dataset))

# =========================
# FINAL SAVE
# =========================
dataset.to_csv("behavior_dataset.csv", index=False)

print("✅ Advanced Dataset Created Successfully!")
print(dataset.head())
