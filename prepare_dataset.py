import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

print("="*60)
print("COMPLETE DATASET PREPARATION - ALL REQUIREMENTS")
print("="*60)

# =========================
# LOAD DATASETS
# =========================
print("\n1. Loading files...")
logon = pd.read_csv("logon.csv")
http = pd.read_csv("http.csv", header=None)
http.columns = ["id", "date", "user", "pc", "url"]
device = pd.read_csv("device.csv")

# Clean and convert
logon.columns = logon.columns.str.strip()
device.columns = device.columns.str.strip()
logon['date'] = pd.to_datetime(logon['date'])
http['date'] = pd.to_datetime(http['date'])
device['date'] = pd.to_datetime(device['date'])

print(f"   Logon: {len(logon):,} rows, {logon['user'].nunique()} users")
print(f"   HTTP: {len(http):,} rows, {http['user'].nunique()} users")

# =========================
# 1. USER IDs (✅ Green Tick)
# =========================
all_users = logon['user'].unique()
print(f"\n✅ User IDs: {len(all_users)} unique users")

# =========================
# 2. TIMESTAMPS & LOGIN TIMES (✅ Green Tick)
# =========================
print("\n2. Extracting timestamps & login patterns...")
logon_logins = logon[logon["activity"] == "Logon"].copy()
logon_logins = logon_logins.sort_values(['user', 'date'])

# Raw timestamps for each user
timestamps_data = []
for _, row in logon_logins.iterrows():
    timestamps_data.append({
        'user': row['user'],
        'timestamp': row['date'],
        'login_hour': row['date'].hour,
        'day_of_week': row['date'].dayofweek,
        'is_weekend': row['date'].dayofweek >= 5,
        'is_after_hours': (row['date'].hour < 7) or (row['date'].hour > 19)
    })

timestamps_df = pd.DataFrame(timestamps_data)
timestamps_df.to_csv("user_timestamps.csv", index=False)
print(f"   ✅ Saved {len(timestamps_df)} timestamp records to user_timestamps.csv")

# Per-user login statistics
login_stats = []
for user, group in logon_logins.groupby('user'):
    hours = group['date'].dt.hour
    dayofweek = group['date'].dt.dayofweek
    
    time_diffs = group['date'].diff().dt.total_seconds() / 3600
    
    stats = {
        'user': user,
        'total_logins': len(group),
        'avg_login_hour': hours.mean(),
        'std_login_hour': hours.std() if len(hours) > 1 else 0,
        'weekend_login_rate': (dayofweek >= 5).mean(),
        'after_hours_rate': ((hours < 7) | (hours > 19)).mean(),
        'avg_hours_between_logins': time_diffs.mean() if len(time_diffs) > 1 else 0,
        'min_login_hour': hours.min(),
        'max_login_hour': hours.max()
    }
    login_stats.append(stats)

login_features = pd.DataFrame(login_stats)
print(f"   ✅ Processed login features for {len(login_features)} users")

# =========================
# 3. FILE COUNTS (✅ Green Tick)
# =========================
print("\n3. Extracting file counts...")
web_stats = []
for user, group in http.groupby('user'):
    is_download = group['url'].str.contains('download|exe|zip|tar|gz|rar|pdf|doc|xls', case=False, na=False)
    is_upload = group['url'].str.contains('upload|submit|post', case=False, na=False)
    
    stats = {
        'user': user,
        'total_web_visits': len(group),
        'file_download_count': is_download.sum(),
        'file_download_rate': is_download.mean(),
        'file_upload_count': is_upload.sum(),
        'file_upload_rate': is_upload.mean(),
        'unique_sites': group['url'].nunique()
    }
    web_stats.append(stats)

web_features = pd.DataFrame(web_stats)
print(f"   ✅ File counts: avg downloads = {web_features['file_download_count'].mean():.1f} per user")

# =========================
# 4. NETWORK USAGE (✅ Green Tick)
# =========================
print("\n4. Estimating network usage...")

def estimate_network_mb(row):
    base_network = row['total_web_visits'] * 0.5
    download_network = row['file_download_count'] * 5.0
    upload_network = row['file_upload_count'] * 2.0
    return base_network + download_network + upload_network

web_features['network_usage_mb'] = web_features.apply(estimate_network_mb, axis=1)
web_features['network_anomaly_score'] = web_features['network_usage_mb'] / (web_features['network_usage_mb'].max() + 1)

print(f"   ✅ Network usage: avg = {web_features['network_usage_mb'].mean():.1f} MB, max = {web_features['network_usage_mb'].max():.1f} MB")

# =========================
# 5. ROLES (✅ Green Tick)
# =========================
print("\n5. Assigning roles based on behavior...")

def assign_role(row):
    if row['file_download_rate'] > 0.3 and row['network_usage_mb'] > 5000:
        return 'Engineer'
    elif row['file_download_rate'] > 0.2 and row.get('after_hours_rate', 0) > 0.3:
        return 'Developer'
    elif row['total_web_visits'] > 2000:
        return 'IT_Admin'
    elif row.get('weekend_login_rate', 0) > 0.2:
        return 'Support'
    else:
        return 'General'

# Merge to get role features
temp_df = login_features.merge(web_features, on='user', how='left').fillna(0)
temp_df['role'] = temp_df.apply(assign_role, axis=1)

print(f"   ✅ Role distribution:")
print(temp_df['role'].value_counts())

# =========================
# 6. USB DEVICE USAGE (✅ Green Tick)
# =========================
print("\n6. Extracting USB device usage...")
usb_stats = []
if 'user' in device.columns and len(device) > 0:
    device_connect = device[device["activity"].str.contains("Connect", case=False, na=False)]
    
    for user, group in device_connect.groupby('user'):
        stats = {
            'user': user,
            'usb_connect_count': len(group),
            'unique_usb_devices': group['pc'].nunique() if 'pc' in group.columns else len(group)
        }
        usb_stats.append(stats)

usb_features = pd.DataFrame(usb_stats)
if len(usb_features) == 0:
    usb_features = pd.DataFrame({'user': login_features['user']})
    usb_features['usb_connect_count'] = 0
    usb_features['unique_usb_devices'] = 0

print(f"   ✅ USB data: {len(usb_features[usb_features['usb_connect_count'] > 0])} users used USB")

# =========================
# 7. MERGE EVERYTHING (✅ Green Tick)
# =========================
print("\n7. Merging all features...")
final_dataset = login_features.merge(web_features, on='user', how='left')
final_dataset = final_dataset.merge(usb_features, on='user', how='left')
final_dataset = final_dataset.merge(temp_df[['user', 'role']], on='user', how='left')
final_dataset = final_dataset.fillna(0)

# =========================
# 7.5 ADD ENGINEERED FEATURES (NEW - For better anomaly detection)
# =========================
print("\n7.5 Creating engineered features for better detection...")

# Feature 1: Download to upload ratio (insiders download more than they upload)
final_dataset['download_upload_ratio'] = (
    final_dataset['file_download_count'] / (final_dataset['file_upload_count'] + 1)
)

# Feature 2: Unusual hour score (working at 2am is more suspicious than 8am)
final_dataset['unusual_hour_score'] = (
    (final_dataset['avg_login_hour'] < 6) | (final_dataset['avg_login_hour'] > 20)
).astype(int)

# Feature 3: Combined risk multiplier (when multiple bad behaviors happen together)
final_dataset['combined_risk'] = (
    final_dataset['after_hours_rate'] * 
    final_dataset['file_download_rate'] * 
    (final_dataset['usb_connect_count'] > 0).astype(int)
)

print(f"   ✅ Added 3 engineered features")

# =========================
# 8. NORMALIZE TO 0-1 (✅ Green Tick) - UPDATE THIS LIST
# =========================
print("\n8. Normalizing features to 0-1...")
features_to_normalize = [
    'total_logins', 'avg_login_hour', 'std_login_hour',
    'weekend_login_rate', 'after_hours_rate', 'avg_hours_between_logins',
    'total_web_visits', 'file_download_count', 'file_download_rate',
    'file_upload_count', 'file_upload_rate', 'network_usage_mb',
    'usb_connect_count', 'unique_usb_devices',
    'download_upload_ratio', 'unusual_hour_score', 'combined_risk'  # ADD THESE 3
]


# Only normalize columns that exist
existing_features = [f for f in features_to_normalize if f in final_dataset.columns]
scaler = MinMaxScaler(feature_range=(0, 1))
final_dataset[existing_features] = scaler.fit_transform(final_dataset[existing_features])

print(f"   ✅ All {len(existing_features)} features normalized to 0-1")

# =========================
# 9. SEPARATE ANOMALY SCORES (✅ Green Tick)
# =========================
print("\n9. Calculating separate anomaly scores...")

# Login Anomaly Score
final_dataset['anomaly_login'] = (
    final_dataset['after_hours_rate'] * 0.4 +
    final_dataset['weekend_login_rate'] * 0.3 +
    final_dataset['std_login_hour'] * 0.3
)

# Volume/File Anomaly Score
final_dataset['anomaly_volume'] = (
    final_dataset['file_download_rate'] * 0.5 +
    final_dataset['file_upload_rate'] * 0.3 +
    final_dataset['total_web_visits'] * 0.2
)

# Network Anomaly Score
final_dataset['anomaly_network'] = (
    final_dataset['network_usage_mb'] * 0.6 +
    final_dataset['file_upload_rate'] * 0.4
)

# USB Anomaly Score
final_dataset['anomaly_usb'] = final_dataset['usb_connect_count']

# Overall Risk Score (weighted)
final_dataset['anomaly_overall'] = (
    final_dataset['anomaly_login'] * 0.25 +
    final_dataset['anomaly_volume'] * 0.35 +
    final_dataset['anomaly_network'] * 0.25 +
    final_dataset['anomaly_usb'] * 0.15
)

print(f"   ✅ Created 5 separate anomaly scores per user")

# =========================
# 10. GROUND TRUTH LABELS (Fixed - No syntax errors)
# =========================
print("\n10. Creating ground truth labels...")
final_dataset['is_insider_threat'] = 0

# Create conditions separately to avoid syntax issues
condition1 = final_dataset['anomaly_overall'] > 0.6
condition2 = final_dataset['anomaly_login'] > 0.7
condition3 = final_dataset['anomaly_volume'] > 0.7
condition4 = final_dataset['anomaly_network'] > 0.7
condition5 = final_dataset['anomaly_usb'] > 0.5

# Top 5% threshold
top_threshold = final_dataset['anomaly_overall'].quantile(0.95)
condition6 = final_dataset['anomaly_overall'] > top_threshold

# Combine all conditions
final_dataset.loc[condition1 | condition2 | condition3 | condition4 | condition5 | condition6, 'is_insider_threat'] = 1

print(f"   ✅ Marked {final_dataset['is_insider_threat'].sum()} users as threats ({final_dataset['is_insider_threat'].mean():.1%})")

# =========================
# SAVE EVERYTHING
# =========================
print("\n11. Saving all files...")
final_dataset.to_csv("behavior_dataset.csv", index=False)
print("   ✅ behavior_dataset.csv (main dataset)")

# Export summary for dashboard
summary = final_dataset[['user', 'role', 'anomaly_login', 'anomaly_volume', 
                          'anomaly_network', 'anomaly_usb', 'anomaly_overall', 
                          'is_insider_threat']]
summary.to_csv("anomaly_scores_separate.csv", index=False)
print("   ✅ anomaly_scores_separate.csv (separate scores for dashboard)")

# Export normalized features only (for autoencoder)
norm_features = final_dataset[['user'] + existing_features]
norm_features.to_csv("normalized_features_for_autoencoder.csv", index=False)
print("   ✅ normalized_features_for_autoencoder.csv (for your friend)")

print("\n" + "="*60)
print("🎉 ALL GREEN TICKS - DATASET COMPLETE!")
print("="*60)
print(f"\n✅ User IDs: {len(final_dataset)} users")
print(f"✅ Timestamps: Saved to user_timestamps.csv")
print(f"✅ File counts: Avg {final_dataset['file_download_count'].mean():.2f} downloads/user")
print(f"✅ Network usage: Avg {final_dataset['network_usage_mb'].mean():.2f} MB/user")
print(f"✅ Roles: {final_dataset['role'].nunique()} types")
print(f"✅ Normalized: {len(existing_features)} features in 0-1 range")
print(f"✅ Separate anomalies: 5 scores per user")
print(f"✅ Ground truth: {final_dataset['is_insider_threat'].sum()} threats marked")

