import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("behavior_dataset.csv")

print("\n🔹 ML Input Matrix Sample:")
print(df[["logins", "web_visits", "usb_usage"]].head())

# =========================
# FEATURE ENGINEERING (ADVANCED UEBA STYLE)
# =========================

# Total activity
df["activity_score"] = df["logins"] + df["web_visits"] + df["usb_usage"]

# Ratios
df["web_log_ratio"] = df["web_visits"] / (df["logins"] + 1)
df["usb_log_ratio"] = df["usb_usage"] / (df["logins"] + 1)

# Statistical deviation (VERY IMPORTANT - INDUSTRY)
df["login_zscore"] = (df["logins"] - df["logins"].mean()) / df["logins"].std()
df["web_zscore"] = (df["web_visits"] - df["web_visits"].mean()) / df["web_visits"].std()

# =========================
# NORMALIZATION
# =========================

features = [
    "logins",
    "web_visits",
    "usb_usage",
    "activity_score",
    "web_log_ratio",
    "usb_log_ratio",
    "login_zscore",
    "web_zscore"
]

scaler_input = MinMaxScaler()
X_scaled = scaler_input.fit_transform(df[features])

# =========================
# MODELS (ENSEMBLE APPROACH)
# =========================

login_model = IsolationForest(contamination=0.05, random_state=42)
web_model = IsolationForest(contamination=0.05, random_state=42)
usb_model = IsolationForest(contamination=0.05, random_state=42)
behavior_model = IsolationForest(contamination=0.05, random_state=42)

login_model.fit(df[["logins"]])
web_model.fit(df[["web_visits", "web_log_ratio"]])
usb_model.fit(df[["usb_usage", "usb_log_ratio"]])
behavior_model.fit(X_scaled)

# =========================
# ANOMALY SCORES
# =========================

df["login_score"] = login_model.decision_function(df[["logins"]])
df["web_score"] = web_model.decision_function(df[["web_visits", "web_log_ratio"]])
df["usb_score"] = usb_model.decision_function(df[["usb_usage", "usb_log_ratio"]])
df["behavior_score"] = behavior_model.decision_function(X_scaled)

# Normalize scores
scaler = MinMaxScaler()
score_cols = ["login_score", "web_score", "usb_score", "behavior_score"]
df[score_cols] = scaler.fit_transform(df[score_cols])

# =========================
# CONTEXT-AWARE RISK SCORING (VERY ADVANCED)
# =========================

# Weighted risk (behavior more important)
df["risk_score"] = (
    0.15 * df["login_score"] +
    0.30 * df["web_score"] +
    0.25 * df["usb_score"] +
    0.30 * df["behavior_score"]
)

# Add statistical anomaly influence
df["risk_score"] += 0.1 * abs(df["login_zscore"])
df["risk_score"] += 0.1 * abs(df["web_zscore"])

# Normalize final risk
df["risk_score"] = MinMaxScaler().fit_transform(df[["risk_score"]])

# =========================
# DYNAMIC THRESHOLDING
# =========================

threshold_high = df["risk_score"].quantile(0.95)
threshold_medium = df["risk_score"].quantile(0.85)

# =========================
# RISK LEVEL CLASSIFICATION
# =========================

def classify(score):
    if score >= threshold_high:
        return "High Risk"
    elif score >= threshold_medium:
        return "Medium Risk"
    else:
        return "Low Risk"

df["risk_level"] = df["risk_score"].apply(classify)

# =========================
# ALERT ENGINE (SOC STYLE)
# =========================

def generate_alert(row):
    if row["risk_level"] == "High Risk":
        return "🚨 CRITICAL: Insider Threat Suspected"
    elif row["risk_level"] == "Medium Risk":
        return "⚠️ WARNING: Suspicious Behavior"
    else:
        return "✅ Normal"

df["alert"] = df.apply(generate_alert, axis=1)

# =========================
# EXPLAINABILITY ENGINE
# =========================

def explain(row):
    reasons = []

    if row["login_score"] > 0.8:
        reasons.append("Abnormal login frequency")
    if row["web_score"] > 0.8:
        reasons.append("Excessive web usage")
    if row["usb_score"] > 0.8:
        reasons.append("Suspicious USB activity")
    if abs(row["login_zscore"]) > 2:
        reasons.append("Login deviation from normal")
    if abs(row["web_zscore"]) > 2:
        reasons.append("Web activity spike")

    return ", ".join(reasons) if reasons else "Normal behavior"

df["reason"] = df.apply(explain, axis=1)

# =========================
# SAVE OUTPUT
# =========================

df.to_csv("final_security_output.csv", index=False)

# =========================
# DISPLAY
# =========================

print("\n🚀 FINAL SECURITY SYSTEM OUTPUT\n")

print(df[[
    "user",
    "risk_score",
    "risk_level",
    "alert",
    "reason"
]].head(20))

high_risk = df[df["risk_level"] == "High Risk"]

print("\n🚨 HIGH RISK USERS:\n")
print(high_risk[["user", "risk_score", "reason"]])

print("\n📊 SUMMARY:")
print("Total Users:", len(df))
print("High Risk Users:", len(high_risk))

# =========================
# VISUALIZATION (SOC DASHBOARD STYLE)
# =========================

# Distribution
plt.figure()
plt.hist(df["risk_score"], bins=50)
plt.title("Risk Score Distribution")
plt.xlabel("Risk Score")
plt.ylabel("Users")
plt.show()

# Scatter
plt.figure()
plt.scatter(df["logins"], df["web_visits"])
plt.scatter(high_risk["logins"], high_risk["web_visits"])
plt.title("High Risk Users Highlighted")
plt.xlabel("Logins")
plt.ylabel("Web Visits")
plt.show()