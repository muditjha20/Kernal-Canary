#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd

# Load the log file
log_path = '../data/HDFS_100k.log_structured.csv'
df = pd.read_csv(log_path)

# Show basic structure
print("Columns:", df.columns)
print("Shape:", df.shape)
df.head()


# In[21]:


import datetime
import datetime

# 1. Merge Date and Time columns into one datetime object
df['Timestamp'] = pd.to_datetime(
    df['Date'].astype(str) + df['Time'].astype(str).str.zfill(6),
    format='%y%m%d%H%M%S'
)

# 2. Drop unnecessary columns
df.drop(columns=['LineId', 'Date', 'Time', 'Pid'], inplace=True)

# 3. Sort logs by timestamp
df.sort_values(by='Timestamp', inplace=True)
df.reset_index(drop=True, inplace=True)

# Preview
df.head()


# In[22]:


# Example: Break logs into windows of 50 lines
WINDOW_SIZE = 50
log_windows = []

for i in range(0, len(df), WINDOW_SIZE):
    window = df.iloc[i:i + WINDOW_SIZE]
    if len(window) == WINDOW_SIZE:
        log_windows.append(window)

print("Total windows created:", len(log_windows))
print("First window:\n", log_windows[0].head())


# In[23]:


import numpy as np

# Get all unique EventIds in the dataset
unique_events = sorted(df['EventId'].unique().tolist())
print("Total unique EventIds:", len(unique_events))  # Should be ~28

# Build a lookup to convert EventId to index in feature vector
event_to_index = {event: idx for idx, event in enumerate(unique_events)}

# Convert each window to a frequency vector
window_vectors = []

for window in log_windows:
    vector = np.zeros(len(unique_events))  # start with all zeroes
    for event_id in window['EventId']:
        if event_id in event_to_index:
            vector[event_to_index[event_id]] += 1
    window_vectors.append(vector)

# Convert list to NumPy array
X = np.array(window_vectors)

# Check shape
print("Feature matrix shape:", X.shape)
print("Example vector (first window):", X[0])


# In[24]:


from sklearn.ensemble import IsolationForest
import numpy as np

# Initialize model
iso_forest = IsolationForest(
    n_estimators=200,       # Number of trees
    contamination=0.05,     # Expected proportion of anomalies
    random_state=42
)

# Fit model
iso_forest.fit(X)

# Get anomaly scores
scores = iso_forest.decision_function(X)  # higher = more normal
predictions = iso_forest.predict(X)       # 1 = normal, -1 = anomaly

# Count anomalies
num_anomalies = np.sum(predictions == -1)
print(f"Detected anomalies: {num_anomalies}/{len(X)} windows")

# Preview suspicious windows
for i, (pred, score) in enumerate(zip(predictions, scores)):
    if pred == -1:  # anomaly
        print(f"Window {i} → Anomaly score: {score:.4f}")
        print(log_windows[i].head(3))  # show first 3 logs in the window
        print("="*60)


# In[25]:


# Create synthetic labels: 95% normal, 5% anomaly
y_labels = np.ones(X.shape[0])
y_labels[predictions == -1] = -1  # Use your previous predictions as pseudo-anomalies

# Save
np.save('../data/X.npy', X)
np.save('../data/y.npy', y_labels)


# In[26]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import json
from pathlib import Path
import os

# Setup paths (works in both Jupyter and .py scripts)
if "__file__" in globals():
    ROOT = Path(__file__).resolve().parent
else:
    ROOT = Path.cwd().parent   # go up one level from /notebooks

# Verify paths
data_path = ROOT / 'data/X.npy'
params_path = ROOT / 'artifacts/best_model_params.json'

if not data_path.exists():
    raise FileNotFoundError(f"Data file not found: {data_path}")
if not params_path.exists():
    raise FileNotFoundError(f"Params file not found: {params_path}")

# Load data
X = np.load(data_path)

# Load best params from NNI
with open(params_path) as f:
    best_params = json.load(f)

# Re-train Isolation Forest with tuned params
model = IsolationForest(
    n_estimators=best_params['n_estimators'],
    contamination=best_params['contamination'],
    random_state=42
)
model.fit(X)

# Score all windows
scores = model.decision_function(X)  # higher = more normal
threshold = np.percentile(scores, best_params['contamination'] * 100)  # dynamic threshold
alerts = scores < threshold  # boolean array for anomalies

# Plotting
plt.figure(figsize=(14, 5))
plt.plot(scores, label='Anomaly Score', color='blue', marker='.')
plt.axhline(y=threshold, color='red', linestyle='--', label='Threshold')
plt.scatter(np.where(alerts), scores[alerts], color='red', label='Anomalies', zorder=5)

plt.title('Kernel Canary++ Anomaly Timeline')
plt.xlabel('Log Window Index')
plt.ylabel('Anomaly Score')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save and show
out_path = ROOT / 'artifacts/anomaly_chart.png'
plt.savefig(out_path)
plt.show()

print(f"Chart saved to: {out_path}")


# In[27]:


import numpy as np
import json
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path

if "__file__" in globals():
    ROOT = Path(__file__).resolve().parent
else:
    ROOT = Path.cwd().parent

# Create pseudo ground truth using baseline model
baseline_model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
baseline_model.fit(X)
y_true = baseline_model.predict(X)  # 1 = normal, -1 = anomaly

# Load best model params from NNI
with open(ROOT / "artifacts/best_model_params.json") as f:
    best = json.load(f)

#  Apply tuned model
tuned_model = IsolationForest(
    n_estimators=best["n_estimators"],
    contamination=best["contamination"],
    random_state=42
)
tuned_model.fit(X)
y_pred = tuned_model.predict(X)

# Step 4: Evaluation report
print("Kernel Canary++ Evaluation vs Baseline Labels")
print("-" * 50)

print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=["Anomaly", "Normal"], labels=[-1, 1]))

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred, labels=[-1, 1]))


# In[28]:


import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.ensemble import IsolationForest

if "__file__" in globals():
    ROOT = Path(__file__).resolve().parent
else:
    ROOT = Path.cwd().parent 

X = np.load(ROOT / "data/X.npy")

with open(ROOT / "artifacts/best_model_params.json") as f:
    params = json.load(f)

model = IsolationForest(
    n_estimators=params["n_estimators"],
    contamination=params["contamination"],
    random_state=42
)
model.fit(X)

# Save model
Path("model").mkdir(exist_ok=True)
joblib.dump(model, "model/model.pkl")
print("Model saved.")


# In[29]:


import requests

sample = {
    "window": [0, 0, 2, 1, 0, 0, 0, 1, 0, 4, 0, 11, 0, 2, 11, 2, 0, 0, 8]
}

res = requests.post("http://localhost:5000/score", json=sample)

print("Status code:", res.status_code)
print("Response text:", res.text)  # 👈 helps debug if not JSON

try:
    print("JSON:", res.json())
except Exception as e:
    print("JSON decode failed:", e)


# In[ ]:





# In[ ]:




