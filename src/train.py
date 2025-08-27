import numpy as np
import pandas as pd
import json
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score
from nni import get_next_parameter, report_final_result
from sklearn.model_selection import train_test_split

# Load features
X = np.load('data/X.npy')  # shape: (2096, 19)
y = np.load('data/y.npy')  # shape: (2096,), 1 = normal, -1 = anomaly

# Split for validation (unsupervised learning)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# Get params from NNI
params = get_next_parameter()
n_estimators = params['n_estimators']
contamination = params['contamination']

# Train model
clf = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=42)
clf.fit(X_train)

# Predict and evaluate
y_pred = clf.predict(X_val)
score = f1_score(y_val, y_pred, pos_label=-1)  # we treat -1 as anomaly

# Report to NNI
report_final_result(score)
