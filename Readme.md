# Kernel Canary++ 🐧  
**Log-Based Anomaly Detection for System Security and Monitoring**

Kernel Canary++ is a lightweight, real-time anomaly detection system designed to monitor structured system logs and flag suspicious behavior. It leverages an optimized Isolation Forest model, automated hyperparameter tuning, and a deployable REST API to detect rare events that may indicate system faults, intrusions, or unusual activity.

> 🧠 Inspired by Microsoft’s SR-CNN research on log-based anomaly detection.

---

## 🔍 Problem Statement

Modern operating systems generate massive logs, making manual inspection infeasible. Kernel Canary++ addresses this by:
- Converting structured logs into frequency-based vector representations
- Detecting anomalous patterns in event distributions
- Providing a real-time scoring interface via a Python REST API

---

## 💡 Core Features

- ✅ **Log Window Vectorization** — Converts system logs into 19-dimensional frequency vectors using event templates
- 🤖 **Unsupervised Anomaly Detection** — Uses Isolation Forest trained on pseudo-labels to catch outliers
- 🔧 **AutoML with NNI** — Automatically finds best model parameters using Neural Network Intelligence (NNI)
- 🧠 **Smart Thresholding** — Dynamically calculates an anomaly threshold based on decision scores
- 🌐 **Live Inference API** — Accepts event vectors and responds with anomaly predictions in real-time
- 📈 **Visual Anomaly Timeline** — Generates a timeline of anomaly scores to help identify bursts and shifts

---

## 📊 Model Performance

| Metric        | Value     |
|---------------|-----------|
| **Accuracy**  | 99.0%     |
| **F1-Score** (Anomaly Class) | 0.91 |
| **Precision** | 0.86      |
| **Recall**    | 0.96      |
| **Support**   | 2096 log windows |

**Confusion Matrix**:
```
               Predicted
             | Anomaly | Normal
-------------|---------|--------
Actual:Anom  |   101   |   4
Actual:Norm  |   16    | 1975
```

> 📌 Tuned using NNI on a structured dataset of 100,000+ HDFS logs  
> 🎯 Optimal parameters: `n_estimators=300`, `contamination=5.55%`

---

## 🛠️ Tech Stack

- **Language**: Python 3.11
- **Libraries**: `scikit-learn`, `NumPy`, `Matplotlib`, `Flask`, `joblib`, `pandas`
- **AutoML**: [Microsoft NNI (Neural Network Intelligence)](https://github.com/microsoft/nni)
- **Deployment**: Flask REST API for live anomaly scoring
- **Format**: Structured logs processed into frequency vectors of 19 canonical event templates

---

## 🧪 Example Prediction (API)

**Input**:  
```json
{
  "window": [0, 0, 2, 1, 0, 0, 0, 1, 0, 4, 0, 11, 0, 2, 11, 2, 0, 0, 8]
}
```

**Output**:  
```json
{
  "is_anomaly": true
}
```

---

## 🔬 Why This Matters

- 🌐 **Security Monitoring**: Detect hidden intrusions in kernel logs
- ⚙️ **Fault Detection**: Spot unusual runtime behavior before systems crash
- 📉 **Noise Reduction**: Focus attention only on statistically rare and meaningful events

---

## 👨‍💻 Author

**Mudit Mayank Jha**  
B.Sc. Computer Science @ UWI | Exchange @ University of Richmond  
Passionate about applied machine learning, system-level tools, and security.  
[GitHub](https://github.com/muditjha20)

---

> Built to explore the intersection of system security, unsupervised ML, and real-time monitoring.  
> Inspired by Microsoft’s commitment to scalable, intelligent infrastructure.
