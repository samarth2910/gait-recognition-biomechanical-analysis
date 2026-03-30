# 🚶‍♂️ Gait Recognition System (Open-Set Biometric Identification)

A deep learning–based gait recognition system that identifies individuals from walking patterns using pose estimation and temporal modeling.
The system is designed for **open-set recognition**, meaning it can also detect **unknown / intruder subjects**.

---

## 📌 Overview

This project uses **pose-based gait features** extracted from video and feeds them into an **LSTM neural network** for classification.
To improve real-world robustness, a **multi-stage decision system** is applied using:

* Entropy-based uncertainty rejection
* Confidence thresholding
* Margin-based separation
* Weighted voting across sequences
* Sequence quality filtering

---

## 🧠 Key Features

* ✅ **Deep Learning Model (LSTM)** for temporal gait analysis
* ✅ **Open-set recognition** (detects unknown people)
* ✅ **Entropy-based uncertainty filtering**
* ✅ **Weighted voting across sequences**
* ✅ **Sequence quality filtering (noise removal)**
* ✅ **Per-identity adaptive thresholds**
* ✅ **Real-time Streamlit interface**

---

## 🏗️ System Architecture

```
Video Input
   ↓
MediaPipe Pose Extraction
   ↓
Feature Engineering (7 gait features)
   ↓
Sequence Generation (Time-series)
   ↓
LSTM Model (Deep Learning)
   ↓
Decision Engine:
   - Entropy Gate
   - Confidence Gate
   - Margin Gate
   - Agreement Gate
   ↓
Final Output (Identity / Unknown)
```

---

## 📊 Features Used

The model uses **7 biomechanical gait features**:

| # | Feature              | Description             |
| - | -------------------- | ----------------------- |
| 1 | Right knee flexion   | Hip–Knee–Ankle angle    |
| 2 | Left knee flexion    | Hip–Knee–Ankle angle    |
| 3 | Right hip flexion    | Shoulder–Hip–Knee angle |
| 4 | Left hip flexion     | Shoulder–Hip–Knee angle |
| 5 | Right ankle movement | Knee–Ankle–Foot angle   |
| 6 | Left ankle movement  | Knee–Ankle–Foot angle   |
| 7 | Torso tilt           | Body lean from vertical |

---

## 🔍 Decision Logic (Open-Set Detection)

The system applies **4 rejection gates**:

1. **Entropy Gate**

   * Rejects uncertain predictions
2. **Confidence Gate**

   * Ensures minimum prediction strength
3. **Margin Gate**

   * Ensures separation between top classes
4. **Agreement Gate**

   * Ensures consistency across sequences

👉 This makes the system robust against **intruders and noisy inputs**

---

## 🧪 Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* MediaPipe
* NumPy / SciPy
* Streamlit

---

## 🚀 Installation

```bash
git clone https://github.com/your-username/gait-recognition.git
cd gait-recognition
pip install -r requirements.txt
```

---

## ▶️ Usage

```bash
streamlit run app.py
```

Then:

* Upload a walking video
* System will analyze and display identity or "Unknown"

---

## ⚙️ Configuration

You can tune detection thresholds:

* Entropy threshold
* Confidence threshold
* Margin threshold
* Agreement threshold

Also supports:

* **Per-identity adaptive tuning**
* **Sequence quality filtering**

---

## 📁 Project Structure

```
├── models/                # Trained models (ignored in Git)
├── videos/                # Dataset (ignored)
├── app.py                 # Main Streamlit app
├── bilstm.py              # Model training
├── gait_analyzer.py       # Core logic
├── requirements.txt
└── README.md
```

---

## ⚠️ Notes

* Model files (`.keras`, `.pkl`) are not included due to size
* Dataset videos are excluded from the repository
* Ensure consistent video quality for best results

---

## 📈 Future Improvements

* BiLSTM / Attention-based models
* Multi-view gait recognition
* Real-time CCTV integration
* Larger dataset for better generalization

---

## 👨‍💻 Author

**Samarth Shetty**

---

## ⭐ If you found this useful, consider starring the repo!
