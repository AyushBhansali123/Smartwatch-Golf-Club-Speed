# 🏌️ Smartwatch-Golf-Club-Speed

Use smartwatch motion sensors (IMU) to predict golf swing speed using machine learning.  
This project is a **WORK IN PROGRESS** and will be updated frequently.

Currently, a **Random Forest** model is used for accurate and interpretable predictions based on summary statistics extracted from motion data.

📊 **Current number of swings:** **71**  
📫 Want to get involved? Email me!

---

## 📦 Project Overview

This project compares two approaches for estimating golf swing speed:

- **1D CNN** — uses full wrist motion time series as input
- **Statistical model** — uses mean, std, max, min features from motion data

---

## 📁 Folder Structure

- `src/`: Training scripts
- `models/`: Trained models (`.h5` or `.pkl`)
- `data/`: Cleaned example dataset
- `notebooks/`: (optional) Jupyter notebooks for visualization and analysis

---

## 📊 Results

| Model        | MAE (mph) | R² Score |
|--------------|-----------|----------|
| 1D CNN       | ~16.4     | ~0.00    |
| RandomForest | **2.88**  | **0.51** |

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
python src/swing_summary_model.py
