# 🏌️ Smartwatch-Golf-Club-Speed

Use smartwatch motion sensors (IMU) to predict golf swing speed using machine learning. WORK IN PROGRESS

## 📦 Project Overview

This project compares two approaches for estimating golf swing speed:

- **1D CNN** using full wrist motion time series
- **Statistical model** using mean, std, max, min features

## 📁 Folder Structure

- `src/`: All training scripts
- `models/`: Trained models (.h5 or .pkl)
- `data/`: Cleaned example dataset
- `notebooks/`: Analysis & visualization notebooks

## 📊 Results

| Model        | MAE (mph) | R² Score |
|--------------|-----------|----------|
| CNN (1D)     | ~16.4     | ~0.00    |
| RandomForest | **2.88**  | **0.51** |

## 🚀 How to Run

```bash
pip install -r requirements.txt
python src/swing_summary_model.py
