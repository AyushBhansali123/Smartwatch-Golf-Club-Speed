
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib
import os

# Load the dataset
df = pd.read_csv("/Users/ayushbhansali/Desktop/Elepaio Data/Cleaned_Swing_Data.csv")

# Filter to include only swings with exactly 262 samples
swing_counts = df.groupby('swing_id').size()
valid_swing_ids = swing_counts[swing_counts == 262].index
df = df[df['swing_id'].isin(valid_swing_ids)]

# Compute summary stats for each swing
features = ['accX', 'accY', 'accZ', 'rotX', 'rotY', 'rotZ']
agg_funcs = ['mean', 'std', 'max', 'min']
summary_df = df.groupby('swing_id')[features].agg(agg_funcs)
summary_df.columns = ['_'.join(col) for col in summary_df.columns]  # flatten MultiIndex

# Target variable
y = df.groupby('swing_id')['trackman_speed'].first().loc[summary_df.index]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(summary_df.values, y.values, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a random forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Test MAE: {mae:.2f} mph")
print(f"R^2 Score: {r2:.3f}")

# Plot predicted vs actual
plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel("Actual Swing Speed (mph)")
plt.ylabel("Predicted Swing Speed (mph)")
plt.title("Predicted vs Actual Swing Speeds")
plt.grid(True)
plt.show()

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/random_forest_swing_model.pkl")
print("Model saved to models/random_forest_swing_model.pkl")
