import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# --- 1. Generate Simulated "Normal" Sensor Data ---
# Let's simulate 'vibration amplitude' over time for a normally operating appliance.
# We'll create a time series with some baseline, noise, and a slight trend.
np.random.seed(42) # for reproducibility

num_points_normal = 500
normal_data = np.random.normal(loc=10, scale=1, size=num_points_normal) + np.linspace(0, 2, num_points_normal)
time_normal = pd.date_range(start='2025-01-01', periods=num_points_normal, freq='H')

# Create a DataFrame for normal data
df_normal = pd.DataFrame({'timestamp': time_normal, 'vibration_amplitude': normal_data})

# --- 2. Introduce Simulated Anomalies ---
# We'll add a few types of anomalies:
# a) A sudden, high spike
# b) A sustained, slightly higher level
# c) A drop

# Anomaly 1: Sudden Spike
spike_start_index = 150
spike_end_index = 155
# Corrected calculation for size to include both start and end indices
spike_values = np.random.normal(loc=25, scale=2, size=(spike_end_index - spike_start_index + 1))
df_normal.loc[spike_start_index:spike_end_index, 'vibration_amplitude'] = spike_values

# Anomaly 2: Sustained High Level
high_level_start_index = 300
high_level_end_index = 350
# Corrected calculation for size
high_level_values = np.random.normal(loc=15, scale=0.5, size=(high_level_end_index - high_level_start_index + 1))
df_normal.loc[high_level_start_index:high_level_end_index, 'vibration_amplitude'] = high_level_values

# Anomaly 3: A drop (negative anomaly)
drop_start_index = 450
drop_end_index = 460
# Corrected calculation for size
drop_values = np.random.normal(loc=3, scale=0.5, size=(drop_end_index - drop_start_index + 1))
df_normal.loc[drop_start_index:drop_end_index, 'vibration_amplitude'] = drop_values


# --- 3. Prepare Data for Isolation Forest ---
# The Isolation Forest model expects a 2D array.
# We only have one feature ('vibration_amplitude') for simplicity here.
data_for_model = df_normal[['vibration_amplitude']]

# --- 4. Train the Isolation Forest Model ---
# We train the model on the entire dataset, including anomalies.
# The model will learn to distinguish "normal" behavior.
# 'contamination' is the expected proportion of outliers in the data.
# For demonstration, we can set a small value (e.g., 0.05 or 5%)
# In a real scenario, you'd train on purely normal data if possible.
model = IsolationForest(contamination=0.05, random_state=42)
model.fit(data_for_model)

# --- 5. Predict Anomaly Scores and Labels ---
# decision_function gives the anomaly score (lower means more anomalous)
# predict gives -1 for outliers and 1 for inliers
df_normal['anomaly_score'] = model.decision_function(data_for_model)
df_normal['anomaly_label'] = model.predict(data_for_model)

# --- 6. Visualize the Results ---
plt.figure(figsize=(14, 7))

# Plot normal data points
plt.plot(df_normal['timestamp'], df_normal['vibration_amplitude'], label='Simulated Vibration Data', color='blue', alpha=0.7)

# Highlight detected anomalies
anomalies = df_normal[df_normal['anomaly_label'] == -1]
plt.scatter(anomalies['timestamp'], anomalies['vibration_amplitude'], color='red', label='Detected Anomaly', s=50, zorder=5)

plt.title('Simulated Appliance Vibration Data with Anomaly Detection')
plt.xlabel('Time')
plt.ylabel('Vibration Amplitude')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# --- 7. Save the Plot ---
# In Colab, this will save the image to the Colab environment's file system.
plt.savefig('simulated_anomaly_detection_plot.png')
print("Plot saved as 'simulated_anomaly_detection_plot.png'")
# plt.show() # In Colab, plots are shown automatically below the cell

# --- Optional: Visualize Anomaly Scores Distribution ---
plt.figure(figsize=(10, 6))
plt.hist(df_normal['anomaly_score'], bins=50, density=True, alpha=0.7, color='green')
plt.axvline(x=model.threshold_, color='red', linestyle='--', label='Anomaly Threshold')
plt.title('Distribution of Anomaly Scores')
plt.xlabel('Anomaly Score (Lower = More Anomalous)')
plt.ylabel('Density')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('anomaly_score_distribution.png')
print("Plot saved as 'anomaly_score_distribution.png'")
# plt.show() # In Colab, plots are shown automatically below the cell