# AI-Powered Predictive Maintenance System for Appliances

## Project Overview

This project focuses on developing an AI-powered system for predictive maintenance of domestic or industrial appliances. The primary goal is to shift from reactive (fix-when-broken) to proactive (predict-and-prevent) maintenance strategies, thereby minimizing downtime, reducing repair costs, and extending appliance lifespan.

The system leverages anomaly detection techniques on sensor data to identify early signs of potential equipment failure.

## Problem Statement

Traditional maintenance approaches for appliances often lead to unexpected breakdowns, resulting in:
* Unscheduled downtime and operational inconvenience.
* Increased repair costs due to emergency fixes.
* Reduced appliance longevity due to undetected wear and tear.
* Potential safety hazards from malfunctioning equipment.

This project addresses these challenges by enabling early detection of anomalies, facilitating timely interventions.

## Proposed Solution

Our system proposes an edge-AI driven approach:
1.  **Data Acquisition:** Real-time sensor data (e.g., vibration, temperature) is collected from appliances using an embedded module (e.g., ESP32 with relevant sensors).
2.  **On-Device Preprocessing:** Raw data is filtered and key features are extracted on the embedded device.
3.  **Edge AI Anomaly Detection:** A lightweight, unsupervised machine learning model (Isolation Forest) is deployed directly on the embedded device to analyze data in real-time.
4.  **Alerting:** Anomalies detected trigger immediate alerts, enabling proactive maintenance.

## Key Components & Technologies Used

* **Python:** Main programming language for data simulation, model training, and analysis.
* **NumPy:** For numerical operations and array manipulation.
* **Pandas:** For data handling and time-series management.
* **Scikit-learn:** Used for implementing the **Isolation Forest** anomaly detection algorithm.
* **Matplotlib:** For data visualization and plotting results.
* **Concept of Edge AI / TinyML:** Deployment strategy for running ML models directly on embedded devices (e.g., ESP32).

## How to Run the Simulation Code

The `generate_results.py` script simulates appliance sensor data, applies the Isolation Forest algorithm to detect anomalies, and generates two key visualization plots.

**Prerequisites:**
Make sure you have Python installed. You can install the necessary libraries using pip:
```bash
pip install numpy pandas scikit-learn matplotlib
