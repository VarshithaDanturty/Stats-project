# Importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, expon
import streamlit as st

# Step 1: Generate Random Data
def generate_data():
    np.random.seed(42)
    normal_traffic = np.random.normal(loc=100, scale=20, size=1000)  # Normal network traffic
    anomaly_traffic = np.random.uniform(low=200, high=300, size=50)  # Simulated anomalies
    traffic_data = np.concatenate([normal_traffic, anomaly_traffic])
    np.random.shuffle(traffic_data)
    return pd.DataFrame({'Traffic': traffic_data})

# Step 2: Preprocess Data
def preprocess_data(data):
    # Removing outliers based on IQR
    Q1 = data['Traffic'].quantile(0.25)
    Q3 = data['Traffic'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data['Anomaly'] = (data['Traffic'] < lower_bound) | (data['Traffic'] > upper_bound)
    return data, lower_bound, upper_bound

# Step 3: Visualization
def visualize_data(data, lower_bound, upper_bound):
    st.write("### Traffic Data Distribution")
    sns.histplot(data['Traffic'], kde=True, color='blue', label='Traffic')
    plt.axvline(lower_bound, color='red', linestyle='--', label='Lower Threshold')
    plt.axvline(upper_bound, color='green', linestyle='--', label='Upper Threshold')
    plt.legend()
    st.pyplot(plt.gcf())
    plt.clf()

# Step 4: Fit Probability Models
def fit_probability_model(data):
    mean, std = norm.fit(data['Traffic'][~data['Anomaly']])  # Exclude anomalies
    return mean, std

# Step 5: Anomaly Detection with Threshold Setting
def detect_anomalies(data, mean, std):
    z_scores = (data['Traffic'] - mean) / std
    data['Z-Score'] = z_scores
    data['Probability'] = norm.cdf(z_scores)  # Probability from cumulative distribution function
    data['Anomaly'] = (data['Traffic'] > mean + 3 * std) | (data['Traffic'] < mean - 3 * std)
    return data

# Streamlit App
def main():
    st.title("Network Traffic Anomaly Detection")
    
    # Generate Data
    data = generate_data()
    st.write("### Raw Traffic Data")
    st.write(data)
    
    # Preprocess Data
    data, lower_bound, upper_bound = preprocess_data(data)
    st.write("### Preprocessed Data")
    st.write(data)
    
    # Visualize Data
    visualize_data(data, lower_bound, upper_bound)
    
    # Fit Probability Model
    mean, std = fit_probability_model(data)
    st.write("### Probability Model Parameters")
    st.write(f"Mean: {mean:.2f}, Std Dev: {std:.2f}")
    
    # Detect Anomalies
    data = detect_anomalies(data, mean, std)
    st.write("### Anomalies Detected")
    st.write(data[data['Anomaly']])
    
    # Final Visualization
    st.write("### Anomaly Visualization")
    sns.scatterplot(x=range(len(data)), y=data['Traffic'], hue=data['Anomaly'], palette={True: 'red', False: 'blue'})
    st.pyplot(plt.gcf())
    plt.clf()

if __name__ == "__main__":
    main()
