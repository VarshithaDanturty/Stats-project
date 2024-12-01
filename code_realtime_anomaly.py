import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import streamlit as st

# Streamlit App Configuration
st.set_page_config(page_title="Network Traffic Anomaly Detection", layout="wide")

# Function to simulate more stable real-time traffic data
def generate_real_time_data():
    while True:
        # Generate traffic within a controlled range to reduce variation
        traffic = np.random.normal(loc=100, scale=10) if np.random.rand() > 0.1 else np.random.uniform(150, 180)
        packet_size = np.random.normal(loc=512, scale=20)
        inter_arrival_time = np.random.exponential(scale=0.8)  # More stable exponential
        yield {"Traffic": traffic, "Packet_Size": packet_size, "Inter_Arrival_Time": inter_arrival_time}

# Function to preprocess data and calculate thresholds (IQR method)
def preprocess_data(data):
    Q1 = data['Traffic'].quantile(0.25)
    Q3 = data['Traffic'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data['Anomaly'] = (data['Traffic'] < lower_bound) | (data['Traffic'] > upper_bound)
    return data, lower_bound, upper_bound

# Function to fit a normal distribution to the traffic data
def fit_probability_model(data):
    mean, std = norm.fit(data['Traffic'][~data['Anomaly']])  # Exclude anomalies
    return mean, std

# Function to detect anomalies using a probability model (Z-score)
def detect_anomalies(data, mean, std):
    z_scores = (data['Traffic'] - mean) / std
    data['Z-Score'] = z_scores
    data['Probability'] = norm.cdf(z_scores)
    data['Anomaly'] = (data['Traffic'] > mean + 3 * std) | (data['Traffic'] < mean - 3 * std)
    return data

# Aggregating data to smooth out traffic fluctuations (using last N data points)
def aggregate_data(data, interval=5):
    return data.tail(interval).mean()

# Visualization function to show data and anomaly detection
def visualize_data(data, lower_bound, upper_bound, placeholder):
    with placeholder.container():
        st.write("### Traffic Distribution with Anomaly Thresholds")
        sns.histplot(data['Traffic'], kde=True, color='blue', label='Traffic')
        plt.axvline(lower_bound, color='red', linestyle='--', label='Lower Threshold')
        plt.axvline(upper_bound, color='green', linestyle='--', label='Upper Threshold')
        plt.legend()
        st.pyplot(plt.gcf())
        plt.clf()

        st.write("### Traffic Over Time")
        sns.scatterplot(x=data.index, y=data['Traffic'], hue=data['Anomaly'], palette={True: 'red', False: 'blue'})
        plt.title("Traffic Patterns with Anomalies")
        st.pyplot(plt.gcf())
        plt.clf()

# Main Streamlit App
def main():
    st.title("Network Traffic Anomaly Detection")
    st.sidebar.write("### Settings")
    simulation_speed = st.sidebar.slider("Simulation Speed (ms)", min_value=100, max_value=2000, value=500)

    # Initialize data and placeholders
    traffic_data = pd.DataFrame(columns=["Traffic", "Packet_Size", "Inter_Arrival_Time", "Anomaly"])
    data_placeholder = st.empty()

    # Real-time data generation and processing loop
    data_gen = generate_real_time_data()
    while True:
        # Generate a new traffic data point
        new_data = next(data_gen)
        traffic_data = pd.concat([traffic_data, pd.DataFrame([new_data])], ignore_index=True)

        # Aggregate data to smooth fluctuations (optional)
        aggregated_data = aggregate_data(traffic_data)

        # Preprocess data and calculate thresholds
        processed_data, lower_bound, upper_bound = preprocess_data(traffic_data)

        # Fit probability model and detect anomalies
        mean, std = fit_probability_model(processed_data)
        processed_data = detect_anomalies(processed_data, mean, std)

        # Visualize updated data
        visualize_data(processed_data, lower_bound, upper_bound, data_placeholder)

        # Pause for simulation speed to adjust update rate
        st.sidebar.write(f"Data Points: {len(processed_data)}")
        st.sidebar.write(f"Current Thresholds: [{lower_bound:.2f}, {upper_bound:.2f}]")
        st.sidebar.write(f"Model Mean: {mean:.2f}, Std Dev: {std:.2f}")
        plt.pause(simulation_speed / 1000)

if __name__ == "__main__":
    main()
