#!/usr/bin/env python3

## Project: Edge and Cloud Computing with ML-Based IoT
## This script demonstrates an IoT application that logs resource usage,
## integrates with a SQL Server database, uses AES encryption,
## trains an optimized decision tree model, simulates sensor data,
## and visualizes trends.

import os
import random
import json
import time
import base64
import joblib
import pyodbc
import numpy as np
import csv
import psutil
import pandas as pd
import matplotlib.pyplot as plt

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

## =======================
## Configuration Section
## =======================
DATABASE_CONNECTION = os.getenv("DATABASE_CONNECTION",
    "Driver={ODBC Driver 17 for SQL Server};"
    "Server=<YOUR_SERVER>;"
    "PORT=1433;"
    "Database=<YOUR_DATABASE>;"
    "Uid=<YOUR_USERNAME>;"
    "Pwd=<YOUR_PASSWORD>;"
    "Encrypt=yes;"
    "TrustServerCertificate=no;"
    "Connection Timeout=30;")

## =======================
## Logging and Utility Functions
## =======================
def log_resource_usage():
    ## Logs CPU and memory usage.
    cpu_usage = psutil.cpu_percent(interval=1)
    memory_usage = psutil.virtual_memory().percent
    print(f"CPU Usage: {cpu_usage}% | Memory Usage: {memory_usage}%")

def log_security_event(event_type, details):
    ## Logs security events to a CSV file with a timestamp.
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open("anomaly_logs.csv", "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([timestamp, event_type, details])
    print(f"Logged security event: {event_type}")

## =======================
## Database Integration Functions
## =======================
def test_db_connection():
    ## Tests the database connection.
    try:
        conn = pyodbc.connect(DATABASE_CONNECTION)
        cursor = conn.cursor()
        cursor.execute("SELECT @@VERSION")
        row = cursor.fetchone()
        print("Connected successfully! SQL Server Version:", row[0])
        cursor.close()
        conn.close()
    except Exception as e:
        print("Failed to connect to the database:", e)

def save_to_database(data, encrypted_data):
    ## Saves sensor data and its encrypted version to the database.
    try:
        conn = pyodbc.connect(DATABASE_CONNECTION)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO IoTData (Timestamp, Temperature, Humidity, EncryptedData) VALUES (CURRENT_TIMESTAMP, ?, ?, ?)",
            (data[0], data[1], encrypted_data.hex())
        )
        conn.commit()
        cursor.close()
        conn.close()
        print("Data saved securely.")
    except Exception as e:
        print(f"Failed to save data: {e}")

## =======================
## Secure AES Encryption Setup
## =======================
def get_secure_key():
    ## Generates a new AES key.
    new_key = os.urandom(16)
    print("Warning: Generating a new key. Store it securely!")
    return new_key

key = get_secure_key()
iv = os.urandom(16)

def encrypt_data(data):
    ## Encrypts a string using AES.
    padder = padding.PKCS7(128).padder()
    padded_data = padder.update(data.encode()) + padder.finalize()
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    return encryptor.update(padded_data) + encryptor.finalize()

def decrypt_data(ciphertext):
    ## Decrypts AES-encrypted ciphertext.
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    decrypted_padded_data = decryptor.update(ciphertext) + decryptor.finalize()
    unpadder = padding.PKCS7(128).unpadder()
    return unpadder.update(decrypted_padded_data) + unpadder.finalize()

## =======================
## Optimized Decision Tree Model (ML)
## =======================
class OptimizedDecisionTree:
    ## A decision tree classifier with grid search optimization.
    def __init__(self):
        self.model = None

    def train(self, X, y):
        ## Trains the model using GridSearchCV.
        params = {'max_depth': [2, 4, 6, 8, 10], 'min_samples_leaf': [2, 4, 6, 8]}
        grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), params, cv=5)
        grid_search.fit(X, y)
        self.model = grid_search.best_estimator_

    def predict(self, X):
        ## Returns predictions for the input.
        return self.model.predict(X)

## =======================
## Data Simulation and Processing Functions
## =======================
def generate_simulated_dataset():
    ## Generates simulated sensor data and labels.
    X = np.random.uniform(20, 50, (500, 2))
    y = [0 if (17 <= temp <= 27) and (40 <= hum <= 70) else 1 for temp, hum in X]
    return X, y

def get_sensor_data():
    ## Generates random sensor data.
    temperature = round(random.uniform(20, 50), 2)
    humidity = round(random.uniform(30, 60), 2)
    return [temperature, humidity]

def add_differential_privacy(data, epsilon=0.5):
    ## Adds differential privacy noise to data.
    noise = np.random.laplace(0, 1/epsilon, len(data))
    return [round(d + n, 2) for d, n in zip(data, noise)]

## =======================
## Visualization Functions
## =======================
def visualize_current_transmissions(current_records):
    ## Visualizes sensor data trends.
    if not current_records:
        print("No data to visualize.")
        return

    df_current = pd.DataFrame(current_records)
    df_current["Timestamp"] = pd.to_datetime(df_current["Timestamp"])
    df_current.sort_values("Timestamp", inplace=True)
    start_time = df_current["Timestamp"].min()
    df_current["Elapsed_Seconds"] = (df_current["Timestamp"] - start_time).dt.total_seconds()

    def is_threshold_anomaly(temp, hum):
        return not (17 <= temp <= 27 and 40 <= hum <= 70)

    df_current["ThresholdAnomaly"] = df_current.apply(
        lambda row: 1 if is_threshold_anomaly(row["Temperature"], row["Humidity"]) else 0,
        axis=1
    )

    ## Temperature Plot
    plt.figure(figsize=(12, 5))
    normal = df_current[df_current["ThresholdAnomaly"] == 0]
    anomaly = df_current[df_current["ThresholdAnomaly"] == 1]
    if not normal.empty:
        plt.plot(normal["Elapsed_Seconds"], normal["Temperature"], label="Normal Temperature", linewidth=2)
        plt.scatter(normal["Elapsed_Seconds"], normal["Temperature"])
    if not anomaly.empty:
        plt.plot(anomaly["Elapsed_Seconds"], anomaly["Temperature"], label="Anomaly Temperature", linewidth=2)
        plt.scatter(anomaly["Elapsed_Seconds"], anomaly["Temperature"], marker='x', s=100)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Temperature (°C)")
    plt.title("Temperature Trends")
    plt.legend()
    plt.grid(True)
    plt.show()

    ## Humidity Plot
    plt.figure(figsize=(12, 5))
    if not normal.empty:
        plt.plot(normal["Elapsed_Seconds"], normal["Humidity"], label="Normal Humidity", linewidth=2)
        plt.scatter(normal["Elapsed_Seconds"], normal["Humidity"])
    if not anomaly.empty:
        plt.plot(anomaly["Elapsed_Seconds"], anomaly["Humidity"], label="Anomaly Humidity", linewidth=2)
        plt.scatter(anomaly["Elapsed_Seconds"], anomaly["Humidity"], marker='x', s=100)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Humidity (%)")
    plt.title("Humidity Trends")
    plt.legend()
    plt.grid(True)
    plt.show()

## =======================
## Main Execution Loop
## =======================
def main_loop():
    ## Main loop for simulating sensor data and visualization.
    X, y = generate_simulated_dataset()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

    dtree = OptimizedDecisionTree()
    dtree.train(X_train, y_train)
    joblib.dump(dtree.model, "optimized_decision_tree.pkl")

    current_records = []
    start_time = time.time()
    run_duration = 120  ## 2 minutes
    while time.time() - start_time < run_duration:
        log_resource_usage()
        data = get_sensor_data()
        current_timestamp = pd.Timestamp.now()
        print(f"Sensor Data: {data} at {current_timestamp}")

        data_priv = add_differential_privacy(data, epsilon=0.5)
        data_transformed = pca.transform(scaler.transform([data_priv]))
        prediction = dtree.predict(data_transformed)
        print("Prediction:", "Anomaly" if prediction[0] == 1 else "Normal")

        if prediction[0] == 1:
            print("Anomaly detected!")
            log_security_event("Anomaly Detected", f"Data: {data_priv}")
        else:
            encrypted_data = encrypt_data(json.dumps({"temperature": data_priv[0], "humidity": data_priv[1]}))
            save_to_database(data_priv, encrypted_data)

        current_records.append({
            "Timestamp": current_timestamp,
            "Temperature": data_priv[0],
            "Humidity": data_priv[1],
            "Prediction": prediction[0]
        })
        time.sleep(2)

    visualize_current_transmissions(current_records)

## =======================
## Entry Point
## =======================
if __name__ == "__main__":
    test_db_connection()
    main_loop()
