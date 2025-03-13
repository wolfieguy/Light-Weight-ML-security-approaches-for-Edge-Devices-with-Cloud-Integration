# Edge and Cloud Computing with ML-Based IoT

This project demonstrates a comprehensive IoT application that leverages both edge and cloud computing capabilities. It logs system resource usage, integrates with a SQL Server database, employs AES encryption to secure data transmissions, trains an optimized decision tree model for anomaly detection, simulates sensor data with differential privacy, and visualizes trends in sensor data.

## Table of Contents
- [Features](#features)
- [Architecture Overview](#architecture-overview)
- [Prerequisites](#prerequisites)
- [Setup and Installation](#setup-and-installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Testing](#testing)
- [Continuous Integration](#continuous-integration)
- [Contributing](#contributing)
- [License](#license)

## Features
- Database Integration:  
  Connects to a SQL Server using pyodbc to store sensor data and log events.
- Secure Encryption:  
  Utilizes AES encryption for securing data before database storage.
- Machine Learning:  
  Implements an optimized decision tree model with hyperparameter tuning (via GridSearchCV) for anomaly detection.
- Data Simulation:  
  Generates synthetic sensor data and applies differential privacy techniques using Laplace noise.
- Visualization:  
  Uses matplotlib to visualize trends in temperature and humidity, distinguishing normal readings from anomalies.
- Continuous Integration:  
  Integrates with GitHub Actions to automatically run tests on every push.

## Architecture Overview
The system is designed to run at the edge (e.g., on IoT devices) and communicate with cloud resources:
- Edge Processing:  
  Data is simulated, preprocessed, and analyzed locally. Resource usage is logged to monitor performance.
- Cloud Integration:  
  Secure data is sent to a SQL Server database in the cloud for storage and further analysis.
- ML-Based Anomaly Detection:  
  A decision tree model is trained on simulated data and used in real-time to flag anomalies.
- Visualization:  
  Graphs display sensor trends to assist with quick diagnosis and monitoring.

## Prerequisites
- Python 3.8+  
- Required Libraries:  
  Listed in [requirements.txt](requirements.txt). Some key libraries include:
  - pyodbc
  - joblib
  - psutil
  - pandas
  - matplotlib
  - cryptography
  - scikit-learn
  - numpy
  - pytest
- SQL Server:  
  Ensure you have access to a SQL Server instance for database integration.

## Setup and Installation

1. Clone the Repository:
   ```bash
   git clone https://github.com/<your-username>/<repository-name>.git
   cd IoTSecurityGithub
