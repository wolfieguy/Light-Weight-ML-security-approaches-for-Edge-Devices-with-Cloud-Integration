import pytest
import numpy as np
from app import (
    get_sensor_data,
    add_differential_privacy,
    get_secure_key,
    encrypt_data,
    decrypt_data,
    generate_simulated_dataset,
    OptimizedDecisionTree,
    log_resource_usage,
)

def test_get_sensor_data():
    data = get_sensor_data()
    assert isinstance(data, list)
    assert len(data) == 2
    assert all(isinstance(x, float) for x in data)

def test_add_differential_privacy():
    data = [25.0, 45.0]
    data_priv = add_differential_privacy(data, epsilon=0.5)
    assert len(data_priv) == len(data)

def test_get_secure_key():
    key = get_secure_key()
    assert isinstance(key, bytes)
    assert len(key) == 16

def test_encrypt_decrypt():
    original_message = "This is a test message."
    encrypted = encrypt_data(original_message)
    decrypted = decrypt_data(encrypted)
    if isinstance(decrypted, bytes):
        decrypted = decrypted.decode('utf-8')
    assert decrypted == original_message

def test_generate_simulated_dataset():
    X, y = generate_simulated_dataset()
    assert isinstance(X, np.ndarray)
    assert X.shape == (500, 2)
    assert isinstance(y, list)
    assert len(y) == 500
    for label in y:
        assert label in [0, 1]

def test_optimized_decision_tree_prediction():
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    X, y = generate_simulated_dataset()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
    
    model = OptimizedDecisionTree()
    model.train(X_train, y_train)
    predictions = model.predict(X_test)
    for pred in predictions:
        assert pred in [0, 1]

def test_log_resource_usage_output(capsys):
    log_resource_usage()
    captured = capsys.readouterr().out
    assert "CPU Usage:" in captured and "Memory Usage:" in captured
