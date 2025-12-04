import numpy as np
import json
import pickle
from kafka import KafkaConsumer
from tensorflow.keras.models import load_model
from pathlib import Path
# Add this line at the top with your other imports
from tensorflow.keras.metrics import MeanSquaredError

# --- CONFIGURATION ---
KAFKA_BROKER = 'localhost:9092'
KAFKA_TOPIC = 'network_flows'
MODEL_FILENAME = 'anomaly_autoencoder.h5'
SCALER_FILENAME = 'scaler.pkl' 
THRESHOLD_FILENAME = 'anomaly_threshold.txt'
INPUT_DIM = 78  # Must match the dimension used in training (X_train_normal.shape[1])

print("--- Starting Real-Time Anomaly Prediction Service ---")

# 1. LOAD MODEL ASSETS (Only once at startup)
try:
    # Load the trained model
    autoencoder = load_model(
        Path(MODEL_FILENAME),
        # ADD THIS LINE to fix the deserialization error:
        custom_objects={'mse': MeanSquaredError()}
    )
    print(f"Loaded Autoencoder model from {MODEL_FILENAME}.")
    # Load the fitted scaler
    with open(SCALER_FILENAME, 'rb') as file:
        scaler = pickle.load(file)
    print(f"Loaded StandardScaler from {SCALER_FILENAME}.")

    # Load the saved threshold
    with open(THRESHOLD_FILENAME, 'r') as f:
        THRESHOLD = float(f.read())
    print(f"Loaded Anomaly Threshold: {THRESHOLD:.6f}")

except Exception as e:
    print(f"ERROR: Failed to load critical assets. Ensure files exist. Details: {e}")
    exit()


# 2. INITIALIZE KAFKA CONSUMER
consumer = KafkaConsumer(
    KAFKA_TOPIC,
    bootstrap_servers=[KAFKA_BROKER],
    # The value must be deserialized from JSON bytes to a Python dictionary
    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
    # Start reading from the latest record (new flows only)
    # Change to 'earliest' to re-process all existing records
    auto_offset_reset='latest', 
    group_id='anomaly-detector-group'
)


def process_flow(flow_data: dict, scaler, model, threshold):
    """Preprocesses data, predicts MSE, and checks for anomaly."""
    
    # 2.1 CONVERT TO NUMPY ARRAY
    # The model expects a 2D array: [[feature_1, feature_2, ..., feature_78]]
    # We must handle the missing categorical features if they were one-hot encoded but not in the flow.
    # For simplicity, we assume the flow_data dict contains the 78 scaled features.
    
    # --- IMPORTANT NOTE ON SCALING/ENCODING ---
    # In a full production system, you would need a dictionary defining all 78 feature names
    # and map the incoming flow_data to those names, using the scaler on the correct columns.
    # For this project, we rely on the input order being correct.
    
    # Extract values and convert to numpy array
    try:
        # Convert dictionary values to an array, ensuring correct shape and type
        X_raw = np.array(list(flow_data.values()), dtype=np.float64).reshape(1, -1)
        
        # 2.2 SCALE THE DATA
        # Apply the fitted Standard Scaler to the incoming flow
        X_scaled = scaler.transform(X_raw)

    except ValueError as e:
        # This error often occurs if the number of features is wrong (not 78)
        # or if there are non-numeric values.
        print(f"WARNING: Feature mismatch or bad data detected: {e}")
        return

    # 2.3 PREDICT RECONSTRUCTION ERROR
    reconstruction = model.predict(X_scaled, verbose=0)
    mse = np.mean(np.power(X_scaled - reconstruction, 2), axis=1)[0]
    
    # 2.4 ANOMALY ALERT
    if mse >= threshold:
        print(f"\n🚨 ANOMALY ALERT! MSE: {mse:.6f} (Threshold: {threshold:.6f})")
        # In a real system, you would push this alert to Slack/Database/SIEM tool
    else:
        # Print periodically to show the service is running
        if consumer.total_processed % 1000 == 0:
            print(f"Checked {consumer.total_processed} flows. Last MSE: {mse:.6f}")

    # For simple tracking:
    if not hasattr(consumer, 'total_processed'):
        consumer.total_processed = 1
    else:
        consumer.total_processed += 1
        


# 3. MAIN CONSUMPTION LOOP
print("Service is listening for network flows...")
try:
    for message in consumer:
        # message.value is the Python dictionary of the flow
        process_flow(message.value, scaler, autoencoder, THRESHOLD)
        
except KeyboardInterrupt:
    print("\nAnomaly Prediction Service stopped by user.")
except Exception as e:
    print(f"A runtime error occurred: {e}")
finally:
    consumer.close()