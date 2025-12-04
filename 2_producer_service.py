import pandas as pd
import json
import time
from kafka import KafkaProducer
from pathlib import Path

# --- Configuration ---
KAFKA_BROKER = 'localhost:9092'
KAFKA_TOPIC = 'network_flows'
# Name of the cleaned file from Phase 1
DATA_FILENAME = 'cleaned_merged_cicids2017.csv' 
# Simulated delay between sending records (e.g., 0.01s = 100 messages/sec)
DELAY_SECONDS = 0.005 

print(f"Starting Kafka Producer for topic: {KAFKA_TOPIC}")

# Initialize the Kafka Producer
producer = KafkaProducer(
    bootstrap_servers=[KAFKA_BROKER],
    value_serializer=lambda x: json.dumps(x).encode('utf-8')
)

# Load the cleaned dataset
try:
    df = pd.read_csv(Path(DATA_FILENAME))
    # Drop the 'Label' column here as it's the target and should not be sent
    # The ML model should predict the label.
    df_data = df.drop(columns=['Label']) 
    print(f"Loaded {len(df_data)} records for streaming.")
except FileNotFoundError:
    print(f"Error: {DATA_FILENAME} not found. Ensure Phase 1 ran correctly.")
    exit()

try:
    for index, row in df_data.iterrows():
        # Convert the Pandas Series (row) to a Python dictionary
        record = row.to_dict()

        # Send the record to Kafka
        producer.send(KAFKA_TOPIC, value=record)

        if index % 1000 == 0:
            print(f"Sent {index} records to Kafka...")

        time.sleep(DELAY_SECONDS)

    producer.flush() # Ensure all messages are delivered
    print("--- All records sent successfully! ---")

except Exception as e:
    print(f"An error occurred: {e}")
finally:
    producer.close()