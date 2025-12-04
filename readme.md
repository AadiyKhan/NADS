# Real-Time Network Anomaly Detection System (NIDS) with MLOps

A complete, end-to-end **Machine Learning Operations (MLOps)** pipeline for real-time network anomaly detection using an **Autoencoder Neural Network** and **Apache Kafka**. The system simulates a production-level **Security Operations Center (SOC)** with real-time streaming, inference, and alerting.

* * *

##  Project Goal

Build a **low-latency anomaly detection system** capable of consuming live network flow data, detecting zero-day attacks using an Autoencoder, and raising alerts in real time.

* * *

##  Technology Stack

| Category | Technology | Purpose |
| --- | --- | --- |
| Data Streaming | Apache Kafka (Docker) | Real-time, fault-tolerant message streaming |
| Coordination | Apache Zookeeper (Docker) | Kafka cluster coordination |
| Model Development | TensorFlow / Keras | Autoencoder Neural Network |
| Data Science | Python, Pandas, Scikit-learn | Preprocessing & feature engineering |
| Deployment | Python Prediction Service | Real-time Kafka consumer + inference |
| Containerization | Docker Compose | Orchestrates all services |

### Components Summary

*   **Data Source:** Cleaned CIC-IDS 2017 dataset
    
*   **Producer (`2_producer_service.py`):** Streams rows from CSV → Kafka topic `network_flows`
    
*   **Kafka Broker:** Stores & streams messages
    
*   **Prediction Service (`4_prediction_service.py`):**
    
    *   Consumes Kafka stream
        
    *   Preprocesses using `scaler.pkl`
        
    *   Runs Autoencoder model
        
    *   Flags anomalies using reconstruction error threshold
        

* * *

##  Project Phases & Outputs

### Phase 1: Data Preprocessing

*   Combined 10+ raw CSV files (**~2.5M rows**).
    
*   Removed NaN, Inf, and duplicates.
    
*   Converted the multi-class problem to **binary classification** (0 = Normal, 1 = Anomaly).
    

### Phase 2: Data Engineering

*   Built Kafka + Zookeeper environment using Docker Compose.
    
*   Implemented Producer to stream records into Kafka topic `network_flows`.
    

### Phase 3: Model Development

*   Performed Standard Scaling + One-Hot Encoding (78 features).
    
*   Designed Autoencoder with **bottleneck size 8**.
    
*   Trained **only on normal traffic** to learn reconstruction patterns.
    
*   Exported artifacts:
    
    *   `anomaly_autoencoder.h5` (Trained Model)
        
    *   `scaler.pkl` (Fitted Scaler for live preprocessing)
        
    *   `anomaly_threshold.txt` (Decision boundary set at the 95th percentile of normal reconstruction error).
        

### Phase 4: Real-Time Deployment

*   Prediction Service loads model & scaler at startup.
    
*   Computes reconstruction error (MSE) in real time for every incoming flow.
    
*   Raises **🚨 ANOMALY ALERT!** for high MSE values, demonstrating live detection.
    

* * *

##  Setup & Running the System

### 1\. Prerequisites

*   Python 3.8+
    
*   Docker Desktop
    

### 2\. Cloning and Setup

1.  **Clone the Repository:**
    
    Bash
    
        git clone https://github.com/AadiyKhan/NADS.git
        cd https://github.com/AadiyKhan/NADS.git
    
2.  **Create a Virtual Environment:**
    
    Bash
    
        python -m venv .venv
    
    Activate it:
    
    Bash
    
        # Windows:
        .\.venv\Scripts\Activate.ps1
        # Linux/macOS:
        source .venv/bin/activate
    
3.  **Install dependencies:**
    
    Bash
    
        pip install -r requirements.txt
    

* * *

### 3\. Start Kafka Infrastructure

1.  Start the streaming platform:
    
    Bash
    
        docker compose up -d
    
2.  Visit Kafka UI: `http://localhost:8080`
    
3.  **Create topic:** Manually create the topic named `network_flows`.
    

* * *

### 4\. Run the Full Pipeline

Open **two terminals** (with the virtual environment activated).

**Terminal 1 — Prediction Service (The Detector):**

Bash

    python 4_prediction_service.py

**Terminal 2 — Data Producer (The Simulated Network):**

Bash

    python 2_producer_service.py

You will see continuous MSE values and **🚨 ANOMALY ALERT!** messages when attacks are detected in the stream.

* * *

##  Contribution

Maintained by Aadiy Khan. Pull requests and issues are welcome.
