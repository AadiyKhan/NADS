Real-Time Network Anomaly Detection System (NIDS) with MLOps
============================================================

This project implements a complete, end-to-end Machine Learning Operations (MLOps) pipeline for **real-time network anomaly detection**. It utilizes an **Autoencoder Neural Network** for detecting zero-day attacks and leverages **Apache Kafka** for high-throughput data ingestion, all containerized with **Docker Compose**.

🎯 Project Goal
---------------

The primary goal was to build a low-latency system that can consume a stream of network flow data, identify anomalous (attack) traffic in real-time, and alert the user, simulating a production Security Operations Center (SOC) environment.

🛠️ Technology Stack
--------------------

CategoryTechnologyPurpose**Data StreamingApache Kafka (Docker)**High-throughput, fault-tolerant message broker for real-time flow ingestion.**CoordinationApache Zookeeper (Docker)**Manages the configuration and state of the Kafka cluster.**ModelTensorFlow / Keras**Built the Autoencoder Neural Network.**Data SciencePython, Pandas, Scikit-learn**Data cleaning, preprocessing, and model scaling.**DeploymentPython (Prediction Service)**Consumes from Kafka, loads the model, and performs real-time inference.**ContainerizationDocker Compose**Orchestrates and manages the Kafka and Zookeeper services.Export to Sheets

💻 Architecture Overview
------------------------

The system is designed as a pipeline with decoupled microservices, enabling scalability and resilience.

1.  **Data Source:** Cleaned **CIC-IDS 2017** dataset.
    
2.  **Producer (2\_producer\_service.py):** Reads the static CSV file and simulates a live network, sending records to Kafka one-by-one.
    
3.  **Kafka Broker:** Holds the stream of network flows in the network\_flows topic.
    
4.  **Prediction Service (4\_prediction\_service.py):** Acts as the **Consumer**. It continuously pulls data from Kafka, performs preprocessing, runs the Autoencoder model for inference, and issues alerts.
    

📂 Project Phases & Key Outcomes
--------------------------------

### Phase 1: Data Preprocessing

*   Merged 10+ raw CSV files into a single, clean dataset, handling **2.5 million records**.
    
*   Cleaned data by removing **NaNs, Inf values**, and dropping duplicates.
    
*   Transformed the problem into a binary classification task (0=Normal, 1=Anomaly).
    

### Phase 2: Data Engineering Pipeline Setup

*   Deployed a robust Kafka/Zookeeper environment using docker-compose.yml.
    
*   Developed the **Producer Service** to stream the dataset into the Kafka topic network\_flows.
    

### Phase 3: Model Development

*   **Feature Engineering:** Applied **StandardScaling** and **One-Hot Encoding** to prepare 78 features for the neural network.
    
*   **Model:** Developed and trained an **Autoencoder Neural Network** (input size 78, bottleneck size 8) exclusively on normal traffic.
    
*   **Output:** Saved the essential artifacts for deployment:
    
    *   anomaly\_autoencoder.h5 (Trained Model)
        
    *   scaler.pkl (Fitted Scaler for live preprocessing)
        
    *   anomaly\_threshold.txt (Decision boundary based on the 95th percentile of normal reconstruction error).
        

### Phase 4: Real-Time Deployment

*   Created the **Prediction Service** which loads the model artifacts at runtime.
    
*   The service consumes records from Kafka, applies the saved scaler.pkl for **live transformation**, calculates the **Reconstruction Error (MSE)**, and flags the flow as an **🚨 ANOMALY ALERT!** if the MSE exceeds the learned threshold.
    

🚀 Setup and Run Instructions
-----------------------------

To replicate this project, follow these steps:

### 1\. Prerequisites

*   **Python 3.8+**
    
*   **Docker Desktop** (Required for Kafka/Zookeeper)
    

### 2\. Setup

1.  Bashgit clone \[YOUR-REPO-LINK\]cd \[YOUR-REPO-NAME\]
    
2.  Bashpython -m venv .venv# Windows PowerShell:.\\.venv\\Scripts\\Activate.ps1# Linux/macOS:source .venv/bin/activate
    
3.  Bashpip install -r requirements.txt_(Note: Create this file using pip freeze > requirements.txt after installing all project libraries: pandas, numpy, scikit-learn, tensorflow, kafka-python)_
    

### 3\. Start Infrastructure (Kafka/Zookeeper)

Start the streaming platform in detached mode:

Bash

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   docker compose up -d   `

Verify the services are running by visiting **http://localhost:8080** (Kafka-UI). You must manually create the topic named network\_flows in the UI.

### 4\. Run the Pipeline

Open **two separate terminal windows** (both with the virtual environment activated).

**Terminal 1: Start Prediction Service (The Anomaly Detector)**

Bash

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   python 4_prediction_service.py   `

**Terminal 2: Start Data Stream (The Simulated Network)**

Bash

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   python 2_producer_service.py   `

You will see the Prediction Service start printing the Mean Squared Error (MSE) for each flow. When an attack record is streamed, the MSE will spike, triggering the **🚨 ANOMALY ALERT!** message, confirming successful detection.

🤝 Contribution
---------------

This project is maintained by \[Your Name\] / \[Your GitHub Username\]. Feel free to submit issues or pull requests!