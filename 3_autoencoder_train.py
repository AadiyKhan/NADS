import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path

# --- CONFIGURATION ---
DATA_FILENAME = 'cleaned_merged_cicids2017.csv'
MODEL_FILENAME = 'anomaly_autoencoder.h5'
SCALER_FILENAME = 'scaler.pkl' 
THRESHOLD_FILENAME = 'anomaly_threshold.txt'
TARGET_COLUMN = 'Label'

LATENT_DIM = 8     # Bottleneck size
EPOCHS = 15        # Number of training iterations
BATCH_SIZE = 512   # Batch size for training the NN
EVAL_BATCH_SIZE = 65536 # Larger batch size for memory-safe evaluation (64K records)
THRESHOLD_PERCENTILE = 0.95 # Sets the sensitivity (95% of normal traffic will pass)

print("--- Starting Autoencoder Model Preparation and Training ---")

## 1. DATA LOADING AND PREPARATION
print("Loading and preparing data...")

# Load the full data
df = pd.read_csv(Path(DATA_FILENAME))
y_full = df[TARGET_COLUMN].values

# Drop Identifier and Unnecessary Columns
drop_cols = ['Flow_ID', 'Source_IP', 'Destination_IP', 'Timestamp', 'External_IP']
df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

# Handle Categorical Features (One-Hot Encode)
categorical_cols = ['Protocol', 'Fwd_Flags', 'Bwd_Flags'] 
actual_cat_cols = [col for col in categorical_cols if col in df.columns]

# Drop the target label column for feature encoding
X_df = df.drop(columns=[TARGET_COLUMN])

# One-Hot Encode the categorical features
X_df_encoded = pd.get_dummies(X_df, columns=actual_cat_cols, drop_first=True)
X_full = X_df_encoded.values # Full feature array for testing
print(f"Total features after encoding: {X_full.shape[1]}")

# Filter for Training (Normal Data Only)
X_normal_full = X_full[y_full == 0]
print(f"Data for Autoencoder Training (Normal): {X_normal_full.shape}")

# Initialize and Fit the Scaler ONLY on normal data
scaler = StandardScaler()
X_normal_scaled = scaler.fit_transform(X_normal_full)

# Save the scaler object for Phase 4 deployment
with open(SCALER_FILENAME, 'wb') as file:
    pickle.dump(scaler, file)
print(f"Scaler saved to {SCALER_FILENAME}.")

# Scale the ENTIRE dataset using the fitted scaler (for final evaluation)
X_test_full_scaled = scaler.transform(X_full)

# Split the scaled normal data into training and validation sets
X_train_normal, X_val_normal = train_test_split(
    X_normal_scaled, test_size=0.1, random_state=42
)
print(f"Final Training Set shape (Normal): {X_train_normal.shape}")


## 2. AUTOENCODER ARCHITECTURE AND TRAINING
print("\n--- Training Model ---")

INPUT_DIM = X_train_normal.shape[1]

# Encoder Layers
input_layer = Input(shape=(INPUT_DIM, ), name="input_layer")
encoded = Dense(64, activation="relu", name="encoder_1")(input_layer)
encoded = Dense(32, activation="relu", name="encoder_2")(encoded)
latent_view = Dense(LATENT_DIM, activation="relu", name="latent_view")(encoded)

# Decoder Layers
decoded = Dense(32, activation='relu', name="decoder_2")(latent_view)
decoded = Dense(64, activation='relu', name="decoder_1")(decoded)
output_layer = Dense(INPUT_DIM, activation='linear', name="output_layer")(decoded)

autoencoder = Model(inputs=input_layer, outputs=output_layer)
autoencoder.compile(optimizer='adam', loss='mse') 

# Train the model
history = autoencoder.fit(
    X_train_normal, X_train_normal, # Input and Output are the same (reconstruction)
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    validation_data=(X_val_normal, X_val_normal),
    verbose=1,
    callbacks=[
        ModelCheckpoint(MODEL_FILENAME, save_best_only=True, monitor='val_loss', mode='min')
    ]
)

## 3. EVALUATION AND THRESHOLD SETTING (Memory-Safe Batching)
print("\n--- Model Evaluation and Threshold Setting ---")

autoencoder.load_weights(MODEL_FILENAME) # Load the best saved weights

# Use a generator/loop to calculate Reconstruction Error in batches
print("Calculating Reconstruction Error in Batches...")
all_mse = []

for i in range(0, X_test_full_scaled.shape[0], EVAL_BATCH_SIZE):
    # Slice the input data for the current batch
    X_batch = X_test_full_scaled[i:i + EVAL_BATCH_SIZE]
    
    # Predict the reconstruction for the batch
    reconstructions_batch = autoencoder.predict(X_batch, verbose=0)
    
    # Calculate MSE for the batch
    mse_batch = np.mean(np.power(X_batch - reconstructions_batch, 2), axis=1)
    
    # Store the results
    all_mse.append(mse_batch)
    
    if i % (EVAL_BATCH_SIZE * 10) == 0:
        print(f"Processed {i} records...")

# Concatenate all batch MSE results into a single array
mse = np.concatenate(all_mse)
print(f"Total Reconstruction Errors calculated for {mse.shape[0]} records.")

# Find the threshold based on the MSE of the normal data
NORMAL_MSE = mse[y_full == 0]
THRESHOLD = np.quantile(NORMAL_MSE, THRESHOLD_PERCENTILE) 

# Classify points as anomaly (1) or normal (0)
y_pred = (mse >= THRESHOLD).astype(int)

# 4. Final Report
print(f"\n--- Final Performance Report ---")
print(f"Optimal Reconstruction Error Threshold ({THRESHOLD_PERCENTILE*100:.0f}%): {THRESHOLD:.6f}")
print("\nClassification Report (Anomalies vs. Normal)")
print(classification_report(y_full, y_pred, target_names=['Normal (0)', 'Anomaly (1)']))

# Save the final threshold for Phase 4
with open(THRESHOLD_FILENAME, 'w') as f:
    f.write(str(THRESHOLD))
    
print(f"\nModel saved to {MODEL_FILENAME}, Scaler to {SCALER_FILENAME}, and Threshold to {THRESHOLD_FILENAME}.")
print("\n--- Phase 3 Complete! Ready for Real-Time Deployment (Phase 4) ---")