import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer, LabelEncoder
from pathlib import Path
import pickle

# --- Configuration ---
DATA_FILENAME = 'cleaned_merged_cicids2017.csv'
NORMALIZED_DATA_FILENAME = 'final_normalized_data.csv'
SCALER_FILENAME = 'scaler.pkl'
ENCODER_FILENAME = 'encoder_params.pkl'
TARGET_COLUMN = 'Label'

print("Starting final ML data preparation...")

# 1. Load Data
df = pd.read_csv(Path(DATA_FILENAME))
print(f"Initial shape: {df.shape}")

# 2. Drop Identifier and Unnecessary Columns
# Drop columns that are constant, near-constant, or are flow identifiers
drop_cols = ['Flow_ID', 'Source_IP', 'Destination_IP', 'Timestamp', 'External_IP']
df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

# 3. Handle Categorical Features
# Network features like Protocol (6=TCP, 17=UDP) or Flags should be One-Hot Encoded
categorical_cols = ['Protocol', 'Fwd_Flags', 'Bwd_Flags'] # Common protocol/flag fields
# Find actual categorical columns present in your dataset
actual_cat_cols = [col for col in categorical_cols if col in df.columns]

# One-Hot Encode the remaining categorical features
df_encoded = pd.get_dummies(df, columns=actual_cat_cols, drop_first=True)
print(f"Shape after One-Hot Encoding: {df_encoded.shape}")

# 4. Separate Target and Features
X = df_encoded.drop(columns=[TARGET_COLUMN])
y = df_encoded[TARGET_COLUMN]

# 5. Filter for Training (Normal Data Only)
# Autoencoders must be trained ONLY on normal (BENIGN) data.
X_normal = X[y == 0]
y_normal = y[y == 0]
print(f"Data for Autoencoder Training (Normal): {X_normal.shape}")

# 6. Scaling (Normalization)
# All features must be normalized for a Neural Network to train efficiently.
scaler = StandardScaler()

# Fit the scaler ONLY on the normal data to prevent data leakage from anomalies
X_normal_scaled = scaler.fit_transform(X_normal)

# Save the scaler object for use in the real-time Prediction Service (Phase 4)
with open(SCALER_FILENAME, 'wb') as file:
    pickle.dump(scaler, file)
print(f"Scaler saved to {SCALER_FILENAME}. Necessary for real-time inference.")

# 7. Create Final Training and Testing Splits (of the Normal Data)
# Split the normal data into training (used for the model) and a validation/test set
X_train_normal, X_test_normal, _, _ = train_test_split(
    X_normal_scaled, y_normal, test_size=0.2, random_state=42
)
print(f"Final Training Set shape (Normal): {X_train_normal.shape}")

# 8. Save Final Prepared Data (Optional, but good for quick model retraining)
# If you want to use the original data for testing later, you might want to scale the entire dataset here
# X_full_scaled = scaler.transform(X.values) # <-- Already commented out
# df_final = pd.DataFrame(X_full_scaled, columns=X.columns)
# df_final[TARGET_COLUMN] = y.values
# df_final.to_csv(NORMALIZED_DATA_FILENAME, index=False) # <-- COMMENT THIS OUT TOO!

print("Data preparation complete. Ready for model training.")