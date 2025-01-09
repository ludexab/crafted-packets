import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from river import stream
from river.forest import ARFClassifier
from river.metrics import Accuracy, Precision, Recall
from kafka import KafkaConsumer
import pickle
import json

#===============================
# PHASE 1: OFFLINE DEVELOPMENT
#===============================

# Step 1: Data Preprocessing
def preprocess_data(filepath, selected_columns):
    # Load the dataset
    data = pd.read_csv(filepath)

    # Select relevant columns (Feature extraction)
    data = data[selected_columns]

    # Encode target labels
    label_encoder = LabelEncoder()
    data['Class'] = label_encoder.fit_transform(data['Class'])

    # Separate features and target
    X = data.drop('Class', axis=1)
    y = data['Class']

    # Handle class imbalance using SMOTE
    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_resampled)

    return pd.DataFrame(X_scaled, columns=X.columns), pd.Series(y_resampled), label_encoder

# Step 2: Offline Training and Model Saving
def train_offline_rf(filepath, selected_columns, model_path):
    # Preprocess the data
    X, y, label_encoder = preprocess_data(filepath, selected_columns)

    # Convert to river stream format
    data_stream = stream.iter_pandas(X, y)

    # Initialize Online Random Forest Classifier
    model = ARFClassifier(
        n_models=10,            # Default: Number of trees in the ensemble
        max_features=None,      # Default: sqrt(number of features)
        lambda_value=6,         # Default: Weight of each tree in ensemble
        grace_period=50,        # Default: Number of instances before splitting a node
        split_criterion='gini', # Default: Gini index for classification splits
        seed=42
    )

    # Metrics
    metrics = {
        'accuracy': Accuracy(),
        'precision': Precision(),
        'recall': Recall()
    }

    # Train the model on the dataset
    print("Starting Offline Training...")
    for x, y in data_stream:
        y_pred = model.predict_one(x)
        model.learn_one(x, y)

        # Update metrics
        for name, metric in metrics.items():
            metric.update(y, y_pred)

    # Print final metrics
    print("Offline Training Metrics:", {name: metric.get() for name, metric in metrics.items()})

    # Save the model and label encoder to disk
    with open(model_path, 'wb') as f:
        pickle.dump({'model': model, 'label_encoder': label_encoder}, f)

    print(f"Model saved to {model_path}")

# Full Pipeline Execution (Example)
if __name__ == "__main__":
    # Define selected columns
    selected_columns = [
        'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
        'Flow Bytes/s', 'Flow Packets/s', 'Fwd Packet Length Mean',
        'Bwd Packet Length Mean', 'Packet Length Mean', 'Packet Length Std',
        'Flow IAT Mean', 'Flow IAT Std', 'Fwd IAT Mean', 'Bwd IAT Mean',
        'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count',
        'PSH Flag Count', 'ACK Flag Count', 'Class'
    ]

    # Filepath to dataset
    dataset_path = "datasets/cicddos2019_dataset.csv"

    # Path to save/load the model
    model_path = "models/online-random-forest.pkl"

    # Step 1: Offline Training and Saving
    train_offline_rf(filepath=dataset_path, selected_columns=selected_columns, model_path=model_path)
