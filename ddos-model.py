import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from river import stream
from river.forest import ARFClassifier
from river.metrics import Accuracy, Precision, Recall, F1
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

    # Split into 70% for offline and 30% for online
    split_index = int(0.7 * len(data))
    data = data[:split_index]

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

    return pd.DataFrame(X_scaled, columns=X.columns), pd.Series(y_resampled), label_encoder, scaler

# Step 2: Offline Training and Model Saving
def train_offline_rf(filepath, selected_columns, model_path):
    # Preprocess the data
    X, y, label_encoder, scaler = preprocess_data(filepath, selected_columns)

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

    # Save the model, label encoder, and scaler to disk
    with open(model_path, 'wb') as f:
        pickle.dump({'model': model, 'label_encoder': label_encoder, 'scaler': scaler}, f)

    print(f"Model saved to {model_path}")

# ==============================
# PHASE 2: REAL-TIME INTEGRATION
# ==============================

# Step 3: Online Training with Kafka and Loaded Model
def train_online_rf_with_kafka(topic, selected_columns, model_path):
    # Load the pre-trained model, label encoder, and scaler
    with open(model_path, 'rb') as f:
        saved_objects = pickle.load(f)
        model = saved_objects['model']
        label_encoder = saved_objects['label_encoder']

    print("Loaded pre-trained model.")

    # Initialize Kafka Consumer
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers='localhost:9092',
        value_deserializer=lambda x: json.loads(x.decode('utf-8'))
    )

    # Metrics
    metrics = {
        'accuracy': Accuracy(),
        'precision': Precision(),
        'recall': Recall()
    }

    print("Starting Online Training...")

    # Online Training Loop
    for message in consumer:
        # Deserialize Kafka message
        record = message.value

        # Extract features and target from record
        x = {col: record[col] for col in selected_columns if col != 'Class'}
        y = label_encoder.transform([record['Class']])[0]

        # Train the model incrementally
        y_pred = model.predict_one(x)
        model.learn_one(x, y)

        # Update metrics
        for name, metric in metrics.items():
            metric.update(y, y_pred)

        # Print metrics periodically
        print({name: metric.get() for name, metric in metrics.items()})

# ================
# MODEL VALIDATION
# ================

# Function to preprocess the holdout data
def preprocess_holdout(filepath, selected_columns, scaler, label_encoder):
    # Load the dataset
    data = pd.read_csv(filepath)

    # Select relevant columns (Feature extraction)
    data = data[selected_columns]

    # Split into 30% holdout set
    split_index = int(0.7 * len(data))
    holdout_data = data[split_index:]

    # Encode target labels
    holdout_data['Class'] = label_encoder.transform(holdout_data['Class'])

    # Separate features and target
    X_holdout = holdout_data.drop('Class', axis=1)
    y_holdout = holdout_data['Class']

    # Feature scaling using the pre-fitted scaler
    X_holdout_scaled = scaler.transform(X_holdout)

    return pd.DataFrame(X_holdout_scaled, columns=X_holdout.columns), pd.Series(y_holdout)

# Function to validate the model using a holdout dataset
def validate_with_holdout(model_path, holdout_filepath, selected_columns):
    # Load the pre-trained model, label encoder, and scaler
    with open(model_path, 'rb') as f:
        saved_objects = pickle.load(f)
        model = saved_objects['model']
        label_encoder = saved_objects['label_encoder']
        scaler = saved_objects['scaler']

    print("Loaded pre-trained model for validation.")

    # Preprocess the holdout dataset
    X_holdout, y_holdout = preprocess_holdout(holdout_filepath, selected_columns, scaler, label_encoder)

    # Convert holdout data to river stream format
    holdout_stream = zip(X_holdout.to_dict(orient='records'), y_holdout)

    # Initialize metrics
    metrics = {
        'accuracy': Accuracy(),
        'precision': Precision(),
        'recall': Recall(),
        'f1': F1()
    }

    print("Starting Holdout Validation...")

    # Perform validation
    for x, y in holdout_stream:
        y_pred = model.predict_one(x)

        # Update metrics
        for name, metric in metrics.items():
            metric.update(y, y_pred)

    # Print validation metrics
    print("Holdout Validation Metrics:", {name: metric.get() for name, metric in metrics.items()})

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

    # Step 2: Online Training with Kafka
    train_online_rf_with_kafka(topic="ddos-detection", selected_columns=selected_columns, model_path=model_path)

    # Validate model with holdout data
    validate_with_holdout(model_path=model_path, holdout_filepath=dataset_path, selected_columns=selected_columns)
