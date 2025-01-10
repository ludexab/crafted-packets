import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from river.forest import ARFClassifier
from river.metrics import Accuracy
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Paths
MODEL_PATH = "models/online-random-forest-1.pkl"

# Ensure the models directory exists
os.makedirs("models", exist_ok=True)

# Load and preprocess data
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    features = data.iloc[:, :-1]
    labels = data.iloc[:, -1]

    # Standardize features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    return features, labels

def preprocess_holdout_data(holdout_data):
    # Ensure data is preprocessed the same way as training data
    holdout_data = holdout_data.copy()  # Avoid SettingWithCopyWarning
    features = holdout_data.iloc[:, :-1]
    labels = holdout_data.iloc[:, -1]

    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    return features, labels

# Train model
def train_offline_model(features, labels):
    # Use an AdaptiveRandomForestClassifier for training
    model = ARFClassifier()
    metric = Accuracy()

    for x, y in zip(features, labels):
        y_pred = model.predict_one(x)
        metric.update(y, y_pred)
        model.learn_one(x, y)

    return model, metric

def evaluate_model(model, features, labels):
    predictions = [model.predict_one(x) for x in features]
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='weighted', zero_division=0)
    recall = recall_score(labels, predictions, average='weighted', zero_division=0)
    f1 = f1_score(labels, predictions, average='weighted', zero_division=0)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# Main pipeline
def main():
    # Load training and holdout datasets -- datasets/cicddos2019_dataset.csv
    training_data = load_data("datasets/cicddos2019_offline_train.csv")
    holdout_data = load_data("datasets/cicddos2019_validation_holdout.csv")

    # Preprocess datasets
    X_train, y_train = preprocess_data(training_data)
    X_holdout, y_holdout = preprocess_holdout_data(holdout_data)

    # Encode labels
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_holdout = label_encoder.transform(y_holdout)

    # Train offline model
    print("Starting Offline Training...")
    model, training_metric = train_offline_model(X_train, y_train)
    print(f"Offline Training Metrics: {training_metric}")

    # Save the model
    with open(MODEL_PATH, "wb") as model_file:
        pickle.dump(model, model_file)

    print(f"Model saved to {MODEL_PATH}")

    # Load model and validate on holdout set
    print("Starting Holdout Validation...")
    with open(MODEL_PATH, "rb") as model_file:
        loaded_model = pickle.load(model_file)

    holdout_metrics = evaluate_model(loaded_model, X_holdout, y_holdout)
    print(f"Holdout Validation Metrics: {holdout_metrics}")

if __name__ == "__main__":
    main()
