import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv("datasets/cicddos2019_dataset.csv")

# Split into Offline Training (60%) and Temp (40%)
offline_train, temp = train_test_split(data, test_size=0.4, random_state=42)

# Split Temp into Online Training (75% of temp) and Validation Holdout (25% of temp)
online_train, validation_holdout = train_test_split(temp, test_size=0.25, random_state=42)

# Check sizes
print(f"Offline Training size: {len(offline_train)}")
print(f"Online Training size: {len(online_train)}")
print(f"Validation Holdout size: {len(validation_holdout)}")

# Save the offline training data
offline_train.to_csv("cicddos2019_offline_train.csv", index=False)

# Save the online training data
online_train.to_csv("cicddos2019_online_train.csv", index=False)

# Save the validation holdout data
validation_holdout.to_csv("cicddos2019_validation_holdout.csv", index=False)
