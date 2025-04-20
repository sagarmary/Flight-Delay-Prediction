import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Read the dataset
df = pd.read_csv('Data/delay.csv')

# Convert categorical columns using Label Encoding
label_encoders = {}
for col in ['carrier', 'origin', 'dest']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le



# Select features and target variable
X = df[['year', 'month', 'day', 'carrier', 'origin', 'dest', 'departure_delay']].values
y = df['delayed']

# Encode the target variable
le_target = LabelEncoder()
y = le_target.fit_transform(y)

# Scale numerical features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=62)

# Create and train the XGBoostClassifier with optimized parameters
xgb = XGBClassifier(
    n_estimators=300,  # More iterations
    learning_rate=0.05,  # Reduced learning rate for better generalization
    max_depth=8,  # Increase depth to capture complex patterns
    colsample_bytree=0.8,  # Use 80% features per tree
    subsample=0.9,  # Use 90% samples per tree
    scale_pos_weight=1,  # Adjust if the dataset is imbalanced
    random_state=62
)

xgb.fit(X_train, y_train)

# Make predictions
predicted_values = xgb.predict(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, predicted_values)
print("Improved Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, predicted_values))


# Saving model and label encoders to disk
pickle.dump(xgb,open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))