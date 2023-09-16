import os 
import tensorflow as tf
import pandas as pd
import mlflow
import mlflow.tensorflow
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Set the MLflow tracking URI
mlflow.set_tracking_uri('mlflow')  # Replace with your MLflow tracking URI

# Create a new experiment
experiment_name = 'compareModels'
mlflow.set_experiment(experiment_name)

# Load the dataset
df = pd.read_csv('data/weatherHistory.csv')

# Specify the features and target column names based on your dataset
feature_columns = ['Temperature (C)', 'Apparent Temperature (C)', 'Humidity',
                   'Wind Speed (km/h)', 'Wind Bearing (degrees)', 'Visibility (km)',
                   'Loud Cover', 'Pressure (millibars)'
                   ]

target_column = 'Summary'

# Split the dataset into features and target variables
X = df[feature_columns].values
y = df[target_column].values

# Encode the target column
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the dataset into training, validation, and testing sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=42)

# Define the models

# Define random forest model
rf_model = RandomForestClassifier(n_estimators=25, random_state=42)

# Define Tensorflow model
tf_model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Train the Random Forest model
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)
print("Random Forest Accuracy: {:.4f}".format(rf_accuracy))

# Train the TensorFlow model
tf_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
tf_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

# Evaluate the TensorFlow model on the testing set
tf_loss, tf_accuracy = tf_model.evaluate(X_test, y_test)
print("TensorFlow Testing Loss: {:.4f}".format(tf_loss))
print("TensorFlow Testing Accuracy: {:.4f}".format(tf_accuracy))

# Compare the models and log the results
with mlflow.start_run():
    mlflow.log_metric('Random Forest Accuracy', rf_accuracy)
    mlflow.log_metric('TensorFlow Accuracy', tf_accuracy)

    if rf_accuracy > tf_accuracy:
        # Random Forest is the winner
        mlflow.log_param('Winner Model', 'Random Forest')
        # Save the Random Forest model using pickle
        with open('saved_model/random_forest_model.pkl', 'wb') as file:
            pickle.dump(rf_model, file)
    else:
        # TensorFlow is the winner
        mlflow.log_param('Winner Model', 'TensorFlow')
        # Save the TensorFlow model using MLflow
        mlflow.tensorflow.save_model(tf_model, 'saved_model/tensorflow_model')


#run api.py to serve the model 
os.system('python3 src/api.py')