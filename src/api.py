import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

# Load the saved model
with open("saved_model/random_forest_model.pkl", "rb") as file:
    model = pickle.load(file)

# Define the FastAPI app
app = FastAPI()

# Define the request body schema using Pydantic
class PredictionRequest(BaseModel):
    features: list

# Define the prediction endpoint
@app.post("/predict")
def predict():
    # Load the CSV file
    df = pd.read_csv("data/weatherHistory.csv")

    # Select the specific features for prediction
    feature_columns = ['Temperature (C)', 'Apparent Temperature (C)', 'Humidity',
                       'Wind Speed (km/h)', 'Wind Bearing (degrees)', 'Visibility (km)',
                       'Loud Cover', 'Pressure (millibars)']
    df = df[feature_columns]

    # Prepare the input data for prediction
    input_data = df.values

    # Make predictions using the loaded model
    predictions = model.predict(input_data)

    # Format the predictions and return the response
    return {
        "predictions": predictions.tolist()
    }


# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
