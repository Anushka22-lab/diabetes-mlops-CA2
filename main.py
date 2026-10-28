
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Create FastAPI application
app = FastAPI(title="Diabetes Prediction API")

# Load trained model and feature names
artifact = joblib.load("model.joblib")

model = artifact["model"]
features = artifact["features"]


# Define the input format
class DiabetesInput(BaseModel):
    HighBP: int
    HighChol: int
    CholCheck: int
    BMI: float
    Smoker: int
    Stroke: int
    HeartDiseaseorAttack: int
    PhysActivity: int
    Fruits: int
    Veggies: int
    HvyAlcoholConsump: int
    AnyHealthcare: int
    NoDocbcCost: int
    GenHlth: int
    MentHlth: int
    PhysHlth: int
    DiffWalk: int
    Sex: int
    Age: int
    Education: int
    Income: int


# Home endpoint
@app.get("/")
def home():
    return {"message": "Diabetes Prediction API is running!"}


# Prediction endpoint
@app.post("/predict")
def predict(data: DiabetesInput):
    input_data = pd.DataFrame(
        [data.model_dump()],
        columns=features
    )

    prediction = int(model.predict(input_data)[0])

    return {
        "prediction": prediction,
        "result": (
            "No diabetes"
            if prediction == 0
            else "Prediabetes or diabetes"
        )
    }
