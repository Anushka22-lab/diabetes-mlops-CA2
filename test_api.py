
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_home_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == (
        "Diabetes Prediction API is running!"
    )


def test_prediction_endpoint():
    data = {
        "HighBP": 1,
        "HighChol": 1,
        "CholCheck": 1,
        "BMI": 28,
        "Smoker": 0,
        "Stroke": 0,
        "HeartDiseaseorAttack": 0,
        "PhysActivity": 1,
        "Fruits": 1,
        "Veggies": 1,
        "HvyAlcoholConsump": 0,
        "AnyHealthcare": 1,
        "NoDocbcCost": 0,
        "GenHlth": 3,
        "MentHlth": 2,
        "PhysHlth": 2,
        "DiffWalk": 0,
        "Sex": 1,
        "Age": 9,
        "Education": 5,
        "Income": 6
    }

    response = client.post("/predict", json=data)

    assert response.status_code == 200

    result = response.json()
    assert result["prediction"] in [0, 1]
    assert "result" in result
