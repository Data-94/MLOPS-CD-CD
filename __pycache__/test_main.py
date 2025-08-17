from fastapi.testclient import TestClient
from main import app


client = TestClient(app)

def test_knn_predict():
    payload = {
        "Pregnancies": 6,
        "Glucose": 148,
        "BloodPressure": 72,
        "SkinThickness": 35,
        "Insulin": 0,
        "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.627,
        "Age": 50
    }
    response = client.post("/pridict/knn/",json=payload)

    assert response.status_code == 200

    # response.json() -> gelen veriyi json formatına çevirir.
    # assert "Predict" in response.json() -> gelen veride "Predict" kelimesi var mı diye kontrol eder.
    assert "Predict" in response.json()

    # assert response.json()["Predict"] in [0,1] -> gelen verinin içindeki "Predict" key'ine karşılık gelen değer 0 veya 1 mi diye kontrol eder.
    assert response.json()["Predict"] in [0,1]

def test_LR_predict():
    payload = {
        "Pregnancies": 6,
        "Glucose": 148,
        "BloodPressure": 72,
        "SkinThickness": 35,
        "Insulin": 0,
        "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.627,
        "Age": 50
        }
    response = client.post("/pridict/LR/",json=payload)

    assert response.status_code == 200
    assert "Predict" in response.json()
    assert response.json()["Predict"] in [0,1]
