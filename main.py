from fastapi import FastAPI
import pickle
import pandas as pd

# pydantic -> data validation yani gelen verilerin doğruluğunu kontrol etmek için kullanılır.
from pydantic import BaseModel

app = FastAPI()

@app.get("/")
def home():
    return {"mesaj":"ML model Api sine hoşgeldiniz!"}

# Aşağıdaki class yapısını BaseModel'den 
# türetmemizin sebebi pydantic kütüphanesini kullanarak
# gelen verilerin doğruluğunu kontrol etmek.
class ml_model_schema(BaseModel):
    Pregnancies:int
    Glucose:int
    BloodPressure:int
    SkinThickness:int
    Insulin:int
    BMI:float
    DiabetesPedigreeFunction:float
    Age:int

@app.post("/pridict/knn/")
def knn_predict(predict_values:ml_model_schema):
    load_model = pickle.load(open("knn_model.pkl","rb"))

    # predict_values -> gelen verileri bir dataframe'e çeviriyoruz.
    # predict_values.dict().values() -> gelen verilerin değerlerini alıyoruz.
    df = pd.DataFrame(
        [predict_values.dict().values()],
        columns=predict_values.dict().keys()
        )
    # print(df)
    predict =load_model.predict(df)
    return {"Predict":int(predict[0])}

@app.post("/pridict/LR/")
def LR_predict(predict_values:ml_model_schema):
    load_model = pickle.load(open("LR_model.pkl","rb"))
    df = pd.DataFrame([predict_values.dict().values()],columns=predict_values.dict().keys())
    predict =load_model.predict(df)
    return {"Predict":int(predict[0])}