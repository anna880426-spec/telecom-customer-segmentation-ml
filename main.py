import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel   # uvicorn main:app --reload

app = FastAPI()

# load model and preprocessors
model = joblib.load('model.pkl')
encoder = joblib.load('encoder.pkl')
scaler = joblib.load('scaler.pkl')

class CustomerData(BaseModel):
    age: int
    age_group: str
    job: str
    marital: str
    education: str
    default: str
    housing: str
    loan: str
    contact: str
    month: str
    day_of_week: str
    campaign: int
    pdays: float
    previous: int
    poutcome: str
    emp_var_rate: float
    cons_price_idx: float
    cons_conf_idx: float
    euribor3m: float
    nr_employed: float

@app.post('/predict')
def predict(data: CustomerData):
    binary_data = [[
    int(data.age >= 20 and data.age <= 40 and data.education in ['university.degree', 'high.school']),
    int(data.previous > 0),
    int(data.campaign <= 2)
]]
    cat_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan',
               'contact', 'month', 'day_of_week', 'poutcome', 'age_group']
    nu_columns = ['campaign', 'pdays', 'previous', 'emp_var_rate',
                  'cons_price_idx', 'cons_conf_idx', 'euribor3m', 'nr_employed']

    input_dict = data.dict()

    cat_data = [[input_dict[col] for col in cat_columns]]
    num_data = [[input_dict[col] for col in nu_columns]]

    cat_encoded = encoder.transform(cat_data)
    num_scaled = scaler.transform(num_data)

    X = np.concatenate([binary_data, cat_encoded, num_scaled], axis=1)

    prob = model.predict_proba(X)[0][1]
    prediction = int(prob >= 0.4)

    return {
        'subscription_probability': round(float(prob), 4),
        'prediction': prediction
    }