from fastapi import FastAPI
import joblib
import pandas as pd


app = FastAPI()

model = joblib.load("model.pkl")
# explainer = joblib.load("artifacts/shap_explainer.pkl")

@app.post("/predict")
def predict(data: dict):

    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    # shap_values = explainer.shap_values(df)

    # explanation = dict(zip(df.columns, shap_values[0]))

    return {
        "prediction": int(prediction),
        "churn_probability": float(probability),
        # "feature_contribution": explanation
    }