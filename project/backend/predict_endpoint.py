from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
import joblib
import numpy as np

# http://127.0.0.1:8000/
# swagger

MODEL_PATH = "../models/prod_model.pkl"

app = FastAPI()

model = joblib.load(MODEL_PATH)


class PredictionRequest(BaseModel):
    features: list


@app.post("/predict")
def predict(request: PredictionRequest):

    if len(request.features) != 28:
        raise HTTPException(status_code=400, detail=f"Number of features must be 29, got {len(request.features)}.")

    input_data = np.array(request.features).reshape(1, -1)

    prediction = model.predict(input_data)

    return {"prediction": prediction[0]}


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
