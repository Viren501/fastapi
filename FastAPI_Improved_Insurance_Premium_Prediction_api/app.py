import logging

from fastapi import FastAPI, HTTPException

from schema.user_input import UserInput
from schema.prediction_response import PredictionResponse
from model.predict import predict_output, model, MODEL_VERSION

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Insurance Premium Prediction API",
    version=MODEL_VERSION
)


# Human readable
@app.get('/')
def home():
    return {'message': 'Insurance Premium Prediction API'}


# Machine readable
@app.get('/health')
def health_check():
    return {
        'status': 'OK',
        'version': MODEL_VERSION,
        'model_loaded': model is not None
    }


@app.post('/predict', response_model=PredictionResponse)
def predict_premium(data: UserInput):

    features = {
        'bmi': data.bmi,
        'age_group': data.age_group,
        'lifestyle_risk': data.lifestyle_risk,
        'city_tier': data.city_tier,
        'income_lpa': data.income_lpa,
        'occupation': data.occupation
    }

    try:
        return predict_output(features)

    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed"
        )
