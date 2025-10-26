import pickle
from fastapi import FastAPI
from pydantic import BaseModel

model_file = 'pipeline_v1.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = FastAPI()

class Customer(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

@app.post('/predict')
def predict(customer: Customer):
    customer_dict = customer.dict()
    
    X = dv.transform([customer_dict])
    y_pred = model.predict_proba(X)[0, 1]
    churn = y_pred >= 0.5

    result = {
        'churn_probability': float(round(y_pred, 3)),
        'churn': bool(churn)
    }

    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=9696)