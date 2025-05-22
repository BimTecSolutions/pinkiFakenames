from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib
import os
import uvicorn

app = FastAPI()

# Load saved model and vectorizer
model = joblib.load("name_model.pkl")
vectorizer = joblib.load("name_vectorizer.pkl")

class NameInput(BaseModel):
    name: str

@app.post("/check_name")
async def check_name(data: NameInput):
    name = data.name
    vectorized_name = vectorizer.transform([name])
    prediction = model.predict(vectorized_name)[0]
    confidence = model.predict_proba(vectorized_name)[0][prediction]
    return {"valid": bool(prediction), "confidence": float(confidence)}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Railway sets PORT env variable
    uvicorn.run(app, host="0.0.0.0", port=port)
