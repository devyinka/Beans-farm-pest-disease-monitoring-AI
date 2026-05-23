from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle

# Initialize the API
app = FastAPI(title="Beans Farm AI Microservice")

# Load the AI Brain when the server boots up
print("Loading AI Model...")
with open("beans_rf_model.pkl", "rb") as f:
    rf_model = pickle.load(f)
print("Model Loaded Successfully!")

# Define the expected input structure using Pydantic
class FarmPayload(BaseModel):
    Time_of_Day: int 
    Plant_Age_Days: int
    Max_Temp_C: float
    Min_Temp_C: float
    Avg_Day_Hum: float
    Avg_Night_Hum: float
    Soil_Moisture: float
    Sunlight_Hours: float
    Rain_Level_mm: float
    Leaf_Wetness_Hours: float
    Cumulative_Stress_Index: float
    Hot_Days_Past_10_Days: int
    Wet_Nights_Past_10_Days: int
    Dry_Soil_Days_Past_10_Days: int
    Flooded_Days_Past_10_Days: int
    Rainy_Days_Past_10_Days: int
    Total_Rain_Volume_mm_Past_10_Days: float

@app.post("/predict")
async def predict_threat(payload: FarmPayload):
    try:
        #conver the incoming JSON payload into a DataFrame with the same structure as the training data
        df = pd.DataFrame([payload.model_dump()])
        
        #  Ask the AI for the prediction and the confidence percentage
        raw_prediction = rf_model.predict(df)[0]
        probabilities = rf_model.predict_proba(df)[0]
        confidence = round(max(probabilities) * 100, 2)
        
        # processing the raw prediction to match what my express server expects. If it's "Safe", we keep it simple. If it's a disease, we split it into status and name.
        if raw_prediction == "Safe":
            status = "Safe"
            name = "None"
        else:
           # The raw prediction is in the format "Disease: Name", so we split it to get the status and the name.
            parts = raw_prediction.split(": ") 
            status = parts[0]
            name = parts[1]

        # Send the perfect JSON back to Node.js
        return {
            "status": status,
            "threat_name": name,
            "percentage": confidence
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

        