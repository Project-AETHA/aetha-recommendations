from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import recommendation_model as model

# FastAPI app initialization
app = FastAPI()

# Initialize the model variables
R_pred = None
user_index = None
novel_index = None
users = None
novels = None

class RecommendationResponse(BaseModel):
    user_id: str
    recommendations: List[Dict[str, str]]

@app.on_event("startup")
async def setup():
    global R_pred, user_index, novel_index, users, novels
    try:
        # Load data asynchronously from CSV
        user_item_matrix, users, novels = await model.load_data_from_csv('./dummy_data/clicks.csv', './dummy_data/novels.csv')
        
        # print((list(novels.items()))[:2])

        # Apply SVD or any other processing here
        R_pred, user_index, novel_index = await model.apply_svd(user_item_matrix)

        print("Data loaded successfully")
    except Exception as e:
        print(f"Error during setup: {e}")

@app.get("/recommendations/{user_id}", response_model=RecommendationResponse)
async def recommend(user_id: str):
    try:
        # Add debug logging
        print(f"Request received for user_id: {user_id}")
        
        # Ensure we have valid model and data loaded
        if R_pred is None or users is None or novels is None:
            raise HTTPException(status_code=500, detail="Model data is not loaded properly.")
        
        # Generate recommendations
        recommendations = await model.recommend_items(user_id, R_pred, users, novels)
        return {"user_id": user_id, "recommendations": recommendations}
    
    except HTTPException as http_err:
        print(f"HTTP error occurred: {http_err.detail}")
        raise http_err
    except Exception as e:
        # Log and provide a more detailed response to help debug
        print(f"Error in recommend_items: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while generating recommendations")
