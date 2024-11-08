import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from typing import List, Dict
from fastapi import HTTPException

# Function to load data from CSV
async def load_data_from_csv(clicks_file: str, novels_file: str):
    try:
        # Load the clicks data and novels data from CSV
        clicks = pd.read_csv(clicks_file)
        novels = pd.read_csv(novels_file)

        print(f"Data loaded from {clicks_file} and {novels_file}")

        # Calculate the click_count by grouping by 'user' and 'novel'
        click_count = clicks.groupby(['user', 'novel']).size().reset_index(name='click_count')

        # Create user-item matrix from clicks data (user, novel)
        # Setting dtype to float explicitly and filling NaNs with 0.0
        user_item_matrix = click_count.pivot(index='user', columns='novel', values='click_count').fillna(0).astype(float)
        print(user_item_matrix.dtypes)

        # Extract unique users and novels
        users = user_item_matrix.index.tolist()
        novels = novels[['id', 'title']].set_index('id').to_dict()['title']  # Mapping novel ids to titles

        return user_item_matrix, users, novels
    except Exception as e:
        raise Exception(f"Error loading data from CSV: {e}")


async def apply_svd(user_item_matrix: pd.DataFrame, k: int = 50):
    try:

        # print(user_item_matrix.head())
        # print(user_item_matrix.dtypes)

        # Convert all values to numeric, coerce non-numeric values to NaN, then replace NaN with 0
        user_item_matrix = user_item_matrix.applymap(lambda x: pd.to_numeric(x, errors='coerce')).fillna(0).astype(float)
        
        # print("Data types after conversion:")
        # print(user_item_matrix.dtypes)  # Should show float64 or int64 for all columns

        # Demean the user-item matrix (subtract the mean rating of each user)
        R_demeaned = user_item_matrix - user_item_matrix.mean(axis=1).values.reshape(-1, 1)
        
        # Convert to float to ensure compatibility with SVD
        R_demeaned = R_demeaned.values
        
        # Apply Singular Value Decomposition (SVD)
        U, sigma, Vt = svds(R_demeaned, k=min(k, min(R_demeaned.shape) - 1))

        # Convert sigma into a diagonal matrix
        sigma = np.diag(sigma)
        # print(f"U: {U.shape}, Sigma: {sigma.shape}, Vt: {Vt.shape}")
        
        # Reconstruct the approximation of the original matrix
        reconstructed_matrix = U.dot(sigma).dot(Vt)
        
        return reconstructed_matrix, user_item_matrix.index, user_item_matrix.columns
    except Exception as e:
        raise Exception(f"Error during SVD computation: {e}")

# Function to generate recommendations for a user
async def recommend_items(user_id: str, R_pred: np.ndarray, users: List[str], novels: Dict[str, str]) -> List[Dict[str, str]]:
    try:
        # Check if user_id is in the users list
        if user_id not in users:
            raise ValueError(f"User ID {user_id} not found in the user list.")
        
        # Get the index of the user
        user_idx = users.index(user_id)

        # Get the predicted ratings for the user
        predicted_ratings = R_pred[user_idx, :]

        # Sort the novels by predicted rating (highest first)
        recommended_novels_idx = np.argsort(predicted_ratings)[::-1]

        # Return the top 5 recommendations
        recommendations = []
        for idx in recommended_novels_idx[:5]:
            # Ensure novel index is treated as string
            novel_title = novels.get(str(idx), f"Unknown Novel {idx}")  # Get title or default if not found
            # Convert predicted rating to string
            recommendations.append({
                'novel_title': novel_title, 
                'predicted_rating': str(predicted_ratings[idx])
            })

        return recommendations
    except ValueError as e:
        # Catch ValueError for missing user and raise a more specific HTTP error
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        # Handle other unexpected errors
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")
