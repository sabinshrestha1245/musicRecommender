from fastapi import APIRouter, HTTPException, Query
from models.music_recommender import MusicRecommenderFromScratch

# Define the router for handling music-related requests
router = APIRouter()

# Load the recommender with the dataset
recommender = MusicRecommenderFromScratch(data_file="data.csv")

@router.get("/recommend")
async def recommend(query: str = Query(...)):
    """Get music recommendations based on a query."""
    try:
        recommendations = recommender.get_recommendations(query)
        if not recommendations:
            raise HTTPException(status_code=404, detail="No matching song, artist, or genre found")
        return {"recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/popular")
async def get_popular_music():
    """Get the top 10 popular music tracks."""
    try:
        popular_music = recommender.get_popular_music()
        return {"popular_music": popular_music}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/latest")
async def get_latest_music():
    """Get the top 10 latest music tracks."""
    try:
        latest_music = recommender.get_latest_music()
        return {"latest_music": latest_music}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
