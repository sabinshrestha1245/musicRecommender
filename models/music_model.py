from pydantic import BaseModel

class SongRecommendationRequest(BaseModel):
    song_title: str
