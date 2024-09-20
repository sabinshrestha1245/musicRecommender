from fastapi import FastAPI
from controllers.music_controller import router as music_router

app = FastAPI()

# Health check route
@app.get("/")
async def health_check():
    """Check if the API is working."""
    return {"status": "API is working!"}

# Include the music-related routes
app.include_router(music_router, prefix="/music")
