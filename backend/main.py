from fastapi import FastAPI

from backend.api.upload import router as audio_router

app = FastAPI(
    title="AcousticSpace API",
    description="Backend API for deepfake audio detection",
    version="1.0.0",
)

app.include_router(audio_router)


@app.get("/")
def home():
    return {
        "message": "AcousticSpace API is running successfully"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }