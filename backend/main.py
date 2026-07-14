from fastapi import FastAPI

app = FastAPI(
    title="AcousticSpace API",
    description="Backend API for deepfake audio detection",
    version="1.0.0",
)


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