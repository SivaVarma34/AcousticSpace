from backend.services.audio_processor import process_audio
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile

router = APIRouter(prefix="/audio", tags=["Audio"])

UPLOAD_DIR = Path("backend/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a"}


@router.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename was provided",
        )

    extension = Path(file.filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only WAV, MP3, FLAC and M4A audio files are allowed",
        )

    unique_filename = f"{uuid4().hex}{extension}"
    destination = UPLOAD_DIR / unique_filename

    try:
        file_content = await file.read()

        if not file_content:
            raise HTTPException(
                status_code=400,
                detail="The uploaded audio file is empty",
            )

        destination.write_bytes(file_content)

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save audio file: {error}",
        ) from error

    finally:
        await file.close()

    audio_info = process_audio(str(destination))

    return {
    "message": "Audio processed successfully",
    "original_filename": file.filename,
    "saved_filename": unique_filename,
    "content_type": file.content_type,
    "size_bytes": len(file_content),
    "audio_information": audio_info,
    }