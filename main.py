from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import os
from pathlib import Path
from video_sync import create_metadata, simple_json
import shutil
from fastapi.responses import FileResponse
import logging

app = FastAPI()
logging.basicConfig(level=logging.INFO)

UPLOAD_DIRECTORY = Path("uploaded_audios")
if not UPLOAD_DIRECTORY.exists():
    UPLOAD_DIRECTORY.mkdir(parents=True)
print(f"Upload directory is set to {UPLOAD_DIRECTORY.absolute()}")

METADATA_DIRECTORY = Path("metadata")
if not METADATA_DIRECTORY.exists():
    METADATA_DIRECTORY.mkdir(parents=True)
print(f"Metadata directory is set to {METADATA_DIRECTORY.absolute()}")
received_audios = []

@app.get("/list-audios/")
def list_audios():
    files = list(Path("/app/uploaded_audios/").glob('*'))
    return {"files": [str(file) for file in files]}


@app.post("/upload-audio/")
async def upload_audio(audio: UploadFile = File(...)):
    content = await audio.read()
    print(f"Received file content size: {len(content)} bytes")
    try:
        print(f"Received file: {audio.filename}")
        print(f"File size: {audio.file._file.tell()} bytes")
        file_location = UPLOAD_DIRECTORY / audio.filename
        with open(file_location, "wb") as buffer:
            audio.file.seek(0)  # Remet le pointeur au début du fichier
            shutil.copyfileobj(audio.file, buffer)
        print(f"File saved to {file_location}")
        
        received_audios.append(str(file_location))

        if len(received_audios) >= 2:
            print("About to create metadata")
            json_path = create_metadata(received_audios, METADATA_DIRECTORY)
            logging.debug(f"Metadata created at {json_path}")
            print(f"Metadata created at: {json_path}")
            
            for audio_path in received_audios:
                os.remove(audio_path)
            received_audios.clear()
            print(list_audios())
            # Retourner directement le fichier JSON comme réponse
            return FileResponse(json_path)
        else:
            return JSONResponse(content={"message": "Audio received and waiting for more to process."})
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not upload the audio: {e}")
