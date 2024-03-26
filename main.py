from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import os
from pathlib import Path
from video_sync import create_metadata, simple_json
import shutil
from fastapi.responses import FileResponse
app = FastAPI()

UPLOAD_DIRECTORY = Path("uploaded_audios")
if not UPLOAD_DIRECTORY.exists():
    UPLOAD_DIRECTORY.mkdir(parents=True)
print(f"Upload directory is set to {UPLOAD_DIRECTORY.absolute()}")

METADATA_DIRECTORY = './'
received_audios = []



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
            print(f"Metadata created at: {json_path}")
            
            for audio_path in received_audios:
                os.remove(audio_path)
            received_audios.clear()

            # Retourner directement le fichier JSON comme réponse
            return FileResponse(json_path)
        else:
            return JSONResponse(content={"message": "Audio received and waiting for more to process."})
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not upload the audio: {e}")
