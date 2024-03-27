from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, FileResponse
import os
from pathlib import Path
import shutil
import logging
from uuid import UUID
from video_sync import create_metadata

app = FastAPI()
logging.basicConfig(level=logging.INFO)

UPLOAD_DIRECTORY = Path("uploaded_audios")
UPLOAD_DIRECTORY.mkdir(parents=True, exist_ok=True)
print(f"Upload directory is set to {UPLOAD_DIRECTORY.absolute()}")

METADATA_DIRECTORY = Path("metadata")
METADATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
print(f"Metadata directory is set to {METADATA_DIRECTORY.absolute()}")

# Stocker les chemins des fichiers audio reçus, classés par UUID de groupe
received_audios_by_group = {}

@app.post("/upload-audio/{group_id}")
async def upload_audio(group_id: UUID, audio: UploadFile = File(...)):
    content = await audio.read()
    print(f"Received file content size: {len(content)} bytes")
    try:
        print(f"Received file: {audio.filename}")
        file_location = UPLOAD_DIRECTORY / f"{group_id}_{audio.filename}"
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)
        print(f"File saved to {file_location}")

        # Ajout du fichier au groupe correspondant
        if group_id not in received_audios_by_group:
            received_audios_by_group[group_id] = []
        received_audios_by_group[group_id].append(str(file_location))

        # Vérifier si deux fichiers ont été reçus pour ce groupe
        if len(received_audios_by_group[group_id]) == 2:
            # Traitement spécifique pour deux fichiers, par exemple :
            json_path = create_metadata(received_audios_by_group[group_id], METADATA_DIRECTORY)
            # Nettoyer après traitement
            for audio_path in received_audios_by_group[group_id]:
                os.remove(audio_path)
            del received_audios_by_group[group_id]
            # Retourner une réponse appropriée
            return FileResponse(json_path)
        else:
            return JSONResponse(content={"message": "Fichier reçu, en attente du deuxième."})
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not upload the audio: {e}")
