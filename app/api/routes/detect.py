from fastapi import APIRouter, Request, UploadFile, File, HTTPException

from fastapi.templating import Jinja2Templates

from app.services.inference import run_inference

from app.utils.file_handler import save_temp_file, is_valid_file_extension

import os

router = APIRouter()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

ALLOWED_IMAGE_TYPES = [
    "image/png",
    "image/jpeg",
    "application/dicom",
    "application/octet-stream",
]


@router.get("/detect-medical-tamper")
async def detect_page(request: Request):

    return templates.TemplateResponse("index.html", {"request": request})


@router.post("/detect-medical-tamper")
async def detect_medical_tamper(file: UploadFile = File(...)):

    if not file.filename:

        raise HTTPException(status_code=400, detail="No file provided")

    if not is_valid_file_extension(file.filename):

        raise HTTPException(
            status_code=400, detail="Invalid file format. Allowed: JPG, PNG, JPEG, DCM"
        )

    try:

        file_path = save_temp_file(file)

    except ValueError as e:

        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:

        raise HTTPException(status_code=500, detail=f"File upload error: {str(e)}")

    try:

        result = run_inference(file_path)

        return result

    except ValueError as e:

        raise HTTPException(status_code=400, detail=f"Processing error: {str(e)}")

    except Exception as e:

        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    finally:

        try:

            if os.path.exists(file_path):

                os.remove(file_path)

        except:

            pass
