from fastapi import APIRouter, Request, UploadFile, File, HTTPException
from fastapi.templating import Jinja2Templates
from app.services.inference import run_inference_image, run_inference_dicom
from app.utils.file_handler import save_temp_file
from fastapi import Depends

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

ALLOWED_IMAGE_TYPES = [
    "image/png",
    "image/jpeg",
    "application/dicom",
    "application/octet-stream"
]

@router.get("/detect-medical-tamper")
async def detect_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@router.post("/detect-medical-tamper")
async def detect_medical_tamper(file: UploadFile = File(...)):
    # Check file type
    if file.content_type not in ALLOWED_IMAGE_TYPES and not file.filename.lower().endswith(".dcm"):
        raise HTTPException(status_code=400, detail="Invalid image format")

    # Save uploaded image temporarily
    file_path = save_temp_file(file)

    # Run ML inference
    filename = file.filename.lower()
    if filename.endswith((".png", ".jpg", ".jpeg")):
        result = run_inference_image(file_path)
    elif filename.endswith(".dcm"):
        result = run_inference_dicom(file_path)
    else:
        raise HTTPException(400, "Unsupported file type")

    return result
