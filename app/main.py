import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Import API routes
from app.api.routes.health import router as health_router
from app.api.routes.detect import router as detect_router

# Create FastAPI app
app = FastAPI(
    title="Medical Image Tamper Detection API",
    description="Detects deepfake or tampered medical images",
    version="1.0.0"
)

# Allow other websites/apps to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all (restrict later)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (CSS, JS, images)
BASE_DIR = os.path.dirname(__file__)
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

# Jinja2 templates
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Register routes
app.include_router(health_router, prefix="/api")
app.include_router(detect_router, prefix="/api")

@app.get("/")
def home(request: Request):
    # return templates.TemplateResponse("Home.html", {"request": request})
    return "Welcome to the Medical Image Tamper Detection API. Visit /api/detect-medical-tamper for the detection interface."