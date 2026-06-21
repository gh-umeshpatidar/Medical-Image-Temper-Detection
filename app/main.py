import os

from fastapi import FastAPI, Request

from fastapi.middleware.cors import CORSMiddleware

from fastapi.staticfiles import StaticFiles

from fastapi.templating import Jinja2Templates

from app.api.routes.health import router as health_router

from app.api.routes.detect import router as detect_router

app = FastAPI(
    title="Medical Image Tamper Detection API",
    description="Detects deepfake or tampered medical images",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

static_dir = os.path.join(BASE_DIR, "static")

output_dir = os.path.join(os.path.dirname(BASE_DIR), "output")

os.makedirs(static_dir, exist_ok=True)

os.makedirs(output_dir, exist_ok=True)

app.mount("/static", StaticFiles(directory=static_dir), name="static")

app.mount("/output", StaticFiles(directory=output_dir), name="output")

templates_dir = os.path.join(BASE_DIR, "templates")

templates = Jinja2Templates(directory=templates_dir)

app.include_router(health_router, prefix="/api")

app.include_router(detect_router, prefix="/api")


@app.get("/")
def home(request: Request):

    return templates.TemplateResponse("Home.html", {"request": request})

@app.get("/health")
def health_check():

    return {"status": "ok", "message": "Medical Tamper Detection routing work properly"}
