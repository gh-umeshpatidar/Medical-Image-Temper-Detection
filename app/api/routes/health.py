from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health_check():

    return {"status": "ok", "message": "Medical Tamper Detection routing work properly"}
