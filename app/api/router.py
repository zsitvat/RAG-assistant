from fastapi import APIRouter

from app.api.routes import admin, health

router = APIRouter()
router.include_router(health.router)
router.include_router(admin.router)
