from fastapi import APIRouter

from app.api.routes import admin, chat, health

router = APIRouter()
router.include_router(health.router)
router.include_router(admin.router)
router.include_router(chat.router)
