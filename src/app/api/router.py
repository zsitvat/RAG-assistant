from fastapi import APIRouter

from app.api.routes import chat, evaluation, health, ingest, stats

router = APIRouter()
router.include_router(health.router)
router.include_router(ingest.router)
router.include_router(stats.router)
router.include_router(evaluation.router)
router.include_router(chat.router)
