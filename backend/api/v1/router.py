from fastapi import APIRouter 
from backend.api.v1.endpoints import generate, models, system, tokenizer, visualize

api_router = APIRouter()
api_router.include_router(system.router)
api_router.include_router(models.router)
api_router.include_router(generate.router)
api_router.include_router(tokenizer.router)
api_router.include_router(visualize.router)