"""
FastAPI Production Near Duplicate Detection Service
"""

import logging
import uvicorn
from fastapi import FastAPI

from api.lifespan import lifespan
from api.routes import router

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Near Duplicate Detection API", 
    version="1.0.0",
    description="A FastAPI service for finding near duplicate documents using TF-IDF and embedding models",
    lifespan=lifespan
)

# Include routes
app.include_router(router)


if __name__ == "__main__":
    print("🚀 Starting Near Duplicate Detection API...")
    print("📊 The API will be available at: http://localhost:8000")
    print("📖 Interactive docs available at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)