"""
Vibe DevOps Backend API
FastAPI application for GCP resource discovery and task management
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Create FastAPI app
app = FastAPI(
    title="Vibe DevOps API",
    description="Backend API for GCP resource discovery and task management",
    version="1.0.0"
)

# Configure CORS
frontend_url = os.getenv("FRONTEND_URL", "http://localhost:3000")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_url, "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "vibe-devops-api",
        "version": "1.0.0"
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Vibe DevOps API",
        "docs": "/docs",
        "health": "/health"
    }

# Import and include routers
from api.routes import gcp
app.include_router(gcp.router, prefix="/api/gcp", tags=["GCP"])

# Tasks router will be added later
# from api.routes import tasks
# app.include_router(tasks.router, prefix="/api/tasks", tags=["Tasks"])

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    debug = os.getenv("DEBUG", "True").lower() == "true"
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=debug
    )
