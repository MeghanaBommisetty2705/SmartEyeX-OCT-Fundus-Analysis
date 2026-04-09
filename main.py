from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import config
from routes import router

# Initialize FastAPI app
app = FastAPI(
    title="Medical Image Analysis API",
    description="OCT and Fundus Image Classification & Analysis",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")

# Include all routes
app.include_router(router)

@app.get("/")
async def root():
    return {
        "message": "Medical Image Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "oct_classification": "/classify/oct",
            "fundus_classification": "/classify/fundus", 
            "eye_disease_classification": "/classify/eye-disease",
            "advanced_analysis": "/analyze/{condition}",
            "gradcam_analysis": "/analyze/gradcam/{condition}",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "device": str(config.DEVICE),
        "available_classifications": {
            "oct_classes": len(config.OCT_CLASSES),
            "fundus_classes": len(config.FUNDUS_CLASSES),
            "eye_disease_classes": len(config.EYE_DISEASE_CLASSES)
        },
        "available_segmentation": list(config.SEGMENTATION_AVAILABLE.keys()),
        "available_gradcam": list(config.GRADCAM_AVAILABLE.keys())
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)