from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from PIL import Image
import io
import numpy as np
import config
import traceback
# Import all models
from models.oct_classifier import OCTClassifier
from models.fundus_classifier import FundusClassifier
from models.gradcam_unified import generate_gradcam_analysis

router = APIRouter()

# Initialize classifiers
oct_classifier = OCTClassifier()
fundus_classifier = FundusClassifier()

@router.post("/classify/oct")
async def classify_oct_image(file: UploadFile = File(...)):
    """
    Classify OCT image into 8 classes
    Classes: ['DR', 'NORMAL', 'DME', 'AMD', 'CNV', 'DRUSEN', 'MH', 'CSR']
    """
    try:
        # Validate file
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        if file.size and file.size > config.MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="File too large")
        
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Classify
        result = oct_classifier.predict(image)
        
        # Check if advanced analysis is available
        predicted_class = result["predicted_class"].lower()
        has_segmentation = predicted_class in config.SEGMENTATION_AVAILABLE
        has_gradcam = predicted_class in config.GRADCAM_AVAILABLE
        
        # Create clean response without tensors or models
        response_result = {
            "success": True,
            "predicted_class": result["predicted_class"],
            "predicted_class_index": result["predicted_class_index"], 
            "confidence": result["confidence"],
            "all_probabilities": result["all_probabilities"],
            "image_type": "OCT",
            "filename": file.filename,
            "classes_available": config.OCT_CLASSES,
            "advanced_analysis_available": {
                "segmentation": has_segmentation,
                "gradcam": has_gradcam
            }
        }
        
        # Add recommendations
        if has_segmentation:
            response_result["recommendation"] = f"Use /analyze/{predicted_class} for detailed analysis"
        elif has_gradcam:
            response_result["recommendation"] = f"Use /analyze/gradcam/{predicted_class} for attention analysis"
        
        return JSONResponse(content=response_result)
        
    except Exception as e:
        print(f"OCT classification error: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

@router.post("/classify/fundus")
async def classify_fundus_image(file: UploadFile = File(...)):
    """
    Classify Fundus image for diabetic retinopathy
    Classes: ['Mild_NPDR', 'Moderate_NPDR', 'No_DR', 'PDR', 'Severe_NPDR']
    """
    try:
        # Validate file
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        if file.size and file.size > config.MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="File too large")
        
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Classify
        result = fundus_classifier.predict(image)
        
        # Check if advanced analysis is available
        predicted_class = result["predicted_class"]
        has_advanced = predicted_class != 'No_DR'
        
        # Create clean response without tensors or models
        response_result = {
            "success": True,
            "predicted_class": result["predicted_class"],
            "predicted_class_index": result["predicted_class_index"],
            "confidence": result["confidence"],
            "all_probabilities": result["all_probabilities"],
            "status": result["status"],
            "message": result["message"],
            "severity": result["severity"],
            "image_type": "Fundus",
            "filename": file.filename,
            "classes_available": config.FUNDUS_CLASSES,
            "advanced_analysis_available": has_advanced
        }
        
        # Add recommendations
        if has_advanced:
            response_result["recommendation"] = "Use /analyze/diabetic-retinopathy for detailed DR analysis"
        
        return JSONResponse(content=response_result)
        
    except Exception as e:
        print(f"Fundus classification error: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

# Advanced Analysis Routes with Segmentation
@router.post("/analyze/dme")
async def analyze_dme(file: UploadFile = File(...)):
    """Advanced DME analysis with segmentation"""
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        from models.dme_segmentation import DMESegmentation
        dme_analyzer = DMESegmentation()
        result = await dme_analyzer.analyze(image, file.filename)
        
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DME analysis failed: {str(e)}")

@router.post("/analyze/drusen")
async def analyze_drusen(
    file: UploadFile = File(...), 
    confidence_threshold: float = Query(0.25, ge=0.0, le=1.0)
):
    """Advanced Drusen detection and analysis"""
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        from models.drusen_detection import DrusenDetection
        drusen_analyzer = DrusenDetection()
        result = await drusen_analyzer.analyze(image, file.filename, confidence_threshold)
        
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Drusen analysis failed: {str(e)}")

@router.post("/analyze/glaucoma")
async def analyze_glaucoma(
    file: UploadFile = File(...),
    use_roboflow: bool = Query(True)
):
    """Advanced Glaucoma analysis with CDR calculation"""
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        from models.glaucoma_analysis import GlaucomaAnalysis
        glaucoma_analyzer = GlaucomaAnalysis()
        result = await glaucoma_analyzer.analyze(image, file.filename, use_roboflow)
        
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Glaucoma analysis failed: {str(e)}")

@router.post("/analyze/retinitis-pigmentosa")
async def analyze_rp(
    file: UploadFile = File(...),
    confidence_threshold: float = Query(0.5, ge=0.0, le=1.0)
):
    """Advanced Retinitis Pigmentosa analysis"""
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        from models.rp_analysis import RPAnalysis
        rp_analyzer = RPAnalysis()
        result = await rp_analyzer.analyze(image, file.filename, confidence_threshold)
        
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RP analysis failed: {str(e)}")

@router.post("/analyze/diabetic-retinopathy")
async def analyze_diabetic_retinopathy(file: UploadFile = File(...)):
    """Advanced DR analysis with segmentation"""
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        from models.dr_segmentation import DRSegmentation
        dr_analyzer = DRSegmentation()
        result = await dr_analyzer.analyze(image, file.filename)
        
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DR analysis failed: {str(e)}")
@router.post("/classify/eye-disease")
async def classify_eye_disease(file: UploadFile = File(...)):
    """
    General eye disease classification
    Classes: ["cataract", "diabetic_retinopathy", "glaucoma", "normal", "retinitis_pigmentosa"]
    """
    try:
        # Validate file
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        if file.size and file.size > config.MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="File too large")
        
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Import and initialize eye disease classifier
        from models.eye_disease_classifier import EyeDiseaseClassifier
        eye_classifier = EyeDiseaseClassifier()
        
        # Classify
        result = eye_classifier.predict(image)
        
        # Check if advanced analysis is available
        predicted_class = result["predicted_class"].lower()
        has_segmentation = predicted_class in config.SEGMENTATION_AVAILABLE
        has_gradcam = predicted_class in config.GRADCAM_AVAILABLE
        
        # Remove non-serializable items for JSON response
        response_result = {
            "success": True,
            "predicted_class": result["predicted_class"],
            "predicted_class_index": result["predicted_class_index"],
            "confidence": result["confidence"],
            "all_probabilities": result["all_probabilities"],
            "image_type": "Eye Disease",
            "filename": file.filename,
            "classes_available": config.EYE_DISEASE_CLASSES,
            "advanced_analysis_available": {
                "segmentation": has_segmentation,
                "gradcam": has_gradcam or predicted_class != "normal"
            }
        }
        
        # Add recommendations
        if has_segmentation:
            endpoint_name = predicted_class.replace("_", "-")
            response_result["recommendation"] = f"Use /analyze/{endpoint_name} for detailed analysis"
        elif has_gradcam or predicted_class != "normal":
            response_result["recommendation"] = f"Use /analyze/gradcam/{predicted_class} for attention analysis"
        
        return JSONResponse(content=response_result)
        
    except Exception as e:
        print(f"Eye disease classification error: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")
# GradCAM Analysis Route (for conditions without segmentation)
@router.post("/analyze/gradcam/{condition}")
async def gradcam_analysis(
    condition: str,
    file: UploadFile = File(...),
    target_class: int = Query(None)
):
    """
    GradCAM analysis for conditions without segmentation models
    Available conditions: cataract, amd, cnv, mh, csr
    """
    try:
        # Validate condition
        if condition not in config.GRADCAM_AVAILABLE:
            raise HTTPException(
                status_code=400, 
                detail=f"GradCAM not available for {condition}. Available: {list(config.GRADCAM_AVAILABLE.keys())}"
            )
        
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        image_array = np.array(image)
        
        # Determine which classifier to use based on condition
        if condition in ['amd', 'cnv', 'mh', 'csr']:
            # Use OCT classifier
            result = oct_classifier.predict(image)
            model_type = "fastvit"
        elif condition == 'cataract':
            # Use eye disease classifier
            from models.eye_disease_classifier import EyeDiseaseClassifier
            eye_classifier = EyeDiseaseClassifier()
            result = eye_classifier.predict(image)
            model_type = "vit"
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported condition: {condition}")
        
        # Use provided target class or predicted class
        if target_class is None:
            target_class = result["predicted_class_index"]
        
        # Generate GradCAM
        gradcam_result = generate_gradcam_analysis(
            model=result["model"],
            model_type=model_type,
            image_tensor=result["image_tensor"],
            original_image=image_array,
            condition=condition,
            predicted_class_idx=target_class,
            additional_inputs=result.get("glcm_tensor")  # For fundus classifier
        )
        
        return JSONResponse(content={
            "success": True,
            "condition": condition,
            "predicted_class": result["predicted_class"],
            "confidence": result["confidence"],
            "target_class_index": target_class,
            "gradcam_analysis": gradcam_result,
            "filename": file.filename
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GradCAM analysis failed: {str(e)}")