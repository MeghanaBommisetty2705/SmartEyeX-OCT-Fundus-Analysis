import torch
import os

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
WEIGHTS_DIR = "weights"
STATIC_DIR = "static"
TEMP_DIR = "temp"

# Create directories
os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
DEVICE='cpu'
# OCT Classification (8 classes)
OCT_CLASSES = ['DR', 'NORMAL', 'DME', 'AMD', 'CNV', 'DRUSEN', 'MH', 'CSR']
OCT_MODEL_PATH = os.path.join(WEIGHTS_DIR, "fastvit_oct_best.pth")
OCT_MODEL_NAME = "fastvit_t8"

# Fundus Classification (5 classes)
FUNDUS_CLASSES = ['Mild_NPDR', 'Moderate_NPDR', 'No_DR', 'PDR', 'Severe_NPDR']
FUNDUS_MODEL_PATH = os.path.join(WEIGHTS_DIR, "best_densenet_glcm_model.pth")

# Eye Disease Classification (5 classes)
EYE_DISEASE_CLASSES = ["cataract", "diabetic_retinopathy", "glaucoma", "normal", "retinitis_pigmentosa"]
EYE_DISEASE_MODEL_PATH = os.path.join(WEIGHTS_DIR, "best_model.pth")

# Segmentation Model Paths
DME_MODEL_PATH = os.path.join(WEIGHTS_DIR, "oct_complete_model.pth")
DRUSEN_MODEL_PATH = os.path.join(WEIGHTS_DIR, "drusen.pt")
GLAUCOMA_MODEL_PATH = os.path.join(WEIGHTS_DIR, "resnet_unet_optic_disc_cup_fixed.pth")
RP_MODEL_PATH = os.path.join(WEIGHTS_DIR, "best_ultra_stable_rps_net.pth")
DR_SEGMENTATION_PATH = os.path.join(WEIGHTS_DIR, "best.pt")

# Image processing settings
IMAGE_SIZE = 224
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Classes that have segmentation models
SEGMENTATION_AVAILABLE = {
    # OCT conditions
    'dme': True,
    'drusen': True,
    
    # Fundus conditions  
    'diabetic_retinopathy': True,
    'glaucoma': True,
    'retinitis_pigmentosa': True
}

# Classes that use GradCAM (no segmentation)
GRADCAM_AVAILABLE = {
    # OCT conditions
    'amd': True,
    'cnv': True,
    'mh': True,
    'csr': True,
    
    # Eye disease conditions
    'cataract': True
}

# Roboflow API (optional)
ROBOFLOW_API_KEY = "Gqf1hrF7jdAh8EsbOoTM"  # Replace with your key