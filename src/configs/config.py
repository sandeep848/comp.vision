import os
from pathlib import Path

# Environment Detection
IN_COLAB = False
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# Base Paths (Relative to script location or overridden via environment variables)
BASE_DIR = Path(__file__).resolve().parent.parent.parent

if os.getenv("PROJECT_ROOT"):
    PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT"))
elif IN_COLAB:
    PROJECT_ROOT = Path("/content/drive/MyDrive/deepfake_robustness")
else:
    PROJECT_ROOT = BASE_DIR / "deepfake_robustness"

if os.getenv("DATASET_ROOT"):
    DATASET_ROOT = Path(os.getenv("DATASET_ROOT"))
elif IN_COLAB:
    DATASET_ROOT = Path("/content/drive/MyDrive/datasets/FaceForensics")
else:
    DATASET_ROOT = BASE_DIR / "datasets" / "FaceForensics"

PROCESSED_ROOT = PROJECT_ROOT / "processed_faces"
FFPP_FACE_ROOT = PROCESSED_ROOT / "ffpp_c23"          # Primary training dataset (C23 compression)

if os.getenv("OUTPUT_ROOT"):
    OUTPUT_ROOT = Path(os.getenv("OUTPUT_ROOT"))
else:
    OUTPUT_ROOT = PROJECT_ROOT / "outputs"

MANIFEST_PATH = PROJECT_ROOT / "ffpp_manifest.csv"

# Global Random Seed
SEED = 42

# Training & Split Hyperparameters
IMAGE_SIZE = 256  # 256x256 high-resolution crop for sharp SRM frequency traces
BATCH_SIZE = 32  # Per-GPU micro batch size
GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch size = 32 * 4 = 128
NUM_EPOCHS = 30  # 30 epochs for full convergence
BACKBONE_LR = 4e-5  # Reduced learning rate for pretrained backbone to prevent unlearning
CLASSIFIER_LR = 2e-4  # Scaled learning rate for classifier head
WEIGHT_DECAY = 1e-2  # Increased weight decay for regularization
PATIENCE = 5  # Early stopping patience
NUM_WORKERS = 4  # Dataloader workers

# Data Composition & Split Ratios
TRAIN_RATIO = 0.70  # 70% Training
VAL_RATIO = 0.15    # 15% Validation
TEST_RATIO = 0.15   # 15% Testing

# Face Extraction Configuration
FRAME_INTERVAL = 5
MAX_FRAMES_PER_VIDEO = 60
MIN_FACE_PROBABILITY = 0.90
FACE_MARGIN_PERCENT = 0.10
FORCE_REEXTRACT = False
ALIGN_FACES = True
# Caps the raw per-category video LISTING read from disk, before real/fake balancing.
MAX_CANDIDATE_VIDEOS_PER_CATEGORY = None
# Caps how many already-BALANCED/selected videos per manipulation group are actually sent
# to face extraction (applied after MAX_CANDIDATE_VIDEOS_PER_CATEGORY and balancing).
MAX_SELECTED_VIDEOS_PER_CATEGORY = None

# Data Paths (Celeb-DF External Dataset)
CELEBDF_ROOT = BASE_DIR / "datasets" / "Celeb-DF-v2"
CELEBDF_MANIFEST_PATH = PROJECT_ROOT / "celebdf_manifest.csv"

# Official Celeb-DF Download Links
CELEBDF_V2_GDRIVE_URL = "https://drive.google.com/open?id=1iLx76wsbi9itnkxSqz9BVBl4ZvnbIazj"
CELEBDF_V2_BAIDU_URL = "https://pan.baidu.com/s/1EcYX0s4U3kbI1V2vdrP46A"  # Passcode: yxa1

# Model Configuration
MODEL_NAME = "efficientnet_b0"  # Lightweight CNN backbone
MODEL_VARIANT = "fusion"  # Options: 'fusion', 'rgb_only', 'fusion_no_attn'
PRETRAINED = True
STOCHASTIC_DEPTH_PROB = 0.30
BRANCH_MODE = "fusion"  # Options: 'rgb', 'freq', 'fusion'

# Training Strategy
TRAINING_STRATEGY = "clean"  # Options: 'clean', 'degradation'

# Degradation Evaluation Matrix Settings
DEGRADATION_TIERS = {
    "clean": {"jpeg_quality": None, "resize_scale": 1.0, "blur_sigma": None, "noise_std": None},
    "weak_compression": {"jpeg_quality": 90, "resize_scale": 1.0},
    "medium_compression": {"jpeg_quality": 70, "resize_scale": 1.0},
    "strong_compression": {"jpeg_quality": 45, "resize_scale": 1.0},
    "extreme_compression": {"jpeg_quality": 30, "resize_scale": 1.0},
    "resize_75": {"jpeg_quality": None, "resize_scale": 0.75},
    "resize_50": {"jpeg_quality": None, "resize_scale": 0.50},
    "resize_25": {"jpeg_quality": None, "resize_scale": 0.25},
    "gaussian_blur": {"blur_sigma": 1.5},
    "motion_blur": {"motion_blur_size": 5},
    "gaussian_noise": {"noise_std": 6.0},
    "color_jitter": {"color_jitter": 0.2},
    "resize_50_compress_70": {"jpeg_quality": 70, "resize_scale": 0.50},
    # Note: the actual per-tier operation order (resize -> blur -> motion-blur -> color-jitter
    # -> noise -> JPEG) is fixed by apply_advanced_tier_distortion (evaluate.py); these tiers
    # apply resize + JPEG (and, for social_media_pipeline, motion blur) in that fixed order.
    "screenshot_recompress": {"resize_scale": 0.70, "jpeg_quality": 60},
    "social_media_pipeline": {"resize_scale": 0.50, "jpeg_quality": 50, "motion_blur_size": 3},
}

# Preprocessing Thresholds
SAMPLING_STRATEGY = "spaced"
BLUR_THRESHOLD = 25.0
MIN_FACE_SIZE = 60

# Loss & Balancing Configuration
BALANCING_STRATEGY = "focal_loss"
CB_BETA = 0.999
FOCAL_ALPHA = 0.50
FOCAL_GAMMA = 1.5

# Forensic Augmentations
AUG_JPEG_PROB = 0.70
AUG_JPEG_QUALITY_MIN = 50
AUG_JPEG_QUALITY_MAX = 95
AUG_DOWNSCALE_PROB = 0.50
AUG_DOWNSCALE_SCALES = [0.50, 0.75, 1.00]
AUG_BLUR_PROB = 0.50
AUG_BLUR_SIGMA_MIN = 0.1
AUG_BLUR_SIGMA_MAX = 1.2
AUG_MOTION_BLUR_PROB = 0.30
AUG_MOTION_BLUR_SIZES = [3, 5]
AUG_SHARPEN_PROB = 0.20
AUG_SHARPEN_FACTOR_MIN = 1.1
AUG_SHARPEN_FACTOR_MAX = 1.4
AUG_COLOR_JITTER_PROB = 0.4
AUG_COLOR_JITTER_BRIGHTNESS = 0.1
AUG_COLOR_JITTER_CONTRAST = 0.1
AUG_COLOR_JITTER_SATURATION = 0.1
AUG_NOISE_PROB = 0.40
AUG_NOISE_STD_MIN = 1.0
AUG_NOISE_STD_MAX = 4.0

# Regularization & Optimization Options
FREEZE_PERCENT = 0.50
PROGRESSIVE_UNFREEZE = True
DROPOUT = 0.50
LABEL_SMOOTHING = 0.05
USE_MIXED_PRECISION = True

# Degradation Curriculum Configuration
CURRICULUM_ENABLED = True
CURRICULUM_SEVERITY_START = 0.3
CURRICULUM_RAMP_EPOCHS = 10
GRADIENT_CLIPPING = 1.0
SCHEDULER = "cosine"
WARMUP_EPOCHS = 2
EMA_DECAY = 0.9999

# Mixup Configuration
USE_MIXUP = True
MIXUP_PROB = 0.50
MIXUP_ALPHA = 0.8

# Post-Training Checkpoint Averaging Configuration
NUM_CHECKPOINTS_TO_AVERAGE = 3
TENSORBOARD_DIR = OUTPUT_ROOT / "runs"

def ensure_directories():
    """Ensure output and data directories exist."""
    for folder in [PROJECT_ROOT, DATASET_ROOT, PROCESSED_ROOT, FFPP_FACE_ROOT, OUTPUT_ROOT, TENSORBOARD_DIR]:
        folder.mkdir(parents=True, exist_ok=True)


