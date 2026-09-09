import io
import random
import numpy as np
import cv2
from PIL import Image, ImageEnhance

from torchvision import transforms
from torchvision.transforms import InterpolationMode

from src.configs import config

class RandomJPEGCompression:
    def __init__(self, quality_range=(50, 95), probability=0.75):
        self.quality_range = quality_range
        self.probability = probability

    def __call__(self, image):
        if random.random() >= self.probability:
            return image

        quality = random.randint(self.quality_range[0], self.quality_range[1])
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        compressed_image = Image.open(buffer).convert("RGB").copy()
        return compressed_image


class RandomDownscaleRestore:
    def __init__(self, scales=(0.50, 0.75, 1.00), probability=0.75):
        self.scales = scales
        self.probability = probability

    def __call__(self, image):
        if random.random() >= self.probability:
            return image

        width, height = image.size
        scale = random.choice(self.scales)
        if scale >= 1.0:
            return image

        small_width = max(16, int(width * scale))
        small_height = max(16, int(height * scale))

        image = image.resize((small_width, small_height), Image.Resampling.BILINEAR)
        image = image.resize((width, height), Image.Resampling.BILINEAR)
        return image


class RandomSharpen:
    def __init__(self, factor_range=(1.1, 1.5), probability=0.2):
        self.factor_range = factor_range
        self.probability = probability

    def __call__(self, image):
        if random.random() >= self.probability:
            return image
        factor = random.uniform(self.factor_range[0], self.factor_range[1])
        enhancer = ImageEnhance.Sharpness(image)
        return enhancer.enhance(factor)


class RandomGaussianNoise:
    def __init__(self, std_range=(1.0, 4.0), probability=0.1):
        self.std_range = std_range
        self.probability = probability

    def __call__(self, image):
        if random.random() >= self.probability:
            return image
        std = random.uniform(self.std_range[0], self.std_range[1])
        img_np = np.array(image).astype(float)
        noise = np.random.normal(0, std, img_np.shape)
        noisy = np.clip(img_np + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy)


class RandomMotionBlur:
    def __init__(self, sizes=(3, 5), probability=0.15):
        self.sizes = sizes
        self.probability = probability

    def __call__(self, image):
        if random.random() >= self.probability:
            return image
        size = random.choice(self.sizes)
        img_np = np.array(image)
        kernel = np.zeros((size, size))
        if random.random() > 0.5:
            kernel[int((size - 1) / 2), :] = np.ones(size)
        else:
            kernel[:, int((size - 1) / 2)] = np.ones(size)
        kernel /= size
        blurred = cv2.filter2D(img_np, -1, kernel)
        return Image.fromarray(blurred)


def get_transforms():
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    clean_transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    standard_transform = transforms.Compose([
        transforms.RandomResizedCrop(config.IMAGE_SIZE, scale=(0.85, 1.0), ratio=(0.95, 1.05), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
        ], p=getattr(config, "AUG_BLUR_PROB", 0.20)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    degradation_transform = _build_degradation_transform(severity_scale=1.0)

    evaluation_transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    training_transforms = {
        "clean": clean_transform,
        "standard": standard_transform,
        "degradation": degradation_transform,
    }

    if config.TRAINING_STRATEGY not in training_transforms:
        raise ValueError(f"Invalid TRAINING_STRATEGY: {config.TRAINING_STRATEGY}")

    return training_transforms[config.TRAINING_STRATEGY], evaluation_transform


class RandomDegradationChoice:
    """Randomly selects 1 (or at most 2) degradation operations per image sample

    weighted by configuration degradation probabilities to prevent compounding
    destructive transformations simultaneously, which obliterates high-frequency forensic signals.
    """
    def __init__(self, degradation_ops, weights=None, p_clean=0.40):
        self.degradation_ops = degradation_ops
        self.weights = weights
        self.p_clean = p_clean

    def __call__(self, image):
        if random.random() < self.p_clean:
            return image

        num_ops = 2 if random.random() < 0.15 else 1
        if self.weights is not None and len(self.weights) == len(self.degradation_ops):
            # Weighted random sampling without replacement
            probs = np.array(self.weights, dtype=np.float64)
            probs = probs / np.sum(probs)
            chosen_indices = np.random.choice(len(self.degradation_ops), size=min(num_ops, len(self.degradation_ops)), replace=False, p=probs)
            chosen_ops = [self.degradation_ops[i] for i in chosen_indices]
        else:
            chosen_ops = random.sample(self.degradation_ops, k=min(num_ops, len(self.degradation_ops)))

        for op in chosen_ops:
            image = op(image)
        return image


def _build_degradation_transform(severity_scale: float = 1.0):
    """Build the forensic-preserving degradation transform with a given severity_scale in [0, 1]."""
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    s = float(severity_scale)

    jpeg_quality_min = int(config.AUG_JPEG_QUALITY_MIN + (1.0 - s) * (95 - config.AUG_JPEG_QUALITY_MIN))
    jpeg_quality_max = config.AUG_JPEG_QUALITY_MAX

    all_scales = list(config.AUG_DOWNSCALE_SCALES)
    curriculum_scales = [sc for sc in all_scales if sc >= (0.75 - 0.25 * s)] or all_scales

    noise_std_min = config.AUG_NOISE_STD_MIN * s
    noise_std_max = config.AUG_NOISE_STD_MAX * s
    if noise_std_min < 0.1:
        noise_std_min = 0.1
    if noise_std_max < noise_std_min:
        noise_std_max = noise_std_min

    degradation_ops = [
        RandomJPEGCompression(quality_range=(jpeg_quality_min, jpeg_quality_max), probability=1.0),
        RandomDownscaleRestore(scales=curriculum_scales, probability=1.0),
        RandomMotionBlur(sizes=config.AUG_MOTION_BLUR_SIZES, probability=1.0),
        transforms.GaussianBlur(kernel_size=3, sigma=(config.AUG_BLUR_SIGMA_MIN, max(config.AUG_BLUR_SIGMA_MIN + 0.1, config.AUG_BLUR_SIGMA_MAX * s))),
        RandomGaussianNoise(std_range=(noise_std_min, noise_std_max), probability=1.0),
        transforms.ColorJitter(
            brightness=config.AUG_COLOR_JITTER_BRIGHTNESS * s,
            contrast=config.AUG_COLOR_JITTER_CONTRAST * s,
            saturation=config.AUG_COLOR_JITTER_SATURATION * s,
        ),
        RandomSharpen(factor_range=(getattr(config, "AUG_SHARPEN_FACTOR_MIN", 1.1), getattr(config, "AUG_SHARPEN_FACTOR_MAX", 2.0)), probability=1.0),
    ]

    weights = [
        getattr(config, "AUG_JPEG_PROB", 0.35),
        getattr(config, "AUG_DOWNSCALE_PROB", 0.25),
        getattr(config, "AUG_MOTION_BLUR_PROB", 0.15),
        getattr(config, "AUG_BLUR_PROB", 0.10),
        getattr(config, "AUG_NOISE_PROB", 0.10),
        getattr(config, "AUG_COLOR_JITTER_PROB", 0.05),
        getattr(config, "AUG_SHARPEN_PROB", 0.05),
    ]

    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=3, translate=(0.02, 0.02), scale=(0.98, 1.02), interpolation=InterpolationMode.BILINEAR),
        RandomDegradationChoice(degradation_ops, weights=weights, p_clean=max(0.25, 0.50 - 0.25 * s)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_degradation_transform_for_epoch(epoch: int, num_epochs: int) -> "transforms.Compose":
    """Return a degradation transform with curriculum-scaled severity for the given epoch."""
    if (
        not getattr(config, "CURRICULUM_ENABLED", False)
        or config.TRAINING_STRATEGY != "degradation"
    ):
        return _build_degradation_transform(severity_scale=1.0)

    ramp_epochs = getattr(config, "CURRICULUM_RAMP_EPOCHS", max(1, num_epochs // 2))
    severity_start = getattr(config, "CURRICULUM_SEVERITY_START", 0.3)

    if epoch <= ramp_epochs:
        frac = (epoch - 1) / max(1, ramp_epochs - 1)
        severity = severity_start + frac * (1.0 - severity_start)
    else:
        severity = 1.0

    return _build_degradation_transform(severity_scale=severity)
