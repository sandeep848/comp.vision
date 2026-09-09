from deepfake_robustness.configs import config
from torchvision import transforms
from .transforms import RandomJPEGCompression, RandomDownscaleRestore, RandomMotionBlur, RandomGaussianNoise, RandomSharpen

def get_operator(op_name):
    if op_name == "jpeg":
        return RandomJPEGCompression(quality_range=(50, 95), probability=1.0)
    elif op_name == "downscale":
        return RandomDownscaleRestore(scales=(0.50, 0.75), probability=1.0)
    elif op_name == "motion_blur":
        return RandomMotionBlur(sizes=(3, 5), probability=1.0)
    elif op_name == "gaussian_blur":
        return transforms.GaussianBlur(kernel_size=3, sigma=(0.5, 1.5))
    elif op_name == "gaussian_noise":
        return RandomGaussianNoise(std_range=(2.0, 4.0), probability=1.0)
    elif op_name == "sharpen":
        return RandomSharpen(factor_range=(1.2, 1.8), probability=1.0)
    else:
        raise ValueError(f"Unknown operator {op_name}")

def build_pipeline(pipeline_id):
    if pipeline_id == "clean" or not pipeline_id:
        return []
    
    if "->" in pipeline_id:
        ops = pipeline_id.split("->")
        return [get_operator(op) for op in ops]
    else:
        return [get_operator(pipeline_id)]

