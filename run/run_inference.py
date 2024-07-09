from pathlib import Path
import sys
from PIL import Image
import argparse
from utils_ootd import get_mask_location

# Ensure basicsr is installed
try:
    import basicsr
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "basicsr"])

# Set project root and import necessary modules
PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from preprocess.openpose.run_openpose import OpenPose
from preprocess.humanparsing.run_parsing import Parsing
from ootd.inference_ootd_hd import OOTDiffusionHD
from ootd.inference_ootd_dc import OOTDiffusionDC

# Argument parser for command line arguments
parser = argparse.ArgumentParser(description='Run OOTD Inference')
parser.add_argument('--gpu_id', '-g', type=int, default=0, required=False, help='GPU ID to use')
parser.add_argument('--model_path', type=str, required=True, help='Path to the model image')
parser.add_argument('--cloth_path', type=str, required=True, help='Path to the cloth image')
parser.add_argument('--model_type', type=str, default="hd", required=False, help='Type of model: "hd" or "dc"')
parser.add_argument('--category', '-c', type=int, default=0, required=False, help='Category: 0 for upperbody, 1 for lowerbody, 2 for dress')
parser.add_argument('--scale', type=float, default=2.0, required=False, help='Image scaling factor')
parser.add_argument('--step', type=int, default=20, required=False, help='Number of diffusion steps')
parser.add_argument('--sample', type=int, default=4, required=False, help='Number of samples to generate')
parser.add_argument('--seed', type=int, default=-1, required=False, help='Random seed for reproducibility')
args = parser.parse_args()

# Initialize OpenPose and Parsing models
openpose_model = OpenPose(args.gpu_id)
parsing_model = Parsing(args.gpu_id)

# Category and model type validation
category_dict = ['upperbody', 'lowerbody', 'dress']
category_dict_utils = ['upper_body', 'lower_body', 'dresses']

model_type = args.model_type
category = args.category
cloth_path = args.cloth_path
model_path = args.model_path

image_scale = args.scale
n_steps = args.step
n_samples = args.sample
seed = args.seed

if model_type == "hd":
    model = OOTDiffusionHD(args.gpu_id)
elif model_type == "dc":
    model = OOTDiffusionDC(args.gpu_id)
else:
    raise ValueError("model_type must be 'hd' or 'dc'!")

# Main inference function
if __name__ == '__main__':
    if model_type == 'hd' and category != 0:
        raise ValueError("model_type 'hd' requires category == 0 (upperbody)!")

    # Load and preprocess images
    cloth_img = Image.open(cloth_path).resize((768, 1024))
    model_img = Image.open(model_path).resize((768, 1024))
    keypoints = openpose_model(model_img.resize((384, 512)))
    model_parse, _ = parsing_model(model_img.resize((384, 512)))
    
    mask, mask_gray = get_mask_location(model_type, category_dict_utils[category], model_parse, keypoints)
    print(mask)
    mask = mask.resize((768, 1024), Image.NEAREST)
    mask_gray = mask_gray.resize((768, 1024), Image.NEAREST)

    masked_vton_img = Image.composite(mask_gray, model_img, mask)
    print(masked_vton_img)
    masked_vton_img.save('./images_output/mask.jpg')

    # Run OOTDiffusion model
    images = model(
        model_type=model_type,
        category=category_dict[category],
        image_garm=cloth_img,
        image_vton=masked_vton_img,
        mask=mask,
        image_ori=model_img,
        num_samples=n_samples,
        num_steps=n_steps,
        image_scale=image_scale,
        seed=seed,
    )

    # Save output images
    output_dir = Path('./images_output')
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, image in enumerate(images):
        image.save(output_dir / f'out_{model_type}_{idx}.png')
