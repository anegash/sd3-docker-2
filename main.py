import torch
import logging
import traceback
import os
import shutil
import time
import io
import boto3
import torch.optim as optim
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException, BackgroundTasks
from diffusers import StableDiffusion3Pipeline, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from peft import get_peft_model, LoraConfig, PeftModel
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from pydantic import BaseModel

# Initialize logging
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(module)s | %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI()

# AWS S3 Configuration (Pulled from Runpod Environment Variables)
S3_BUCKET = "little-legends-dev"
S3_TRAINING_PATH = "training_data/"
S3_WEIGHTS_PATH = "lora_models/"
# Model Paths
model_id = "stabilityai/stable-diffusion-3.5-large"
model_dir = "/workspace/models"
model_path = os.path.join(model_dir, model_id.replace("/", "_"))
lora_model_path = "/workspace/lora_models"
os.makedirs(lora_model_path, exist_ok=True)


# Load AWS credentials from environment variables
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")  # Default to us-east-1 if not set

# Initialize S3 Client with credentials
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)

# Initialize model
try:
    logger.info("🔥 Initializing Stable Diffusion API...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_path, torch_dtype=torch.float16, variant="fp16"
    )

    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
        logger.info("🚀 Running on CUDA")
    else:
        pipe = pipe.to("cpu")
        logger.warning("⚠️ Running on CPU (slow performance).")

except Exception as e:
    logger.error("🔥 Error loading Stable Diffusion model: %s", str(e))
    logger.error(traceback.format_exc())
    pipe = None  # Prevent API from running with an unloaded model

# Helper function to download images from S3
def download_images_from_s3(local_path: str, subfolder: str):
    """
    Downloads images from S3 stored in a specific child's subfolder.

    :param local_path: Local directory where images will be saved.
    :param subfolder: The specific child's subfolder in S3.
    """
    # Construct full S3 path with the child's subfolder
    s3_folder_path = os.path.join(S3_TRAINING_PATH, subfolder).replace("\\", "/")

    os.makedirs(local_path, exist_ok=True)
    objects = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=s3_folder_path)

    if "Contents" not in objects:
        raise HTTPException(status_code=404, detail=f"No images found in S3 for {subfolder}.")

    for obj in objects["Contents"]:
        file_name = obj["Key"].split("/")[-1]
        local_file_path = os.path.join(local_path, file_name)

        if file_name:  # Ensure it's a file, not just a directory
            s3_client.download_file(S3_BUCKET, obj["Key"], local_file_path)
            logging.info(f"✅ Downloaded {obj['Key']} to {local_file_path}")

# Custom dataset for training
class ImageDataset(Dataset):
    def __init__(self, image_folder):
        self.image_folder = image_folder
        self.image_files = [f for f in os.listdir(image_folder) if f.endswith((".png", ".jpg", ".jpeg"))]
        self.tokenizer = CLIPTokenizer.from_pretrained(model_path)
        self.captions = ["A portrait of a child."] * len(self.image_files)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, self.image_files[idx])
        image = Image.open(img_path).convert("RGB").resize((512, 512))
        image = torch.tensor(torchvision.transforms.ToTensor()(image))

        inputs = self.tokenizer(self.captions[idx], return_tensors="pt", padding=True)
        return image, inputs["input_ids"]

# LoRA training function
def train_lora(dataset_path, output_lora_path, steps=1000, lr=1e-4):
    logging.info("🚀 Starting LoRA fine-tuning...")
    
    # Load base model components
    unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet")
    text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder")

    # Load dataset
    dataset = ImageDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Apply LoRA
    config = LoraConfig(
        r=4,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["to_q", "to_v"]
    )
    unet = get_peft_model(unet, config)
    optimizer = optim.AdamW(unet.parameters(), lr=lr)

    # Training loop
    unet.train()
    for step, (images, captions) in enumerate(dataloader):
        if step >= steps:
            break

        optimizer.zero_grad()
        text_embeddings = text_encoder(captions).last_hidden_state
        loss = F.mse_loss(unet(images), text_embeddings)
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            logging.info(f"🔄 Step {step}/{steps}: Loss = {loss.item()}")

    # Save LoRA weights
    os.makedirs(output_lora_path, exist_ok=True)
    unet.save_pretrained(output_lora_path)
    logging.info("✅ LoRA fine-tuning complete!")

# Upload LoRA model to S3
def upload_lora_to_s3(local_path, s3_path):
    for file in os.listdir(local_path):
        local_file = os.path.join(local_path, file)
        s3_file = os.path.join(s3_path, file)
        s3_client.upload_file(local_file, S3_BUCKET, s3_file)
        logging.info(f"📤 Uploaded {local_file} to S3 at {s3_file}")

# API Request Models
class TrainRequest(BaseModel):
    subfolder: str  # The child's subfolder in S3
    output_lora_name: str  # Name of the trained LoRA model
    steps: int = 1000
    lr: float = 1e-4

@app.post("/train_lora")
async def train_lora_endpoint(request: TrainRequest, background_tasks: BackgroundTasks):
    local_dataset_path = os.path.join("/workspace/training_data", request.subfolder)
    local_lora_path = os.path.join(lora_model_path, request.output_lora_name)

    # Download dataset from S3
    try:
        logging.info(f"📥 Downloading images for {request.subfolder} from S3...")
        download_images_from_s3(local_dataset_path, request.subfolder)
    except Exception as e:
        logging.error(f"❌ Error downloading images: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to download images.")

    # Train LoRA in the background
    background_tasks.add_task(train_lora, local_dataset_path, local_lora_path, request.steps, request.lr)

    # Upload weights to S3 after training
    background_tasks.add_task(upload_lora_to_s3, local_lora_path, os.path.join(S3_WEIGHTS_PATH, request.output_lora_name))

    return {"message": f"LoRA training started for {request.subfolder}!", "output_path": local_lora_path}
# API Endpoint to load LoRA
@app.post("/load_lora")
async def load_lora(lora_model_name: str):
    lora_path = os.path.join(lora_model_path, lora_model_name)

    if not os.path.exists(lora_path):
        raise HTTPException(status_code=404, detail="LoRA model not found.")

    global pipe
    try:
        pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path)
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
        logging.info(f"✅ Loaded LoRA model: {lora_model_name}")
        return {"message": "LoRA model loaded successfully", "model": lora_model_name}

    except Exception as e:
        logging.error(f"❌ Error loading LoRA model: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to load LoRA model.")

# API Health Check
@app.get("/")
def home():
    logger.info("🛠️ Health check requested.")
    return {"message": "Stable Diffusion 3.5 API is running with LoRA support!"}