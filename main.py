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

import torchvision.transforms as transforms

from huggingface_hub import snapshot_download


# Initialize logging
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(module)s | %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

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


# Initialize FastAPI
app = FastAPI()

# AWS S3 Configuration (Pulled from Runpod Environment Variables)
S3_BUCKET = "little-legends-dev"
S3_TRAINING_PATH = "training_data/"
S3_WEIGHTS_PATH = "lora_models/"
# Model Paths
model_id = "stabilityai/stable-diffusion-3.5-large"
model_dir = "/workspace/models"
lora_model_path = "/workspace/lora_models"
os.makedirs(lora_model_path, exist_ok=True)



# Updated main.py snippet using persistent storage
@app.on_event("startup")
def load_model():
    global pipe, model_dir  # Add 'model_dir' to the global scope
    HF_TOKEN = os.getenv("HF_TOKEN")
    if not HF_TOKEN:
        raise ValueError("HuggingFace token not found! Set HF_TOKEN environment variable.")

    # Use global model_dir correctly
    model_dir = snapshot_download(
        repo_id=model_id,
        local_dir=model_dir,  # Persistent volume mount point
        token=HF_TOKEN,
        local_dir_use_symlinks=False,
        resume_download=True
    )

    print(f"✅ Model downloaded to: {model_dir}")

    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, use_safetensors=True
    )
    pipe.enable_xformers_memory_efficient_attention()
    pipe.to("cuda")
    
    print("🔥 Model loaded successfully!")


    
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
        self.tokenizer = CLIPTokenizer.from_pretrained(model_dir)
        self.captions = ["A portrait of a child."] * len(self.image_files)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, self.image_files[idx])
        image = Image.open(img_path).convert("RGB").resize((512, 512))
        image = transforms.ToTensor()(image)

        inputs = self.tokenizer(self.captions[idx], return_tensors="pt", padding=True)
        return image, inputs["input_ids"]

# LoRA training function
def train_lora(dataset_path, output_lora_path, steps=1000, lr=1e-4):
    logging.info("🚀 Starting LoRA fine-tuning...")
    
    # Load base model components
    unet = UNet2DConditionModel.from_pretrained(model_dir, subfolder="unet")
    text_encoder = CLIPTextModel.from_pretrained(model_dir, subfolder="text_encoder")

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
    unet.save_pretrained(output_lora_path, torch_dtype=torch.float16)
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
    """
    Starts a LoRA fine-tuning job in the background.

    - Downloads dataset from S3.
    - Starts the training job in the background.
    - Provides real-time logging at each step.
    """

    logging.info(f"🚀 Received training request for LoRA model: {request.output_lora_name}")
    logging.info(f"📂 Subfolder in S3: {request.subfolder}")
    logging.info(f"📌 Training Steps: {request.steps}, Learning Rate: {request.lr}")

    # Define local paths
    local_dataset_path = os.path.join("/workspace/training_data", request.subfolder)
    local_lora_path = os.path.join(lora_model_path, request.output_lora_name)

    logging.info(f"🛠️ Local dataset path: {local_dataset_path}")
    logging.info(f"🛠️ Local LoRA weights path: {local_lora_path}")

    # Step 1: Download dataset from S3
    try:
        logging.info(f"📥 Attempting to download dataset from S3 (Bucket: {S3_BUCKET}, Path: {S3_TRAINING_PATH}{request.subfolder})...")
        download_images_from_s3(local_dataset_path, request.subfolder)
        logging.info(f"✅ Dataset downloaded successfully to {local_dataset_path}")
    except Exception as e:
        logging.error(f"❌ ERROR: Failed to download dataset from S3: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to download images.")

    # Step 2: Start training LoRA in the background
    try:
        logging.info(f"🛠️ Starting LoRA fine-tuning for {request.output_lora_name}...")
        background_tasks.add_task(train_lora, local_dataset_path, local_lora_path, request.steps, request.lr)
        logging.info(f"✅ Training process started in the background for {request.output_lora_name}")
    except Exception as e:
        logging.error(f"❌ ERROR: Failed to start training: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to start LoRA training.")

    return {
        "message": f"LoRA training started for {request.subfolder}!",
        "output_path": local_lora_path,
        "s3_subfolder": request.subfolder,
        "training_steps": request.steps,
        "learning_rate": request.lr
    }



def download_lora_from_s3(lora_folder: str):
    """
    Downloads LoRA weights from S3 if they are missing locally.

    :param lora_folder: The LoRA folder name in S3.
    """
    local_lora_path = os.path.join(lora_model_path, lora_folder)
    
    if os.path.exists(local_lora_path):
        logger.info(f"✅ LoRA weights already available locally: {local_lora_path}")
        return local_lora_path

    logger.info(f"📥 LoRA weights not found locally. Downloading from S3: {lora_folder}...")

    s3_folder_path = os.path.join(S3_WEIGHTS_PATH, lora_folder).replace("\\", "/")
    objects = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=s3_folder_path)

    if "Contents" not in objects:
        raise HTTPException(status_code=404, detail=f"No LoRA weights found in S3 for {lora_folder}.")

    os.makedirs(local_lora_path, exist_ok=True)

    for obj in objects["Contents"]:
        file_name = obj["Key"].split("/")[-1]
        local_file_path = os.path.join(local_lora_path, file_name)

        if file_name:  # Ensure it's a file, not just a directory
            s3_client.download_file(S3_BUCKET, obj["Key"], local_file_path)
            logger.info(f"✅ Downloaded {obj['Key']} to {local_file_path}")

    return local_lora_path

class GenerateRequest(BaseModel):
    prompt: str
    steps: int = 15
    guidance: float = 7.5
    lora_model_name: str = None  # Optional LoRA model


@app.post("/generate")
async def generate_image(req_data: GenerateRequest):
    """
    Generates an image using Stable Diffusion 3.5 with optional LoRA fine-tuning.
    If the LoRA model is not found locally, it is downloaded from S3.
    """
    global pipe

    if pipe is None:
        logger.error("🚨 Model not loaded. Rejecting request.")
        raise HTTPException(status_code=500, detail="Model not loaded. Check server logs.")

    # Load LoRA model if provided
    if req_data.lora_model_name:
        logger.info(f"🔍 Checking for LoRA model: {req_data.lora_model_name}")

        # Check if LoRA model is available locally, otherwise download from S3
        local_lora_path = os.path.join(lora_model_path, req_data.lora_model_name)
        if not os.path.exists(local_lora_path):
            logger.info(f"📥 LoRA model '{req_data.lora_model_name}' not found locally. Downloading from S3...")
            download_lora_from_s3(req_data.lora_model_name)

        try:
            # Load the LoRA model
            pipe.unet = PeftModel.from_pretrained(local_lora_path)
            pipe.unet = pipe.unet.merge_and_unload() 
            if torch.cuda.is_available():
                pipe = pipe.to("cuda")
            logger.info(f"✅ Loaded LoRA model: {req_data.lora_model_name}")

        except Exception as e:
            logger.error(f"❌ Error loading LoRA model '{req_data.lora_model_name}': {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to load LoRA model.")

    try:
        # Validate parameters
        if not (1 <= req_data.steps <= 150):
            logger.warning("❌ Invalid steps parameter: %d", req_data.steps)
            raise HTTPException(status_code=400, detail="steps must be between 1 and 150")
        if not (0.0 <= req_data.guidance <= 15.0):
            logger.warning("❌ Invalid guidance parameter: %.2f", req_data.guidance)
            raise HTTPException(status_code=400, detail="guidance must be between 0.0 and 15.0")

        logger.info(f"🖼️ Generating image with prompt: {req_data.prompt}")
        gen_start_time = time.time()

        # Generate the image
        with torch.inference_mode():
            image = pipe(
                req_data.prompt,
                num_inference_steps=req_data.steps,
                guidance_scale=req_data.guidance,
            ).images[0]

        gen_end_time = time.time()
        logger.info(f"✅ Image generated in {gen_end_time - gen_start_time:.2f} seconds!")

        # Convert image to Base64
        img_io = io.BytesIO()
        image.save(img_io, format="PNG")
        img_io.seek(0)
        base64_img = base64.b64encode(img_io.read()).decode("utf-8")

        logger.info("📤 Sending image response back to client.")
        return {"image": base64_img}

    except Exception as e:
        # Log full error details
        error_message = f"❌ Error generating image: {str(e)}"
        logger.error(error_message)
        raise HTTPException(status_code=500, detail="An internal error occurred. Check server logs for details.")
    


# API Health Check
@app.get("/")
def home():
    logger.info("🛠️ Health check requested.")
    return {"message": "Stable Diffusion 3.5 API is running with LoRA support!"}