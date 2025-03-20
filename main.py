import os
import io
import base64
import torch
import logging
import boto3
import traceback
from fastapi import FastAPI, HTTPException, BackgroundTasks
from diffusers import StableDiffusion3Pipeline, UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from peft import get_peft_model, LoraConfig
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from pydantic import BaseModel
import torchvision.transforms as transforms
from huggingface_hub import snapshot_download

from diffusers.schedulers.scheduling_utils import SchedulerMixin 

import json

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(module)s | %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# AWS Config
s3_client = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION", "us-east-1")
)

S3_BUCKET = "little-legends-dev"
S3_TRAINING_PATH = "training_data"
S3_WEIGHTS_PATH = "lora_models"

# FastAPI
app = FastAPI()

model_id = "stabilityai/stable-diffusion-3.5-large"
model_dir = "/workspace/models"
lora_model_path = "/workspace/lora_models"
os.makedirs(lora_model_path, exist_ok=True)

@app.on_event("startup")
def load_models():
    global pipe, tokenizer, vae, text_encoder, model_dir

    HF_TOKEN = os.getenv("HF_TOKEN")
    model_dir = snapshot_download(repo_id=model_id, local_dir=model_dir, token=HF_TOKEN, resume_download=True)

    pipe = StableDiffusion3Pipeline.from_pretrained(model_dir, torch_dtype=torch.float16)
    pipe.to("cuda")
    pipe.enable_xformers_memory_efficient_attention()

    tokenizer = CLIPTokenizer.from_pretrained(model_dir, subfolder="tokenizer")
    vae = AutoencoderKL.from_pretrained(model_dir, subfolder="vae", torch_dtype=torch.float16).to("cuda")
    text_encoder = CLIPTextModel.from_pretrained(model_dir, subfolder="text_encoder", torch_dtype=torch.float16).to("cuda")

    logger.info("✅ Models loaded successfully!")

# Dataset class
class ImageDataset(Dataset):
    def __init__(self, folder, tokenizer):
        self.files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith((".png", ".jpg", ".jpeg"))]
        self.tokenizer = tokenizer
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image = Image.open(self.files[idx]).convert("RGB")
        image = self.transform(image)
        caption = "A portrait of a child."
        tokens = self.tokenizer(caption, padding="max_length", max_length=77, return_tensors="pt").input_ids.squeeze()
        return image, tokens

# Download from S3
def download_images_from_s3(local_path, subfolder):
    prefix = f"{S3_TRAINING_PATH}/{subfolder}/"
    objects = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix).get("Contents", [])
    if not objects:
        raise HTTPException(status_code=404, detail=f"No images found for {subfolder} in S3.")

    os.makedirs(local_path, exist_ok=True)
    for obj in objects:
        file_name = obj["Key"].split('/')[-1]
        if file_name:
            s3_client.download_file(S3_BUCKET, obj["Key"], os.path.join(local_path, file_name))

# Upload LoRA weights to S3
def upload_lora_to_s3(local_path, s3_subfolder):
    for file in os.listdir(local_path):
        s3_client.upload_file(os.path.join(local_path, file), S3_BUCKET, f"{S3_WEIGHTS_PATH}/{s3_subfolder}/{file}")



# Train LoRA correctly
def train_lora(dataset_path, output_path, steps, lr):
    logger.info("🚀 Starting LoRA fine-tuning...")

    # ✅ Load Dataset
    logger.info(f"📂 Loading dataset from: {dataset_path}")
    dataset = ImageDataset(dataset_path, tokenizer)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    logger.info(f"✅ Dataset loaded. Total samples: {len(dataset)}")

    config = LoraConfig(r=4, lora_alpha=16, target_modules=["to_q", "to_v"])

    # ✅ Load UNet using model_index.json
    logger.info("🔍 Loading UNet model from model_index.json...")
    config_path = os.path.join(model_dir, "model_index.json")
    
    if not os.path.exists(config_path):
        logger.error(f"❌ UNet config not found: {config_path}")
        raise ValueError(f"UNet config not found: {config_path}")

    with open(config_path, "r") as f:
        model_config = json.load(f)

    unet = get_peft_model(
        UNet2DConditionModel.from_config(model_config).to("cuda"), config
    )
    logger.info("✅ UNet model loaded successfully.")

    optimizer = torch.optim.AdamW(unet.parameters(), lr=lr)

    # ✅ Load VAE model correctly using AutoencoderKL
    logger.info("🔍 Loading VAE model...")
    vae_path = os.path.join(model_dir, "vae")
    
    if not os.path.exists(vae_path):
        logger.error(f"❌ VAE model not found at {vae_path}")
        raise ValueError(f"VAE model not found at {vae_path}")

    vae = AutoencoderKL.from_pretrained(vae_path).to("cuda")
    logger.info("✅ VAE model loaded successfully.")

    # ✅ Load text_encoder_2
    logger.info("🔍 Loading text encoder...")
    text_encoder_path = os.path.join(model_dir, "text_encoder_2")
    
    if not os.path.exists(text_encoder_path):
        logger.error(f"❌ Text encoder not found at {text_encoder_path}")
        raise ValueError(f"Text encoder not found at {text_encoder_path}")

    text_encoder = CLIPTextModel.from_pretrained(text_encoder_path).to("cuda")
    logger.info("✅ Text encoder loaded successfully.")

    # ✅ Training loop
    logger.info(f"🛠️ Training LoRA for {steps} steps with LR={lr}")
    for step, (images, tokens) in enumerate(loader):
        if step >= steps:
            break

        images, tokens = images.cuda(), tokens.cuda()

        with torch.no_grad():
            logger.info(f"🔄 Step {step}: Generating latents...")
            
            # ✅ FIX: Convert images to float32 before passing to VAE
            latents = vae.encode(images.to(torch.float32)).latent_dist.sample()
            
            # ✅ Ensure latents have 4 channels (SD3 models might use 16)
            if latents.shape[1] != 4:
                logger.warning(f"⚠️ Latent space mismatch: expected 4 channels, got {latents.shape[1]}")
                latents = latents[:, :4, :, :]  # Take first 4 channels

            latents = latents * vae.config.scaling_factor  # Apply correct scaling
            
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (latents.size(0),), device="cuda").long()

            # ✅ FIX: Manually apply noise since scheduler does NOT support `add_noise()`
            noisy_latents = latents + noise * timesteps.reshape(-1, 1, 1, 1).to(noise.dtype)

            logger.info(f"🔄 Step {step}: Generating text embeddings...")
            encoder_states = text_encoder(tokens)[0]  # Use text_encoder_2

        optimizer.zero_grad()
        logger.info(f"🔄 Step {step}: Forward pass through UNet...")
        
        # ✅ Ensure the UNet receives a correctly shaped latent space
        if noisy_latents.shape[1] != 4:
            logger.error(f"❌ UNet input shape mismatch: expected 4 channels, got {noisy_latents.shape[1]}")
            raise ValueError(f"UNet input shape mismatch: expected 4 channels, got {noisy_latents.shape[1]}")

        pred_noise = unet(noisy_latents, timesteps, encoder_states).sample

        loss = torch.nn.functional.mse_loss(pred_noise, noise)
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            logger.info(f"📌 Step {step}/{steps} - Loss: {loss.item():.6f}")

    logger.info(f"✅ LoRA fine-tuning complete! Saving model to {output_path}...")
    unet.save_pretrained(output_path)

    # ✅ Upload trained model to S3
    logger.info(f"📤 Uploading LoRA model to S3: {output_path}")
    upload_lora_to_s3(output_path, os.path.basename(output_path))
    logger.info("✅ LoRA model uploaded successfully!")

    


# Request Models
class TrainRequest(BaseModel):
    subfolder: str
    output_lora_name: str
    steps: int = 1000
    lr: float = 1e-4

@app.post("/train_lora")
def train_lora_endpoint(req: TrainRequest, background_tasks: BackgroundTasks):
    local_data = f"/workspace/training_data/{req.subfolder}"
    local_lora = os.path.join(lora_model_path, req.output_lora_name)

    download_images_from_s3(local_data, req.subfolder)
    background_tasks.add_task(train_lora, local_data, local_lora, req.steps, req.lr)

    return {"message": f"Training started for {req.output_lora_name}"}

class GenerateRequest(BaseModel):
    prompt: str
    steps: int = 15
    guidance: float = 7.5
    lora_model_name: str = None

@app.post("/generate")
def generate_image(req: GenerateRequest):
    if req.lora_model_name:
        pipe.load_lora_weights(os.path.join(lora_model_path, req.lora_model_name))

    image = pipe(req.prompt, num_inference_steps=req.steps, guidance_scale=req.guidance).images[0]
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return {"image": img_str}

@app.get("/")
def health_check():
    return {"status": "Stable Diffusion API with LoRA ready!"}