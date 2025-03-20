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
    dataset = ImageDataset(dataset_path, tokenizer)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    config = LoraConfig(r=4, lora_alpha=16, target_modules=["to_q", "to_v"])

    # ✅ Load UNet using model_index.json instead of missing unet/
    unet = get_peft_model(
        UNet2DConditionModel.from_pretrained(model_dir, config="model_index.json").to("cuda"), config
    )

    optimizer = torch.optim.AdamW(unet.parameters(), lr=lr)

    # ✅ Load VAE model correctly
    vae_path = os.path.join(model_dir, "vae")
    vae = UNet2DConditionModel.from_pretrained(vae_path).to("cuda")

    # ✅ Load text_encoder_2
    text_encoder_path = os.path.join(model_dir, "text_encoder_2")
    text_encoder = CLIPTextModel.from_pretrained(text_encoder_path).to("cuda")

    for step, (images, tokens) in enumerate(loader):
        if step >= steps:
            break
        images, tokens = images.cuda(), tokens.cuda()

        with torch.no_grad():
            # ✅ Ensure images are cast to float16
            latents = vae.encode(images.to(torch.float16)).latent_dist.sample() * vae.config.scaling_factor
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, pipe.scheduler.config.num_train_timesteps, (latents.size(0),), device="cuda"
            ).long()
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

            encoder_states = text_encoder(tokens)[0]  # ✅ Use text_encoder_2

        optimizer.zero_grad()
        pred_noise = unet(noisy_latents, timesteps, encoder_states).sample
        loss = torch.nn.functional.mse_loss(pred_noise, noise)
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            logger.info(f"Step {step}/{steps} - Loss: {loss.item()}")

    unet.save_pretrained(output_path)
    upload_lora_to_s3(output_path, os.path.basename(output_path))

    
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