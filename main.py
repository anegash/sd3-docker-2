import os
import torch
import logging
from huggingface_hub import snapshot_download
from fastapi import FastAPI, HTTPException
from diffusers import StableDiffusion3Pipeline
from pydantic import BaseModel
from pathlib import Path

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# 🟢 Define model storage path inside RunPod volume
MODEL_DIR = "/workspace/models"

# Hugging Face Token
HF_TOKEN = os.getenv("HF_TOKEN")

# 🟢 Initialize FastAPI
app = FastAPI()

# 🟢 Load or Download Model on Startup
@app.on_event("startup")
def load_model():
    global pipe

    # Check if the model exists in persistent storage
    if Path(MODEL_DIR).exists():
        logger.info(f"✅ Model found in persistent storage: {MODEL_DIR}")
    else:
        logger.info(f"⬇️ Model not found. Downloading to {MODEL_DIR}...")
        snapshot_download(
            repo_id="stabilityai/stable-diffusion-3.5-large",
            local_dir=MODEL_DIR,
            token=HF_TOKEN,
            local_dir_use_symlinks=False,
            resume_download=True
        )

    # Load the model
    pipe = StableDiffusion3Pipeline.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16, use_safetensors=True
    )
    pipe.enable_xformers_memory_efficient_attention()
    pipe.to("cuda")
    logger.info("✅ Model Loaded Successfully!")

# 🟢 Define Image Generation Request Model
class GenerateRequest(BaseModel):
    prompt: str
    num_inference_steps: int = 30
    guidance_scale: float = 7.5

@app.post("/generate")
def generate(req: GenerateRequest):
    if pipe is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    images = pipe(
        req.prompt,
        num_inference_steps=req.num_inference_steps,
        guidance_scale=req.guidance_scale
    ).images

    img = images[0]

    return {"message": "✅ Image generated successfully!"}

@app.get("/healthz")
def health():
    return {"status": "ok"}