import os
import io
import uuid
import boto3
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from diffusers import StableDiffusion3Pipeline
from huggingface_hub import snapshot_download

class GenerationRequest(BaseModel):
    prompt: str
    negative_prompt: str = None
    num_inference_steps: int = 30
    guidance_scale: float = 7.5

app = FastAPI()
pipe = None

# Updated main.py snippet using persistent storage
@app.on_event("startup")
def load_model():
    global pipe
    HF_TOKEN = os.getenv("HF_TOKEN")
    if not HF_TOKEN:
        raise ValueError("HuggingFace token not found! Set HF_TOKEN environment variable.")

    model_path = snapshot_download(
        repo_id="stabilityai/stable-diffusion-3.5-large",
        local_dir="/data/models",  # Persistent volume mount point
        token=HF_TOKEN,
        local_dir_use_symlinks=False,
        resume_download=True
    )

    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, use_safetensors=True
    )
    pipe.enable_xformers_memory_efficient_attention()
    pipe.to("cuda")

@app.get("/healthz")
def health():
    return {"status": "ok"}

@app.post("/generate")
def generate(req: GenerationRequest):
    if pipe is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    images = pipe(
        req.prompt,
        negative_prompt=req.negative_prompt or "",
        num_inference_steps=req.num_inference_steps,
        guidance_scale=req.guidance_scale
    ).images

    img = images[0]

    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    buffered.seek(0)

    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_KEY"),
        aws_secret_access_key=os.getenv("AWS_SECRET")
    )

    bucket_name = "my-sd-output-bucket"
    img_key = f"outputs/{uuid.uuid4()}.png"
    s3.upload_fileobj(buffered, bucket_name, img_key, ExtraArgs={"ContentType": "image/png"})

    url = s3.generate_presigned_url(
        "get_object", Params={"Bucket": bucket_name, "Key": img_key}, ExpiresIn=3600
    )

    return {"image_url": url}