import torch
import logging
import traceback
import os
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from diffusers import StableDiffusion3Pipeline
from PIL import Image
import io
import base64

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Define model storage path
model_id = "stabilityai/stable-diffusion-3.5-large"
model_dir = "/workspace/models"  # Local model storage directory
model_path = os.path.join(model_dir, model_id.replace("/", "_"))  # Unique model folder

# Ensure model directory exists
os.makedirs(model_dir, exist_ok=True)

# Load or download model
try:
    if os.path.exists(model_path):
        logger.info("📂 Model directory found. Loading from local storage.")
        pipe = StableDiffusion3Pipeline.from_pretrained(
            model_path, torch_dtype=torch.float16, variant="fp16"
        )
    else:
        logger.info("⬇️ Model not found locally. Downloading to /workspace/models...")
        pipe = StableDiffusion3Pipeline.from_pretrained(
            model_id, torch_dtype=torch.float16, variant="fp16", cache_dir=model_path
        )

    # Move model to GPU if available
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")  # Use NVIDIA GPU
        logger.info("🚀 Running on CUDA")
    else:
        pipe = pipe.to("cpu")  # Fallback to CPU
        logger.warning("⚠️ CUDA not available. Running on CPU (slow performance).")

except Exception as e:
    logger.error("🔥 Error loading Stable Diffusion model: %s", str(e))
    logger.error(traceback.format_exc())
    pipe = None  # Prevent using an unloaded model

# Define request body model
class GenerateRequest(BaseModel):
    prompt: str
    steps: int = 15
    guidance: float = 7.5

# API Endpoint (POST request)
@app.post("/generate")
async def generate_image(request: Request, req_data: GenerateRequest):
    if pipe is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Check server logs.")

    try:
        # Log request details
        logger.info("📩 Received request: %s", req_data.dict())

        # Validate parameters
        if not (1 <= req_data.steps <= 150):
            raise HTTPException(status_code=400, detail="steps must be between 1 and 150")
        if not (0.0 <= req_data.guidance <= 15.0):
            raise HTTPException(status_code=400, detail="guidance must be between 0.0 and 15.0")

        # Generate the image
        with torch.inference_mode():
            image = pipe(
                req_data.prompt,
                num_inference_steps=req_data.steps,
                guidance_scale=req_data.guidance,
            ).images[0]

        # Convert image to Base64
        img_io = io.BytesIO()
        image.save(img_io, format="PNG")
        img_io.seek(0)
        base64_img = base64.b64encode(img_io.read()).decode("utf-8")

        return {"image": base64_img}

    except Exception as e:
        # Log full error details
        error_message = f"❌ Error generating image: {str(e)}"
        request_info = f"🔹 Request: {await request.json()}"
        traceback_info = f"🛠️ Traceback: {traceback.format_exc()}"

        logger.error("\n%s\n%s\n%s", error_message, request_info, traceback_info)

        raise HTTPException(status_code=500, detail="An internal error occurred. Check server logs for details.")
    
# Root endpoint
@app.get("/")
def home():
    return {"message": "Stable Diffusion 3.5 API is running on CUDA!"}