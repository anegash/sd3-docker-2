import torch
import logging
import traceback
import os
import shutil
import time
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from diffusers import StableDiffusion3Pipeline
from huggingface_hub import snapshot_download
from PIL import Image
import io
import base64

# Initialize logging with timestamps and module names
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(module)s | %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Define model storage path
model_id = "stabilityai/stable-diffusion-3.5-large"
model_dir = "/workspace/models"  # Base directory for storing models
model_path = os.path.join(model_dir, model_id.replace("/", "_"))  # Unique folder

# Log system information
logger.info("🔥 Initializing Stable Diffusion API...")
logger.info(f"🚀 Torch version: {torch.__version__}")
logger.info(f"🔧 CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logger.info(f"🖥️ Using GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"🔥 GPU Memory Allocated: {torch.cuda.memory_allocated()} bytes")
    logger.info(f"🔥 GPU Memory Cached: {torch.cuda.memory_reserved()} bytes")

# Ensure model directory exists
os.makedirs(model_dir, exist_ok=True)

# Download model if not found or incomplete
model_index_file = os.path.join(model_path, "model_index.json")

if not os.path.exists(model_index_file):
    if os.path.exists(model_path):
        logger.warning("⚠️ Model folder exists but is incomplete. Removing and redownloading...")
        shutil.rmtree(model_path)  # Remove incomplete model directory

    logger.info("⬇️ Downloading model to /workspace/models...")
    start_time = time.time()
    snapshot_download(repo_id=model_id, local_dir=model_path, local_dir_use_symlinks=False)
    end_time = time.time()
    logger.info(f"✅ Model downloaded successfully in {end_time - start_time:.2f} seconds!")

# Load model
try:
    logger.info("🛠️ Loading the model into memory...")
    start_time = time.time()

    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_path, torch_dtype=torch.float16, variant="fp16"
    )

    end_time = time.time()
    logger.info(f"✅ Model loaded successfully in {end_time - start_time:.2f} seconds!")

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
    pipe = None  # Prevent API from running with an unloaded model

# Define request body model
class GenerateRequest(BaseModel):
    prompt: str
    steps: int = 15
    guidance: float = 7.5

# API Endpoint (POST request)
@app.post("/generate")
async def generate_image(request: Request, req_data: GenerateRequest):
    if pipe is None:
        logger.error("🚨 Model not loaded. Rejecting request.")
        raise HTTPException(status_code=500, detail="Model not loaded. Check server logs.")

    try:
        # Log request details
        logger.info(f"📩 Received request: {req_data.dict()}")

        # Validate parameters
        if not (1 <= req_data.steps <= 150):
            logger.warning("❌ Invalid steps parameter: %d", req_data.steps)
            raise HTTPException(status_code=400, detail="steps must be between 1 and 150")
        if not (0.0 <= req_data.guidance <= 15.0):
            logger.warning("❌ Invalid guidance parameter: %.2f", req_data.guidance)
            raise HTTPException(status_code=400, detail="guidance must be between 0.0 and 15.0")

        logger.info("🖼️ Generating image with prompt: %s", req_data.prompt)
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
        request_info = f"🔹 Request: {await request.json()}"
        traceback_info = f"🛠️ Traceback: {traceback.format_exc()}"

        logger.error("\n%s\n%s\n%s", error_message, request_info, traceback_info)

        raise HTTPException(status_code=500, detail="An internal error occurred. Check server logs for details.")
    
# Root endpoint
@app.get("/")
def home():
    logger.info("🛠️ Health check requested.")
    return {"message": "Stable Diffusion 3.5 API is running on CUDA!"}