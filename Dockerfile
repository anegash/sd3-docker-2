FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Install dependencies
RUN pip install --no-cache-dir fastapi uvicorn[standard] diffusers transformers accelerate \
    ftfy xformers boto3 huggingface_hub

WORKDIR /app

# Copy application code
COPY main.py .
COPY download_model.py .

# Download Stable Diffusion model weights
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}
RUN python download_model.py && rm download_model.py

# Expose port for FastAPI
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]