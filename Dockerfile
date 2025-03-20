FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set environment variables
ARG HF_TOKEN
ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ARG AWS_REGION

ENV HF_TOKEN=$HF_TOKEN
ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
ENV AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
ENV AWS_REGION=${AWS_REGION:-us-east-1}

# Install dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    diffusers \
    transformers \
    accelerate \
    ftfy \
    xformers \
    boto3 \
    huggingface_hub[cli] \
    protobuf \
    sentencepiece \
    torchvision \
    peft \
    safetensors

# Create a persistent directory for storing the model
RUN mkdir -p /workspace/models

# Set working directory
WORKDIR /app

# Copy the application
COPY main.py .

# Expose API port
EXPOSE 8000

# Run API server
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

CMD ["sleep", "infinity"]