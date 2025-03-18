FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

ARG HF_TOKEN
ENV HF_TOKEN=$HF_TOKEN

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
    sentencepiece 

# Copy pre-downloaded model from EC2 to Docker image
COPY models /app/models

WORKDIR /app
COPY main.py .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
