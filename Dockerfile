FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    diffusers \
    transformers \
    accelerate \
    ftfy \
    xformers \
    boto3 \
    huggingface_hub \
    protobuf \
    sentencepiece

WORKDIR /app
COPY main.py .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]