import os
from huggingface_hub import snapshot_download

HF_TOKEN = os.getenv("HF_TOKEN")

snapshot_download(
    repo_id="stabilityai/stable-diffusion-3.5-large",
    local_dir="models",
    token=HF_TOKEN,
    local_dir_use_symlinks=False,
    resume_download=True
)