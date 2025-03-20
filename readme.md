docker build -t antenehmtk/sd3-docker-lazy .
docker tag antenehmtk/sd3-docker-lazy antenehmtk/sd3-docker-lazy:latest
docker push antenehmtk/sd3-docker-lazy:latest


# for lora
docker build -t antenehmtk/sd3-docker-lazy-lora .
docker tag antenehmtk/sd3-docker-lazy-lora antenehmtk/sd3-docker-lazy-lora:latest
docker push antenehmtk/sd3-docker-lazy-lora:latest



apt update && apt install -y git