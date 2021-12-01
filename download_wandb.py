import wandb
import os

EXPERIMENT="pretrained_ssj"
RUN="3prn98ej"

api = wandb.Api()
run = api.run(f"syde671-copy-paste/{EXPERIMENT}/{RUN}")
files = run.files()

download_dir = f"downloads/{EXPERIMENT}/{RUN}/logs"
os.makedirs(download_dir, exist_ok=True)

# Download logs
for file in files:
    if "events.out.tfevents" in file.name:
        file.download(root=download_dir)
