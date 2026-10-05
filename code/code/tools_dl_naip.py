from huggingface_hub import snapshot_download
p = snapshot_download("ssocean/NAIP", local_dir="/path/to/data/models/NAIP", max_workers=8)
print("DONE", p)
