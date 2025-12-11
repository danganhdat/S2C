from huggingface_hub import HfApi
import os

# 1. Configuration
# Consider using os.getenv("HF_TOKEN") to keep it out of git history
TOKEN = "YOUR_WRITE_TOKEN_HERE"  
REPO_ID = "danganhdat/s2c-wsss-exp-001-checkpoints"
FOLDER_PATH = "/workspace/S2C/experiments"
path_in_repo = "experiments" # This will put everything inside an 'experiments' folder in the repo

# 2. Initialize API
api = HfApi(token=TOKEN)

# 3. Create Repo (if it doesn't exist yet)
api.create_repo(repo_id=REPO_ID, exist_ok=True, private=True)

# 4. Upload Folder
print(f"Uploading folder {FOLDER_PATH} to {REPO_ID}...")

api.upload_folder(
    folder_path=FOLDER_PATH,
    repo_id=REPO_ID,
    repo_type="model",
    path_in_repo=path_in_repo, # Optional: remove this line to upload contents to root
    ignore_patterns=["*.git*", "*__pycache__*"] # Optional: skip temp files
)

print("Upload complete! 🎉")