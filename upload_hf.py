from huggingface_hub import HfApi

# 1. Configuration
TOKEN = ""  # Paste your WRITE token here
REPO_ID = "danganhdat/s2c-wsss-exp-001-checkpoints"  # Change to your desired repo name
FILE_PATH = "/workspace/S2C/experiments/251211_s2c_wsss-sam-to-cam-exp-1/ckpt/001net_main.pth"
PATH_IN_REPO = "001net_main.pth"  # Name of the file on HuggingFace

# 2. Initialize API
api = HfApi(token=TOKEN)

# 3. Create Repo (if it doesn't exist yet)
# private=True keeps it hidden. Change to False if you want it public.
api.create_repo(repo_id=REPO_ID, exist_ok=True, private=True)

# 4. Upload
print(f"Uploading {PATH_IN_REPO} to {REPO_ID}...")
api.upload_file(
    path_or_fileobj=FILE_PATH,
    path_in_repo=PATH_IN_REPO,
    repo_id=REPO_ID,
    repo_type="model"
)
print("Upload complete! 🎉")