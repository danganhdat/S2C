import kagglehub
import shutil
import os
import glob

def download_and_copy(dataset, target_dir):
    print(f"Downloading {dataset} ...")
    path = kagglehub.dataset_download(dataset)
    print("Downloaded to:", path)

    # Copy all files recursively
    for root, dirs, files in os.walk(path):
        for f in files:
            src = os.path.join(root, f)
            dst = os.path.join(target_dir, f)
            shutil.copy(src, dst)
            print("Copied:", dst)

# target folder
TARGET = "se/default"
os.makedirs(TARGET, exist_ok=True)

# Download & copy
download_and_copy("danganhdat/get-se-map-1", TARGET)
download_and_copy("datanhdang/get-se-map-2", TARGET)

# Count .npy files inside se/default
files = glob.glob(os.path.join(TARGET, "*.npy"))
print("Number of .npy files:", len(files))
print("ALL FILES COPIED TO:", TARGET)