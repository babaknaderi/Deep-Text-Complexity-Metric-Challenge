#!/usr/bin/env python3

from pathlib import Path
from zipfile import ZipFile

from gdown import download as dl


print("Preparing Download...")

# define paths
google_url = "https://drive.google.com/uc?id=1DDnt2OJWXTJRfyE2ZnOXMEswc8IASuUF"
 
curr_dir = Path(__file__).parent.resolve()
model_path = curr_dir / "model"
zip_file_path = model_path / "model.zip"

# download model
dl(google_url, str(zip_file_path), quiet=False)

# unzip model
print("Download finished. Now unpacking...")

with ZipFile(zip_file_path, 'r') as zip_file:
    zip_file.extractall(model_path)

print("Unpacking finished.")

# cleanup
zip_file_path.unlink()