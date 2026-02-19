# VERITAS — Deepfake Face Detector

VERITAS is a production-ready full-stack web app that classifies uploaded face images as **REAL** or **AI GENERATED**, returns confidence scores, and performs EXIF metadata analysis.

## Stack
- Frontend: React + Vite + TailwindCSS (`localhost:3000`)
- Backend: Node.js + Express (`localhost:5000`)
- ML Engine: Python + FastAPI (`localhost:8000`)
- Model: EfficientNet-B0 (timm, fine-tuned)



DATASET DOWNLOAD INSTRUCTIONS
==============================

STEP 1 — PRIMARY DATASET (Real + Fake Western Faces)
Download from Kaggle:
https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces

- This gives you 70k real + 70k fake faces at 256x256
- After download, copy:
  * 7000 images from /real folder → /VeritasDetector/dataset/real/
  * 8000 images from /fake folder → /VeritasDetector/dataset/fake/

STEP 2 — INDIAN REAL FACES (Critical for Indian accuracy)
Download Dataset A from Kaggle:
https://www.kaggle.com/datasets/havingfun/indian-celebrities-faces
- Copy 2000 images → /VeritasDetector/dataset/real/

Download Dataset B from Kaggle:
https://www.kaggle.com/datasets/jangedoo/utkface-new
- Run this command after download to extract Indian faces:
  python ml_engine/dataset_prep.py --source /path/to/UTKFace --dest dataset/real
- This auto-filters race=3 (Indian) and copies ~2000 faces

Download Dataset C from Kaggle:
https://www.kaggle.com/datasets/sushilyadav1998/bollywood-celeb-localized-face-dataset
- Copy 1000 images → /VeritasDetector/dataset/real/

STEP 3 — INDIAN FAKE FACES (Critical for Indian fake detection)
Go to Google Colab: https://colab.research.google.com
Create new notebook and run this code to generate 2000 Indian
AI fake faces using Stable Diffusion:

--- COLAB CODE START ---
!pip install diffusers transformers accelerate torch -q

from diffusers import StableDiffusionPipeline
import torch, os, uuid
from google.colab import files

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    safety_checker=None
)
pipe = pipe.to("cuda")
pipe.enable_attention_slicing()

os.makedirs("indian_fakes", exist_ok=True)

prompts = [
    "realistic portrait photo of indian man face, professional headshot, sharp focus, 4k",
    "realistic portrait photo of indian woman face, professional headshot, sharp focus, 4k",
    "closeup face photo south asian man natural lighting realistic photography",
    "closeup face photo south asian woman natural lighting realistic photography",
    "indian person professional linkedin profile photo realistic face sharp",
    "realistic face photo of young indian man plain background",
    "realistic face photo of young indian woman plain background",
    "south asian man face photo natural sunlight outdoor realistic",
]

negative = "cartoon, anime, painting, illustration, blur, deformed, ugly, watermark, text"

count = 0
target = 2000

for i in range(250):
    for prompt in prompts:
        if count >= target:
            break
        try:
            img = pipe(
                prompt,
                negative_prompt=negative,
                num_inference_steps=25,
                guidance_scale=7.5,
                width=512,
                height=512
            ).images[0]
            img = img.resize((256, 256))
            img.save(f"indian_fakes/sd_indian_fake_{count:05d}.jpg", quality=95)
            count += 1
            if count % 50 == 0:
                print(f"Generated {count}/{target}")
        except Exception as e:
            print(f"Error: {e}")
            continue

print(f"Done! Generated {count} images")

# Zip and download
import shutil
shutil.make_archive("indian_fakes", "zip", "indian_fakes")
files.download("indian_fakes.zip")
--- COLAB CODE END ---

After download, unzip and copy all images → /VeritasDetector/dataset/fake/

FINAL DATASET COUNT CHECK:
dataset/real/ should have ~12,000 images total
dataset/fake/ should have ~10,000 images total
Run: python ml_engine/dataset_prep.py --check to verify counts

## Exact Step-by-Step Dataset Setup (Local Machine)

1. Create a Kaggle account and install Kaggle CLI:
   - `pip3 install kaggle`
2. Download your Kaggle API token (`kaggle.json`) from Kaggle Account settings.
3. Place token:
   - `mkdir -p ~/.kaggle`
   - `mv /path/to/kaggle.json ~/.kaggle/kaggle.json`
   - `chmod 600 ~/.kaggle/kaggle.json`
4. From project root (`VeritasDetector`), create temp download folders:
   - `mkdir -p downloads/kaggle`
5. Download 140k dataset:
   - `kaggle datasets download -d xhlulu/140k-real-and-fake-faces -p downloads/kaggle`
   - `unzip -o downloads/kaggle/140k-real-and-fake-faces.zip -d downloads/kaggle/140k`
6. Copy required subset into dataset folders (choose any 7000 real + 8000 fake):
   - `find downloads/kaggle/140k -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | rg '/real/' | head -n 7000 | while read -r f; do cp "$f" dataset/real/; done`
   - `find downloads/kaggle/140k -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | rg '/fake/' | head -n 8000 | while read -r f; do cp "$f" dataset/fake/; done`
7. Download Indian Celebrities dataset:
   - `kaggle datasets download -d havingfun/indian-celebrities-faces -p downloads/kaggle`
   - `unzip -o downloads/kaggle/indian-celebrities-faces.zip -d downloads/kaggle/indian_celeb`
   - `find downloads/kaggle/indian_celeb -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | head -n 2000 | while read -r f; do cp "$f" dataset/real/; done`
8. Download UTKFace and extract Indian faces:
   - `kaggle datasets download -d jangedoo/utkface-new -p downloads/kaggle`
   - `unzip -o downloads/kaggle/utkface-new.zip -d downloads/kaggle/utkface`
   - `python ml_engine/dataset_prep.py --source downloads/kaggle/utkface --dest dataset/real`
9. Download Bollywood localized faces and copy 1000 images:
   - `kaggle datasets download -d sushilyadav1998/bollywood-celeb-localized-face-dataset -p downloads/kaggle`
   - `unzip -o downloads/kaggle/bollywood-celeb-localized-face-dataset.zip -d downloads/kaggle/bollywood`
   - `find downloads/kaggle/bollywood -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | head -n 1000 | while read -r f; do cp "$f" dataset/real/; done`
10. Generate Indian fakes in Colab using the provided notebook block, then unzip and copy outputs:
   - `unzip -o /path/to/indian_fakes.zip -d downloads/indian_fakes`
   - `find downloads/indian_fakes -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | while read -r f; do cp "$f" dataset/fake/; done`
11. Verify final counts:
   - `python ml_engine/dataset_prep.py --check`

## Installation

From project root:

```bash
chmod +x setup.sh && ./setup.sh
```

## Training and Startup Order (Must Follow)

1. Fill `dataset/real` and `dataset/fake` first.
2. Train model:

```bash
cd ml_engine && python train.py
```

3. Start ML API (requires `ml_engine/model.pth`):

```bash
cd ml_engine && python api.py
```

4. Start Node backend:

```bash
cd server && node server.js
```

5. Start frontend:

```bash
cd frontend && npm run dev
```

Open `http://localhost:3000`.

## How to Test

1. Upload a known real smartphone photo.
2. Upload a known AI-generated face.
3. Confirm verdict, confidence score, and metadata panel update correctly.
4. Stop ML server and test upload again to verify graceful offline handling.

## Expected Accuracy

- Western faces: 90-95%
- Indian/South Asian faces: 85-90%

Accuracy depends heavily on dataset balance, image quality, and training quality.

## Troubleshooting

- **Error: model.pth not found**
  - Run `cd ml_engine && python train.py` first.
- **ML engine offline in frontend**
  - Start ML API: `cd ml_engine && python api.py`.
- **Backend offline in frontend**
  - Start backend: `cd server && node server.js`.
- **Training exits early with dataset errors**
  - Ensure `dataset/real` and `dataset/fake` contain images.
  - Run `python ml_engine/dataset_prep.py --check`.
- **Kaggle CLI unauthorized**
  - Verify `~/.kaggle/kaggle.json` exists and permission is `600`.
- **Slow training**
  - Confirm MPS is available on Mac (`torch.backends.mps.is_available()`).

## Deploy on Render (Free Tier)

This repo includes a Blueprint file: `render.yaml`.

### Important

- Render deploys from a Git provider repo (GitHub/GitLab/Bitbucket).
- The ML service needs `ml_engine/model.pth` at runtime.
- `ml_engine/model.pth` is ignored by `.gitignore`, so for deployment either:
  - upload the model to a public URL and add startup download logic, or
  - force-add model file to git for this deployment:
    - `git add -f ml_engine/model.pth`

### 1. Push project to GitHub

```bash
git init
git add .
git add -f ml_engine/model.pth
git commit -m "VERITAS ready for Render deployment"
git branch -M main
git remote add origin https://github.com/<your-user>/<your-repo>.git
git push -u origin main
```

### 2. Open Render Blueprint

Open:

`https://dashboard.render.com/blueprint/new?repo=https://github.com/<your-user>/<your-repo>`

### 3. Fill required env vars in Render UI

- `ML_ENGINE_URL` (for `veritas-server`):
  - set to your ML service URL after it is created, for example:
  - `https://veritas-ml-engine.onrender.com`
- `CORS_ORIGINS` (for `veritas-server`):
  - set to your frontend URL, for example:
  - `https://veritas-frontend.onrender.com`
- `VITE_API_BASE_URL` (for `veritas-frontend`):
  - set to backend API URL:
  - `https://veritas-server.onrender.com/api`

### 4. Deploy order

1. Deploy `veritas-ml-engine` and verify `/health`.
2. Set `ML_ENGINE_URL` in `veritas-server`, then deploy backend.
3. Set `VITE_API_BASE_URL` and `CORS_ORIGINS`, then deploy frontend.

### 5. Verify

- Backend health: `https://<server>.onrender.com/api/health`
- ML health: `https://<ml>.onrender.com/health`
- Frontend: `https://<frontend>.onrender.com`
