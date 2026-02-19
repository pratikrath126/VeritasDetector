import os
from huggingface_hub import HfApi, create_repo, upload_folder

HF_TOKEN = os.getenv("HF_TOKEN")
HF_SPACE_ID = os.getenv("HF_SPACE_ID")  # e.g. username/veritas-detector

if not HF_TOKEN:
    raise SystemExit("Missing HF_TOKEN env var")
if not HF_SPACE_ID:
    raise SystemExit("Missing HF_SPACE_ID env var (example: username/veritas-detector)")

api = HfApi(token=HF_TOKEN)

create_repo(
    repo_id=HF_SPACE_ID,
    token=HF_TOKEN,
    repo_type="space",
    space_sdk="gradio",
    exist_ok=True,
)

upload_folder(
    repo_id=HF_SPACE_ID,
    repo_type="space",
    folder_path=".",
    token=HF_TOKEN,
)

print(f"Deployed: https://huggingface.co/spaces/{HF_SPACE_ID}")
