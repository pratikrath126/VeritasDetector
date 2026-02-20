import io
import os
from typing import Tuple

import gradio as gr
from PIL import Image

from predict import predict_image
from metadata import check_metadata


def _ensure_rgb_image(input_image):
    if input_image is None:
        raise ValueError("Please upload an image.")

    if isinstance(input_image, str):
        img = Image.open(input_image).convert("RGB")
    elif isinstance(input_image, Image.Image):
        img = input_image.convert("RGB")
    else:
        # numpy array path from gradio image component
        img = Image.fromarray(input_image).convert("RGB")

    return img


def _to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _summary_label(label: str, confidence: float, meta_status: str) -> str:
    if label == "Fake" and meta_status in {"Suspicious", "Likely Fake"}:
        return f"HIGH CONFIDENCE AI GENERATED ({confidence:.2f}%)"
    if label == "Real" and meta_status == "Likely Real":
        return f"HIGH CONFIDENCE REAL PERSON ({confidence:.2f}%)"
    return f"ANALYSIS COMPLETE ({label}, {confidence:.2f}%)"


def analyze(image_input) -> Tuple[str, str, str]:
    try:
        img = _ensure_rgb_image(image_input)
        image_bytes = _to_bytes(img)

        prediction = predict_image(image_bytes)
        metadata = check_metadata(image_bytes)

        label = prediction["label"]
        confidence = prediction["confidence"]
        scores = prediction["scores"]

        verdict = "AI GENERATED FACE DETECTED" if label == "Fake" else "REAL PERSON DETECTED"
        summary = _summary_label(label, confidence, metadata.get("status", "Inconclusive"))

        details = metadata.get("details", {})
        report = (
            f"Verdict: {verdict}\n"
            f"Model Confidence: {confidence:.2f}%\n"
            f"Scores -> Real: {scores.get('real', 0):.2f}% | Fake: {scores.get('fake', 0):.2f}%\n\n"
            f"Metadata Status: {metadata.get('status', 'Inconclusive')}\n"
            f"Reason: {metadata.get('reason', 'No metadata analysis available')}\n"
            f"Camera: {details.get('camera_make', 'Not found')}\n"
            f"Model: {details.get('camera_model', 'Not found')}\n"
            f"Software: {details.get('software', 'Not found')}\n"
            f"GPS: {'Yes' if details.get('has_gps') else 'No'}\n"
        )

        badge = f"{label} ({confidence:.2f}%)"
        return badge, summary, report
    except Exception as exc:
        error_message = (
            "Prediction failed. Ensure model.pth exists in this Space and is compatible.\n"
            f"Error: {exc}"
        )
        return "Error", "ANALYSIS FAILED", error_message


DESCRIPTION = """
# VERITAS - Deepfake Face Detector
Upload a face image to classify it as **Real** or **AI Generated** using EfficientNet-B0 and EXIF metadata heuristics.
"""

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload Face Image")

    analyze_btn = gr.Button("Analyze", variant="primary")

    with gr.Row():
        badge_out = gr.Textbox(label="Prediction")
        summary_out = gr.Textbox(label="Overall Summary")

    report_out = gr.Textbox(label="Detailed Analysis", lines=14)

    analyze_btn.click(
        fn=analyze,
        inputs=[image_input],
        outputs=[badge_out, summary_out, report_out],
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        show_api=False,
        share=True,
    )
