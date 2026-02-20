import os
from typing import Tuple

import gradio as gr

from metadata import check_metadata
from predict import predict_image


def _read_bytes(file_path: str) -> bytes:
    if not file_path:
        raise ValueError("Please upload an image.")
    with open(file_path, "rb") as f:
        return f.read()


def _summary_label(label: str, confidence: float, meta_status: str) -> str:
    if label == "Fake" and meta_status in {"Suspicious", "Likely Fake"}:
        return f"HIGH CONFIDENCE AI GENERATED ({confidence:.2f}%)"
    if label == "Real" and meta_status == "Likely Real":
        return f"HIGH CONFIDENCE REAL PERSON ({confidence:.2f}%)"
    return f"ANALYSIS COMPLETE ({label}, {confidence:.2f}%)"


def _safe_detail(details: dict, key: str, default: str = "Not found") -> str:
    value = details.get(key, default)
    if value in (None, "", "{}"):
        return default
    return str(value)


def _status_style(status: str):
    status = (status or "Inconclusive").strip()
    if status == "Likely Real":
        return "#16a34a", "#dcfce7", "#14532d", "✅"
    if status == "Likely Fake":
        return "#dc2626", "#fee2e2", "#7f1d1d", "❌"
    if status == "Suspicious":
        return "#ca8a04", "#fef9c3", "#713f12", "⚠️"
    return "#6b7280", "#f3f4f6", "#111827", "❓"


def _verdict_html(label: str, confidence: float, real_score: float, fake_score: float) -> str:
    is_fake = label == "Fake"
    primary = "#dc2626" if is_fake else "#16a34a"
    bg = "#450a0a" if is_fake else "#052e16"
    title = "AI GENERATED FACE DETECTED" if is_fake else "REAL PERSON DETECTED"

    return f"""
    <div style='border:1px solid {primary}; border-radius:14px; padding:16px; background:{bg}; color:#fff;'>
      <div style='font-size:20px; font-weight:800; letter-spacing:0.4px;'>{title}</div>
      <div style='margin-top:8px; font-size:14px; opacity:0.95;'>Confidence: <b>{confidence:.2f}%</b></div>
      <div style='margin-top:12px;'>
        <div style='font-size:13px; margin-bottom:4px;'>Model score split</div>
        <div style='background:#1f2937; border-radius:999px; height:12px; overflow:hidden;'>
          <div style='width:{confidence:.2f}%; height:12px; background:{primary};'></div>
        </div>
        <div style='margin-top:8px; font-size:12px; color:#d1d5db;'>Real: {real_score:.2f}% | Fake: {fake_score:.2f}%</div>
      </div>
    </div>
    """


def _metadata_html(metadata: dict) -> str:
    status = metadata.get("status", "Inconclusive")
    reason = metadata.get("reason", "No metadata analysis available")
    details = metadata.get("details", {}) or {}

    color, light_bg, text_color, icon = _status_style(status)
    camera = _safe_detail(details, "camera_make")
    model = _safe_detail(details, "camera_model")
    software = _safe_detail(details, "software")
    gps = "Yes" if details.get("has_gps") else "No"

    return f"""
    <div style='border:1px solid #1f2937; border-radius:14px; padding:16px; background:#0b1220; color:#e5e7eb;'>
      <div style='font-size:18px; font-weight:700; margin-bottom:10px;'>Metadata Analysis</div>
      <div style='display:inline-block; background:{light_bg}; color:{text_color}; border:1px solid {color};
                  border-radius:999px; padding:6px 10px; font-size:13px; font-weight:700;'>
        {icon} {status}
      </div>
      <div style='margin-top:10px; font-size:13px; color:#cbd5e1;'>Reason: {reason}</div>
      <div style='margin-top:14px; display:grid; grid-template-columns:1fr 1fr; gap:10px;'>
        <div style='background:#111827; border-radius:8px; padding:10px;'><div style='font-size:11px; color:#94a3b8;'>Camera</div><div style='font-size:13px;'>{camera}</div></div>
        <div style='background:#111827; border-radius:8px; padding:10px;'><div style='font-size:11px; color:#94a3b8;'>Model</div><div style='font-size:13px;'>{model}</div></div>
        <div style='background:#111827; border-radius:8px; padding:10px;'><div style='font-size:11px; color:#94a3b8;'>Software</div><div style='font-size:13px;'>{software}</div></div>
        <div style='background:#111827; border-radius:8px; padding:10px;'><div style='font-size:11px; color:#94a3b8;'>GPS</div><div style='font-size:13px;'>{gps}</div></div>
      </div>
    </div>
    """


def _summary_html(summary: str) -> str:
    return f"""
    <div style='border:1px solid #334155; border-radius:14px; padding:14px; background:#020617; color:#f8fafc;'>
      <div style='font-size:12px; color:#94a3b8; margin-bottom:6px;'>OVERALL VERDICT</div>
      <div style='font-size:18px; font-weight:800;'>{summary}</div>
    </div>
    """


def analyze(image_path: str) -> Tuple[str, str, str]:
    try:
        image_bytes = _read_bytes(image_path)

        prediction = predict_image(image_bytes)
        metadata = check_metadata(image_bytes)

        label = prediction["label"]
        confidence = float(prediction["confidence"])
        real_score = float(prediction["scores"].get("real", 0.0))
        fake_score = float(prediction["scores"].get("fake", 0.0))

        summary = _summary_label(label, confidence, metadata.get("status", "Inconclusive"))

        return (
            _verdict_html(label, confidence, real_score, fake_score),
            _metadata_html(metadata),
            _summary_html(summary),
        )
    except Exception as exc:
        err = f"<div style='padding:14px;border:1px solid #ef4444;border-radius:12px;background:#450a0a;color:#fee2e2;'>Error: {exc}</div>"
        return err, err, err


CUSTOM_CSS = """
:root { --radius-lg: 16px; }
.gradio-container {
  background: radial-gradient(circle at top right, #0f172a, #020617 55%);
}
.footer {display:none !important;}
"""

DESCRIPTION = """
# VERITAS - Deepfake Face Detector
Upload a face image to classify it as **Real** or **AI Generated** with model confidence and EXIF metadata analysis.
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue"), css=CUSTOM_CSS, title="VERITAS") as demo:
    gr.Markdown(DESCRIPTION)

    image_input = gr.Image(type="filepath", label="Upload Face Image")
    analyze_btn = gr.Button("Analyze", variant="primary", size="lg")

    with gr.Row():
        verdict_out = gr.HTML(label="Prediction")
        metadata_out = gr.HTML(label="Metadata")

    summary_out = gr.HTML(label="Summary")

    analyze_btn.click(
        fn=analyze,
        inputs=[image_input],
        outputs=[verdict_out, metadata_out, summary_out],
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        show_api=False,
        share=True,
    )
