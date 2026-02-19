from PIL import Image, ExifTags
import io

# AI software keywords to detect in EXIF
AI_SOFTWARE_KEYWORDS = [
    "stable diffusion", "midjourney", "dall-e", "dalle",
    "adobe firefly", "firefly", "runway", "pika", "canva ai",
    "generative", "ai generated", "synthetic", "stylegan",
    "deepfake", "faceswap", "reface"
]

# Known real camera manufacturers
REAL_CAMERA_MAKERS = [
    "apple", "samsung", "google", "oneplus", "xiaomi", "oppo",
    "vivo", "realme", "nokia", "sony", "canon", "nikon", "fujifilm",
    "olympus", "panasonic", "leica", "huawei", "motorola", "lg"
]

def check_metadata(image_bytes):
    """
    Analyze image EXIF metadata to determine if image is
    likely real or AI generated.

    Returns dict with status and reason.
    Status values: "Likely Real" | "Suspicious" | "Likely Fake" | "Inconclusive"
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))

        # Check if image has EXIF data
        exif_raw = img._getexif()

        if not exif_raw:
            return {
                "status": "Suspicious",
                "reason": "No camera metadata found — AI generated images typically have no EXIF data",
                "details": {}
            }

        # Map EXIF codes to readable tag names
        exif = {}
        for tag_id, value in exif_raw.items():
            tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
            try:
                if isinstance(value, bytes):
                    value = value.decode('utf-8', errors='ignore')
                exif[tag_name] = str(value)
            except:
                pass

        # Extract key fields
        make = exif.get('Make', '').strip().lower()
        model_name = exif.get('Model', '').strip()
        software = exif.get('Software', '').strip().lower()
        datetime_taken = exif.get('DateTimeOriginal', '')
        gps_info = exif.get('GPSInfo', '')

        details = {
            "camera_make": exif.get('Make', 'Not found'),
            "camera_model": exif.get('Model', 'Not found'),
            "software": exif.get('Software', 'Not found'),
            "date_taken": datetime_taken or 'Not found',
            "has_gps": bool(gps_info),
        }

        # RULE 1: Check for AI software in metadata
        for keyword in AI_SOFTWARE_KEYWORDS:
            if keyword in software:
                return {
                    "status": "Likely Fake",
                    "reason": f"AI software detected in metadata: {exif.get('Software', software)}",
                    "details": details
                }

        # RULE 2: Check for real camera manufacturer
        for cam_maker in REAL_CAMERA_MAKERS:
            if cam_maker in make:
                result = {
                    "status": "Likely Real",
                    "reason": f"Real camera detected: {exif.get('Make', '')} {model_name}",
                    "details": details
                }
                if gps_info:
                    result["reason"] += " + GPS location data found"
                return result

        # RULE 3: GPS data without camera = still suspicious
        if gps_info and not make:
            return {
                "status": "Inconclusive",
                "reason": "GPS data found but no camera manufacturer — metadata may be manually added",
                "details": details
            }

        # RULE 4: Has timestamp but no camera
        if datetime_taken and not make:
            return {
                "status": "Suspicious",
                "reason": "Timestamp found but no camera data — could be AI with added metadata",
                "details": details
            }

        # RULE 5: Has some metadata but nothing identifying
        return {
            "status": "Inconclusive",
            "reason": "Some metadata present but cannot confirm camera or AI source",
            "details": details
        }

    except Exception:
        return {
            "status": "Suspicious",
            "reason": "Could not read image metadata — unusual for standard camera photos",
            "details": {}
        }
