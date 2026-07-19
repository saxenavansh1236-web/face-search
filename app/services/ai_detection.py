"""
ai_detection.py

Heuristic checks for whether an uploaded image is likely AI-generated
or edited/composited. Combines several independent, lightweight signals
into one overall verdict. None of these are a substitute for a trained
deepfake-detection model - that requires a neural network trained on
labeled real/fake face datasets (e.g. FaceForensics++), which isn't
something that can be built from scratch or bundled here. What follows
are classical, well-established forensic heuristics that catch a real
(but limited) share of cases, each clearly labeled by confidence.

Signals used:
1. Filename pattern        - high confidence, easily bypassed by renaming
2. Embedded AI metadata     - high confidence, easily stripped by re-saving
3. Missing camera EXIF       - low confidence, common false positive
4. Error Level Analysis       - medium confidence, flags likely edited/
                                 spliced regions (classic JPEG forensics,
                                 weak on lossless formats like PNG)
5. Noise-consistency check     - medium confidence, format-agnostic
                                   alternative to ELA for PNGs and other
                                   lossless formats
6. FFT frequency artifacts       - low-medium confidence, flags GAN/diffusion
                                     upsampling artifacts in the frequency domain

None of these catch a deepfake or edit that was carefully re-compressed,
re-exported, or done with modern inpainting that avoids sharp seams.
This is a triage tool, not a forensic-grade verdict.
"""

from typing import Dict, Any, List
from io import BytesIO
import numpy as np
from PIL import Image, ImageChops, ImageFilter
from PIL.ExifTags import TAGS

_AI_SOFTWARE_SIGNATURES = [
    "stable diffusion", "midjourney", "dall-e", "dalle",
    "comfyui", "automatic1111", "invokeai", "novelai", "leonardo.ai",
    "adobe firefly", "runwayml", "stability ai",
]

_AI_PNG_TEXT_KEYS = ["parameters", "prompt", "workflow", "generation_data"]

_AI_FILENAME_PATTERNS = [
    "chatgpt_image", "chatgpt-image", "dalle_", "dall-e_", "dalle-",
    "midjourney", "stable-diffusion", "stablediffusion", "sd_output",
    "comfyui", "leonardo_ai", "leonardoai", "sora_", "gemini_image",
    "grok_image", "firefly_", "novelai_", "generated_image",
]


def _check_filename(original_filename: str) -> Dict[str, Any]:
    if not original_filename:
        return {}
    normalized = original_filename.lower()
    for sep in (" ", ",", "_", "-"):
        normalized = normalized.replace(sep, "")
    for pattern in _AI_FILENAME_PATTERNS:
        normalized_pattern = pattern.replace("_", "").replace("-", "")
        if normalized_pattern in normalized:
            return {
                "signal": "filename",
                "is_flagged": True,
                "confidence": "high",
                "reason": "Filename ('" + original_filename + "') matches the naming pattern "
                          "used by AI image tools (e.g. ChatGPT/DALL-E, Midjourney).",
            }
    return {}


def _check_metadata(img: Image.Image) -> Dict[str, Any]:
    png_text = getattr(img, "text", {}) or {}
    for key in png_text:
        if key.lower() in _AI_PNG_TEXT_KEYS:
            return {
                "signal": "metadata",
                "is_flagged": True,
                "confidence": "high",
                "reason": "Image contains AI-generation metadata ('" + key + "' field), "
                          "typical of tools like Stable Diffusion.",
            }

    exif_data = {}
    try:
        raw_exif = img.getexif()
        for tag_id, value in raw_exif.items():
            exif_data[TAGS.get(tag_id, tag_id)] = value
    except Exception:
        pass

    combined = (str(exif_data.get("Software", "")) + " " + str(exif_data.get("Artist", ""))).lower()
    for sig in _AI_SOFTWARE_SIGNATURES:
        if sig in combined:
            return {
                "signal": "metadata",
                "is_flagged": True,
                "confidence": "high",
                "reason": "Image metadata names a known AI generation tool ('" + sig + "').",
            }

    has_camera_info = any(k in exif_data for k in ("Make", "Model", "DateTimeOriginal", "LensModel"))
    if not has_camera_info and not exif_data:
        return {
            "signal": "metadata",
            "is_flagged": True,
            "confidence": "low",
            "reason": "Image has no camera/EXIF metadata at all. Common in AI-generated "
                      "images, but also happens with real photos that were edited, "
                      "screenshotted, or exported without metadata.",
        }
    return {}


def _check_error_level_analysis(img: Image.Image) -> Dict[str, Any]:
    """
    Re-saves the image at a fixed JPEG quality and diffs it against the
    original. Regions that were pasted in from a different source (or
    edited and re-saved separately) tend to show a different error
    level than the rest of the image - a classic photo-forensics
    technique. Only meaningful for JPEG-family images; much weaker
    (often silent) on lossless formats like PNG, which is why
    _check_noise_consistency exists as a format-agnostic backup.
    """
    try:
        rgb = img.convert("RGB")
        buffer = BytesIO()
        rgb.save(buffer, "JPEG", quality=90)
        buffer.seek(0)
        resaved = Image.open(buffer)

        diff = ImageChops.difference(rgb, resaved)
        diff_arr = np.array(diff).astype(np.float32)

        h, w, _ = diff_arr.shape
        rows, cols = 6, 6
        cell_h, cell_w = max(h // rows, 1), max(w // cols, 1)
        cell_means = []
        for r in range(rows):
            for c in range(cols):
                cell = diff_arr[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w]
                if cell.size:
                    cell_means.append(cell.mean())

        if len(cell_means) < 4:
            return {}

        cell_means = np.array(cell_means)
        overall_std = cell_means.std()
        overall_mean = cell_means.mean() + 1e-6
        variability_ratio = overall_std / overall_mean

        if variability_ratio > 0.8 and overall_mean > 0.3:
            return {
                "signal": "error_level_analysis",
                "is_flagged": True,
                "confidence": "medium",
                "reason": "Error Level Analysis found regions with inconsistent compression "
                          "history, which can indicate parts of the image were edited, "
                          "pasted in, or composited from a different source.",
            }
    except Exception:
        pass
    return {}


def _check_noise_consistency(img: Image.Image) -> Dict[str, Any]:
    """
    Format-agnostic alternative to Error Level Analysis. Instead of
    relying on JPEG recompression artifacts (which PNGs and other
    lossless formats don't have), this measures local noise variance
    across a grid of regions using a high-pass residual. Composited/
    blended images - like a double-exposure poster combining a photo
    with translucent overlays - often show noticeably different noise
    levels between regions, since each layer came from a different
    source or was blended/softened differently. Real, unedited photos
    tend to have fairly uniform sensor noise across the whole frame.
    """
    try:
        gray = np.array(img.convert("L")).astype(np.float32)
        h, w = gray.shape
        if h < 60 or w < 60:
            return {}

        blurred_img = img.convert("L").filter(ImageFilter.BoxBlur(2))
        blurred = np.array(blurred_img).astype(np.float32)
        noise = np.abs(gray - blurred)

        rows, cols = 6, 6
        cell_h, cell_w = max(h // rows, 1), max(w // cols, 1)
        cell_noise_means = []
        for r in range(rows):
            for c in range(cols):
                cell = noise[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w]
                if cell.size:                    cell_noise_means.append(cell.mean())

        if len(cell_noise_means) < 4:
            return {}

        cell_noise_means = np.array(cell_noise_means)
        overall_mean = cell_noise_means.mean() + 1e-6
        overall_std = cell_noise_means.std()
        variability_ratio = overall_std / overall_mean

        if variability_ratio > 0.7 and overall_mean > 1.0:
            return {
                "signal": "noise_analysis",
                "is_flagged": True,
                "confidence": "medium",
                "reason": "Noise-level analysis found regions with inconsistent texture/"
                          "noise patterns across the image, which can indicate blended, "
                          "composited, or double-exposure editing (works on PNGs and other "
                          "formats where JPEG-based Error Level Analysis doesn't apply).",
            }
    except Exception:
        pass
    return {}


def _check_frequency_artifacts(img: Image.Image) -> Dict[str, Any]:
    """
    GAN/diffusion model upsampling layers can leave faint periodic
    checkerboard-like patterns, visible as unusually concentrated peaks
    in the image's frequency spectrum away from the center (low
    frequencies). This is a soft signal - real photos with repetitive
    textures (fabric, brick, grids) can also trigger it, hence "low".
    """
    try:
        gray = np.array(img.convert("L").resize((256, 256))).astype(np.float32)
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude = np.log(np.abs(fshift) + 1)

        h, w = magnitude.shape
        cy, cx = h // 2, w // 2
        radius = 20
        y, x = np.ogrid[:h, :w]
        mask = (x - cx) ** 2 + (y - cy) ** 2 > radius ** 2

        high_freq = magnitude[mask]
        if high_freq.size == 0:
            return {}

        peak_ratio = high_freq.max() / (high_freq.mean() + 1e-6)

        if peak_ratio > 6.5:
            return {
                "signal": "frequency_analysis",
                "is_flagged": True,
                "confidence": "low",
                "reason": "Frequency-domain analysis found unusually sharp periodic patterns, "
                          "sometimes left behind by GAN/diffusion image generators. Real "
                          "photos with repetitive textures can also trigger this - treat "
                          "as a weak signal.",
            }
    except Exception:
        pass
    return {}


def detect_ai_generated(image_path: str, original_filename: str = "") -> Dict[str, Any]:
    """
    Runs all checks and returns a combined verdict.
    """
    signals: List[Dict[str, Any]] = []

    fname_result = _check_filename(original_filename)
    if fname_result:
        signals.append(fname_result)

    try:
        with Image.open(image_path) as img:
            meta_result = _check_metadata(img)
            if meta_result:
                signals.append(meta_result)

            ela_result = _check_error_level_analysis(img)
            if ela_result:
                signals.append(ela_result)

            noise_result = _check_noise_consistency(img)
            if noise_result:
                signals.append(noise_result)

            freq_result = _check_frequency_artifacts(img)
            if freq_result:
                signals.append(freq_result)

    except Exception as exc:
        return {
            "is_likely_ai": False,
            "confidence": "none",
            "reason": "Could not analyze image: " + str(exc),
            "all_signals": [],
        }

    if not signals:
        return {
            "is_likely_ai": False,
            "confidence": "none",
            "reason": "No AI-generation or editing signals detected.",
            "all_signals": [],
        }

    confidence_rank = {"high": 3, "medium": 2, "low": 1}
    strongest = max(signals, key=lambda s: confidence_rank.get(s["confidence"], 0))

    return {
        "is_likely_ai": True,
        "confidence": strongest["confidence"],
        "reason": strongest["reason"],
        "all_signals": signals,
    }
