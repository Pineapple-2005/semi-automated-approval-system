"""
template_matcher.py

Document type classifier using image analysis and ORB features.
Heuristic-based classification since no stored reference template images are used.
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Module-level config cache
_configs: dict = {}

# Base directory for template config JSON files
_TEMPLATES_CONFIG_DIR = Path(__file__).resolve().parent.parent / "templates_config"

# Document type heuristic profiles
# aspect_ratio: width / height
# CR80 card standard (85.6mm x 54mm) -> ~1.585
_DOCUMENT_PROFILES = {
    "umid": {
        "aspect_ratio_target": 1.586,
        "aspect_ratio_tolerance": 0.12,
        "dominant_color": "blue",
        "keypoint_count_min": 180,
        "keypoint_count_max": 500,
        "text_density_min": 0.08,
        "weights": {
            "aspect_ratio": 0.30,
            "color": 0.25,
            "keypoint": 0.25,
            "text_density": 0.10,
            "edge_density": 0.10,
        },
    },
    "passport": {
        # Booklet open -> landscape ~1.42; closed cover -> near 1.0 (portrait)
        "aspect_ratio_target": 1.42,
        "aspect_ratio_tolerance": 0.20,
        "dominant_color": "neutral",
        "keypoint_count_min": 100,
        "keypoint_count_max": 400,
        "text_density_min": 0.12,
        "weights": {
            "aspect_ratio": 0.25,
            "color": 0.10,
            "keypoint": 0.20,
            "text_density": 0.25,
            "edge_density": 0.20,
        },
    },
    "drivers_license": {
        "aspect_ratio_target": 1.586,
        "aspect_ratio_tolerance": 0.12,
        "dominant_color": "blue",
        "keypoint_count_min": 150,
        "keypoint_count_max": 450,
        "text_density_min": 0.09,
        "weights": {
            "aspect_ratio": 0.30,
            "color": 0.20,
            "keypoint": 0.25,
            "text_density": 0.12,
            "edge_density": 0.13,
        },
    },
    "philsys": {
        "aspect_ratio_target": 1.586,
        "aspect_ratio_tolerance": 0.12,
        "dominant_color": "gold_blue",
        "keypoint_count_min": 160,
        "keypoint_count_max": 480,
        "text_density_min": 0.08,
        "weights": {
            "aspect_ratio": 0.30,
            "color": 0.25,
            "keypoint": 0.20,
            "text_density": 0.12,
            "edge_density": 0.13,
        },
    },
    "prc": {
        "aspect_ratio_target": 1.586,
        "aspect_ratio_tolerance": 0.12,
        "dominant_color": "blue_gold",
        "keypoint_count_min": 140,
        "keypoint_count_max": 460,
        "text_density_min": 0.08,
        "weights": {
            "aspect_ratio": 0.30,
            "color": 0.25,
            "keypoint": 0.20,
            "text_density": 0.12,
            "edge_density": 0.13,
        },
    },
}

# Minimum score to assert a match; below this -> "unknown"
_MIN_CONFIDENCE_THRESHOLD = 0.35


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_template_config(document_type: str) -> dict:
    """Load JSON config for a document type from templates_config/{document_type}.json.

    Results are cached in the module-level _configs dict.

    Args:
        document_type: One of umid, passport, drivers_license, philsys, prc.

    Returns:
        Parsed config dict, or an empty dict if the file does not exist.
    """
    if document_type in _configs:
        return _configs[document_type]

    config_path = _TEMPLATES_CONFIG_DIR / f"{document_type}.json"
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            _configs[document_type] = data
            logger.debug("Loaded template config for %s from %s", document_type, config_path)
            return data
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not load template config for %s: %s", document_type, exc)

    _configs[document_type] = {}
    return {}


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_for_matching(image_bytes: bytes) -> np.ndarray:
    """Decode image bytes, convert to grayscale, resize to 800x500, apply CLAHE.

    Args:
        image_bytes: Raw image bytes (JPEG, PNG, etc.).

    Returns:
        Preprocessed grayscale numpy array of shape (500, 800).

    Raises:
        ValueError: If the image cannot be decoded.
    """
    nparr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Failed to decode image bytes into a grayscale image.")

    # Resize to standard 800x500 (width x height)
    img_resized = cv2.resize(img, (800, 500), interpolation=cv2.INTER_AREA)

    # CLAHE for local contrast enhancement (helps ORB detect features on low-contrast docs)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_enhanced = clahe.apply(img_resized)

    return img_enhanced


# ---------------------------------------------------------------------------
# Characteristic analysis
# ---------------------------------------------------------------------------

def analyze_document_characteristics(
    image_np: np.ndarray,
    color_image_np: np.ndarray,
) -> dict:
    """Extract heuristic features from a preprocessed document image.

    Args:
        image_np: Grayscale image as returned by preprocess_for_matching (H x W).
        color_image_np: Original color image resized/cropped to the same spatial
                        dimensions as image_np (H x W x 3, BGR).

    Returns:
        dict with keys:
            aspect_ratio (float)        - width / height of the image
            keypoint_count (int)        - number of ORB keypoints detected
            text_density (float)        - fraction of binarized white pixels
            edge_density (float)        - fraction of Canny edge pixels
            color_profile (str)         - dominant color label
    """
    h, w = image_np.shape[:2]
    aspect_ratio: float = w / h if h > 0 else 0.0

    # --- ORB keypoints ---
    orb = cv2.ORB_create(nfeatures=500)
    keypoints, _descriptors = orb.detectAndCompute(image_np, None)
    keypoint_count: int = len(keypoints) if keypoints is not None else 0

    # --- Text density: Otsu binarisation, ratio of foreground pixels ---
    _thresh_val, binary = cv2.threshold(
        image_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    total_pixels = h * w
    white_pixels = int(np.count_nonzero(binary))
    text_density: float = white_pixels / total_pixels if total_pixels > 0 else 0.0

    # --- Edge density: Canny ---
    edges = cv2.Canny(image_np, threshold1=50, threshold2=150)
    edge_pixels = int(np.count_nonzero(edges))
    edge_density: float = edge_pixels / total_pixels if total_pixels > 0 else 0.0

    # --- Dominant color analysis (BGR channels) ---
    color_profile = _dominant_color_label(color_image_np)

    return {
        "aspect_ratio": aspect_ratio,
        "keypoint_count": keypoint_count,
        "text_density": text_density,
        "edge_density": edge_density,
        "color_profile": color_profile,
    }


def _dominant_color_label(color_image_np: np.ndarray) -> str:
    """Determine the dominant color category of a BGR image.

    Categories used for Philippine IDs:
        blue       - strong blue channel dominance (UMID, DL, PRC base)
        gold       - strong red+green with lower blue (PhilSys accents, PRC gold)
        gold_blue  - mixed gold + blue signature (PhilSys)
        blue_gold  - mixed blue + gold signature (PRC)
        neutral    - no single channel heavily dominant (passport)
        green      - green dominant
        red        - red dominant

    Returns:
        One of the label strings above.
    """
    if color_image_np is None or color_image_np.size == 0:
        return "neutral"

    # Ensure color image is resized consistently
    resized = cv2.resize(color_image_np, (200, 125), interpolation=cv2.INTER_AREA)

    # Convert to HSV for more reliable color detection
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

    # Count pixels in hue ranges (H in OpenCV is 0-179)
    # Blue: H ~100-130
    blue_mask = cv2.inRange(hsv, (100, 60, 60), (130, 255, 255))
    # Gold/yellow: H ~20-35
    gold_mask = cv2.inRange(hsv, (20, 60, 100), (35, 255, 255))
    # Red (wraps around): H ~0-10 and 165-179
    red_mask1 = cv2.inRange(hsv, (0, 60, 60), (10, 255, 255))
    red_mask2 = cv2.inRange(hsv, (165, 60, 60), (179, 255, 255))
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    # Green: H ~40-80
    green_mask = cv2.inRange(hsv, (40, 60, 60), (80, 255, 255))

    total = resized.shape[0] * resized.shape[1]
    blue_ratio = float(np.count_nonzero(blue_mask)) / total
    gold_ratio = float(np.count_nonzero(gold_mask)) / total
    red_ratio = float(np.count_nonzero(red_mask)) / total
    green_ratio = float(np.count_nonzero(green_mask)) / total

    _SIGNIFICANT = 0.08  # threshold to consider a color "present"

    has_blue = blue_ratio >= _SIGNIFICANT
    has_gold = gold_ratio >= _SIGNIFICANT
    has_red = red_ratio >= _SIGNIFICANT
    has_green = green_ratio >= _SIGNIFICANT

    if has_blue and has_gold:
        # Distinguish philsys (gold_blue) vs prc (blue_gold) by ratio ordering
        if gold_ratio >= blue_ratio:
            return "gold_blue"
        return "blue_gold"
    if has_blue:
        return "blue"
    if has_gold:
        return "gold"
    if has_red:
        return "red"
    if has_green:
        return "green"
    return "neutral"


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _score_aspect_ratio(measured: float, target: float, tolerance: float) -> float:
    """Return 1.0 if within tolerance, decaying linearly beyond tolerance."""
    diff = abs(measured - target)
    if diff <= tolerance:
        return 1.0
    # Allow up to 2x tolerance before hitting 0
    if diff >= tolerance * 2:
        return 0.0
    return 1.0 - (diff - tolerance) / tolerance


def _score_color(measured_label: str, expected_label: str) -> float:
    """Fuzzy color score; partial credit for related color families."""
    if measured_label == expected_label:
        return 1.0

    # Partial credit matrix for related color families
    _partial: dict[tuple[str, str], float] = {
        ("blue", "blue_gold"): 0.6,
        ("blue_gold", "blue"): 0.6,
        ("blue", "gold_blue"): 0.5,
        ("gold_blue", "blue"): 0.5,
        ("gold", "gold_blue"): 0.6,
        ("gold_blue", "gold"): 0.6,
        ("gold", "blue_gold"): 0.5,
        ("blue_gold", "gold"): 0.5,
        ("gold_blue", "blue_gold"): 0.7,
        ("blue_gold", "gold_blue"): 0.7,
    }
    return _partial.get((measured_label, expected_label), 0.0)


def _score_keypoints(count: int, kp_min: int, kp_max: int) -> float:
    """Score based on whether keypoint count falls in expected range."""
    if kp_min <= count <= kp_max:
        return 1.0
    if count < kp_min:
        ratio = count / kp_min if kp_min > 0 else 0.0
        return max(0.0, ratio)
    # count > kp_max: penalise slightly (could be very noisy image)
    excess = (count - kp_max) / kp_max if kp_max > 0 else 0.0
    return max(0.0, 1.0 - excess * 0.5)


def _score_text_density(density: float, density_min: float) -> float:
    """Score linearly from 0 at 0 to 1.0 at density_min, capped at 1.0."""
    if density >= density_min:
        return 1.0
    return density / density_min if density_min > 0 else 0.0


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_document_type(image_bytes: bytes) -> tuple[str, float]:
    """Classify a document image into a Philippine ID type using heuristics.

    Args:
        image_bytes: Raw image bytes.

    Returns:
        Tuple of (document_type_str, confidence_float).
        If the best score is below _MIN_CONFIDENCE_THRESHOLD, returns ("unknown", score).
    """
    gray = preprocess_for_matching(image_bytes)
    nparr = np.frombuffer(image_bytes, dtype=np.uint8)
    color_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if color_img is None:
        logger.warning("classify_document_type: could not decode color image; proceeding with zeros.")
        color_img = np.zeros((500, 800, 3), dtype=np.uint8)
    else:
        color_img = cv2.resize(color_img, (800, 500), interpolation=cv2.INTER_AREA)

    characteristics = analyze_document_characteristics(gray, color_img)

    scores: dict[str, float] = {}

    for doc_type, profile in _DOCUMENT_PROFILES.items():
        weights = profile["weights"]

        ar_score = _score_aspect_ratio(
            characteristics["aspect_ratio"],
            profile["aspect_ratio_target"],
            profile["aspect_ratio_tolerance"],
        )
        color_score = _score_color(
            characteristics["color_profile"],
            profile["dominant_color"],
        )
        kp_score = _score_keypoints(
            characteristics["keypoint_count"],
            profile["keypoint_count_min"],
            profile["keypoint_count_max"],
        )
        td_score = _score_text_density(
            characteristics["text_density"],
            profile["text_density_min"],
        )
        # Edge density: treated as a positive signal when above a baseline (~0.05)
        ed_baseline = 0.05
        ed_score = min(1.0, characteristics["edge_density"] / ed_baseline) if ed_baseline > 0 else 0.0

        weighted = (
            weights["aspect_ratio"] * ar_score
            + weights["color"] * color_score
            + weights["keypoint"] * kp_score
            + weights["text_density"] * td_score
            + weights["edge_density"] * ed_score
        )
        scores[doc_type] = round(weighted, 4)
        logger.debug(
            "Type=%s ar=%.3f color=%.3f kp=%.3f td=%.3f ed=%.3f -> score=%.4f",
            doc_type,
            ar_score,
            color_score,
            kp_score,
            td_score,
            ed_score,
            weighted,
        )

    best_type = max(scores, key=lambda k: scores[k])
    best_score = scores[best_type]

    if best_score < _MIN_CONFIDENCE_THRESHOLD:
        logger.info(
            "classify_document_type: best score %.4f below threshold %.2f -> unknown",
            best_score,
            _MIN_CONFIDENCE_THRESHOLD,
        )
        return ("unknown", best_score)

    logger.info("classify_document_type: detected=%s confidence=%.4f", best_type, best_score)
    return (best_type, best_score)


# ---------------------------------------------------------------------------
# Region verification
# ---------------------------------------------------------------------------

def verify_document_regions(image_np: np.ndarray, document_type: str) -> dict:
    """Verify expected spatial regions (photo area, text area, document boundary).

    Args:
        image_np: Preprocessed grayscale image (output of preprocess_for_matching).
        document_type: Expected document type string (used for layout hints).

    Returns:
        dict with keys:
            has_photo_region (bool)  - left-side darker region detected
            has_text_region (bool)   - text density above threshold
            coverage_ratio (float)  - largest detected rectangle / total area
    """
    h, w = image_np.shape[:2]
    total_area = h * w

    # --- Coverage ratio: find largest rectangular contour ---
    blurred = cv2.GaussianBlur(image_np, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    # Dilate to close gaps in document boundary
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_rect_area = 0.0
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        rect_area = cw * ch
        if rect_area > largest_rect_area:
            largest_rect_area = float(rect_area)

    coverage_ratio = largest_rect_area / total_area if total_area > 0 else 0.0

    # --- Photo region detection: left ~25% of image should be darker on average ---
    # On most Philippine IDs the photo is on the left side.
    photo_col_end = int(w * 0.28)
    rest_col_start = int(w * 0.35)

    left_strip = image_np[:, :photo_col_end]
    right_strip = image_np[:, rest_col_start:]

    left_mean = float(np.mean(left_strip)) if left_strip.size > 0 else 128.0
    right_mean = float(np.mean(right_strip)) if right_strip.size > 0 else 128.0

    # Photo region tends to be darker (photo) vs lighter (white background + text)
    # Allow a relaxed threshold: left is noticeably darker than the rest
    has_photo_region: bool = (right_mean - left_mean) >= 8.0

    # --- Text region detection: overall text density > 0.15 ---
    _thresh_val, binary = cv2.threshold(
        image_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    white_pixels = int(np.count_nonzero(binary))
    text_density = white_pixels / total_area if total_area > 0 else 0.0
    has_text_region: bool = text_density > 0.15

    return {
        "has_photo_region": has_photo_region,
        "has_text_region": has_text_region,
        "coverage_ratio": round(coverage_ratio, 4),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_template_matching(image_bytes: bytes, expected_type: str) -> dict:
    """Run full template matching pipeline for a document image.

    Args:
        image_bytes: Raw image bytes from MinIO / upload.
        expected_type: The document type declared by the submitting user/system.

    Returns:
        dict with keys:
            detected_type (str)       - classified document type or "unknown"
            confidence (float)        - classification confidence in [0, 1]
            region_analysis (dict)    - output of verify_document_regions
            matches_expected (bool)   - True if detected == expected OR detected == "unknown"
    """
    detected_type, confidence = classify_document_type(image_bytes)

    gray = preprocess_for_matching(image_bytes)

    nparr = np.frombuffer(image_bytes, dtype=np.uint8)
    color_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if color_img is None:
        logger.warning("run_template_matching: could not decode color image for region analysis.")
        color_img = np.zeros((500, 800, 3), dtype=np.uint8)

    region_analysis = verify_document_regions(gray, expected_type)

    # "unknown" means we could not confidently classify -- do not flag as mismatch
    matches_expected: bool = (detected_type == expected_type) or (detected_type == "unknown")

    result = {
        "detected_type": detected_type,
        "confidence": round(confidence, 4),
        "region_analysis": region_analysis,
        "matches_expected": matches_expected,
    }

    logger.info(
        "run_template_matching: expected=%s detected=%s confidence=%.4f matches=%s",
        expected_type,
        detected_type,
        confidence,
        matches_expected,
    )
    return result
