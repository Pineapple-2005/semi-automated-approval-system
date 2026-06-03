"""OCR service for Philippine ID document verification.

Uses Tesseract + OpenCV to preprocess images, assess quality, extract text,
parse document-type-specific fields, and compute per-field confidence scores.
"""

import logging
import re
from typing import Optional

import cv2
import numpy as np
import pytesseract
from pytesseract import Output

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_DIMENSION = 2000          # pixels – resize target
_RESIZE_TRIGGER = 3000         # pixels – trigger resize above this
_BLUR_THRESHOLD = 100.0        # Laplacian variance; below = blurry
_DARK_THRESHOLD = 50           # mean brightness; below = dark
_OVEREXPOSED_THRESHOLD = 210   # mean brightness; above = overexposed

# ---------------------------------------------------------------------------
# Compiled regex constants shared across all document-type parsers
# ---------------------------------------------------------------------------

_DATE_VAL = r"(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}|\w+ \d{1,2},?\s*\d{4})"

# "Name: DELA CRUZ, JUAN PABLO"
_NAME_RE = re.compile(r"Name[:\s]+([A-Z][A-Z\s,\.]+)(?:\n|$)", re.IGNORECASE)

# "Date of Birth: 01/15/1990" / "DOB: 15 Jan 1990" etc.
_DOB_RE = re.compile(
    rf"(?:Date of Birth|DOB|Birthdate|Birthday)[:\s]+{_DATE_VAL}",
    re.IGNORECASE,
)

# "Expiry: 12/31/2025" / "Valid Until: 31 Dec 2025"
_EXPIRY_RE = re.compile(
    rf"(?:Expiry|Expiration|Valid Until|Exp\.?)[:\s]+{_DATE_VAL}",
    re.IGNORECASE,
)

# "Address: 123 Rizal St, Quezon City" (multi-line block)
_ADDR_RE = re.compile(
    r"(?:Address|Addr)[:\s]+(.+)(?:\n\n|$)",
    re.IGNORECASE | re.DOTALL,
)

# PRC label words to skip when scanning ALL-CAPS lines for the profession field
_PRC_LABEL_WORDS = frozenset({
    "REPUBLIC", "PHILIPPINES", "PROFESSIONAL", "REGULATION", "COMMISSION", "LICENSE",
})

# ---------------------------------------------------------------------------
# Image pre-processing
# ---------------------------------------------------------------------------


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Load, denoise, normalise contrast, and binarise an ID card image.

    Steps
    -----
    1. Decode raw bytes with cv2.imdecode.
    2. Convert to grayscale.
    3. Downscale if either dimension exceeds 3 000 px (longest side → 2 000 px).
    4. Bilateral filter  – smooths noise while keeping sharp text edges.
    5. CLAHE             – compensates for uneven illumination.
    6. Adaptive threshold – produces a clean binary image for Tesseract.

    Args:
        image_bytes: Raw bytes of a JPEG / PNG / BMP document image.

    Returns:
        Preprocessed single-channel (grayscale / binary) NumPy array.

    Raises:
        ValueError: If the bytes cannot be decoded as an image.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("cv2.imdecode returned None – image bytes may be corrupt or unsupported.")

    # --- grayscale ---
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # --- optional resize ---
    h, w = gray.shape[:2]
    if max(h, w) > _RESIZE_TRIGGER:
        scale = _MAX_DIMENSION / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        logger.debug("Resized image from (%d, %d) to (%d, %d)", w, h, new_w, new_h)

    # --- bilateral filter (denoise, preserve edges) ---
    denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    # --- CLAHE ---
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # --- adaptive threshold ---
    binary = cv2.adaptiveThreshold(
        enhanced,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=11,
        C=2,
    )

    return binary


# ---------------------------------------------------------------------------
# Image quality assessment
# ---------------------------------------------------------------------------


def check_image_quality(image_np: np.ndarray) -> dict:
    """Assess blur, brightness, and overexposure of a (possibly colour) image.

    The function works on both grayscale and BGR arrays. A separate
    single-channel copy is used for blur scoring.

    Args:
        image_np: NumPy array – may be grayscale (H×W) or BGR (H×W×3).

    Returns:
        Dictionary with keys:
            ``is_blurry``      (bool)  – Laplacian variance below threshold.
            ``blur_score``     (float) – Laplacian variance (higher = sharper).
            ``is_dark``        (bool)  – Mean pixel value below 50.
            ``is_overexposed`` (bool)  – Mean pixel value above 210.
            ``mean_brightness``(float) – Mean of all pixel values.
    """
    # Grayscale slice for blur detection
    if image_np.ndim == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_np

    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    is_blurry = blur_score < _BLUR_THRESHOLD

    mean_brightness = float(np.mean(image_np))
    is_dark = mean_brightness < _DARK_THRESHOLD
    is_overexposed = mean_brightness > _OVEREXPOSED_THRESHOLD

    logger.debug(
        "Quality check – blur=%.2f blurry=%s brightness=%.2f dark=%s overexposed=%s",
        blur_score, is_blurry, mean_brightness, is_dark, is_overexposed,
    )

    return {
        "is_blurry": is_blurry,
        "blur_score": blur_score,
        "is_dark": is_dark,
        "is_overexposed": is_overexposed,
        "mean_brightness": mean_brightness,
    }


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------


def extract_text_with_confidence(image_np: np.ndarray) -> tuple[str, float]:
    """Run Tesseract OCR and return the full text plus average confidence.

    Uses ``--psm 6`` (assume a uniform block of text) and the ``eng`` language
    model.  Per-word confidence values (filtered to those where Tesseract
    returns conf > 0) are averaged on a 0–100 scale.

    Args:
        image_np: Preprocessed grayscale/binary NumPy array.

    Returns:
        Tuple of (full_text: str, avg_confidence: float 0-100).
        avg_confidence is 0.0 when no valid words are found.
    """
    config = "--psm 6 -l eng"

    full_text: str = pytesseract.image_to_string(image_np, config=config)

    data = pytesseract.image_to_data(image_np, config=config, output_type=Output.DICT)

    confidences = [
        float(c) for c in data.get("conf", [])
        if c is not None and str(c).lstrip("-").isdigit() and int(c) > 0
    ]

    avg_confidence = float(np.mean(confidences)) if confidences else 0.0

    logger.debug(
        "OCR completed – chars=%d words_with_conf=%d avg_conf=%.1f",
        len(full_text), len(confidences), avg_confidence,
    )

    return full_text, avg_confidence


# ---------------------------------------------------------------------------
# Document-type-specific field parsers
# ---------------------------------------------------------------------------

def _clean(text: Optional[str]) -> Optional[str]:
    """Strip extra whitespace from a matched string, or return None."""
    return text.strip() if text else None


def _parse_umid(text: str) -> dict:
    """Extract fields from a Unified Multi-Purpose Identification (UMID) card."""
    fields: dict = {
        "crn": None,
        "name": None,
        "date_of_birth": None,
        "raw_address": None,
    }

    # CRN format: 02-1234567890-1  (2 digits, hyphen, 9 digits, hyphen, 1 digit)
    crn_match = re.search(r"\b(\d{2}-\d{9}-\d)\b", text)
    if crn_match:
        fields["crn"] = crn_match.group(1)

    name_match = _NAME_RE.search(text)
    if name_match:
        fields["name"] = _clean(name_match.group(1))

    dob_match = _DOB_RE.search(text)
    if dob_match:
        fields["date_of_birth"] = _clean(dob_match.group(1))

    addr_match = _ADDR_RE.search(text)
    if addr_match:
        fields["raw_address"] = _clean(addr_match.group(1).replace("\n", " "))

    return fields


def _parse_passport(text: str) -> dict:
    """Extract fields from a Philippine passport (including MRZ)."""
    fields: dict = {
        "passport_number": None,
        "surname": None,
        "given_names": None,
        "date_of_birth": None,
        "expiry_date": None,
    }

    # Passport number: P followed by 7 alphanumeric chars (e.g. PA1234567)
    pno_match = re.search(r"\b(P[A-Z0-9]{7})\b", text)
    if pno_match:
        fields["passport_number"] = pno_match.group(1)

    # MRZ line 2 format: PPHLSURNAME<<GIVEN<<NAMES…  (line 1)
    #   Line 2: passport_no + check + nationality + YYMMDD + check + sex + expiry + …
    mrz_dob = re.search(r"(?<=[A-Z0-9<]{9}\d[A-Z]{3})(\d{6})\d", text)
    if mrz_dob:
        raw_dob = mrz_dob.group(1)  # YYMMDD
        fields["date_of_birth"] = f"{raw_dob[4:6]}/{raw_dob[2:4]}/{raw_dob[:2]}"

    # Expiry from MRZ (7 chars after sex field)
    mrz_expiry = re.search(
        r"(?<=[A-Z0-9<]{9}\d[A-Z]{3}\d{6}\d[MF<])(\d{6})\d", text
    )
    if mrz_expiry:
        raw_exp = mrz_expiry.group(1)
        fields["expiry_date"] = f"{raw_exp[4:6]}/{raw_exp[2:4]}/{raw_exp[:2]}"

    # Surname / Given names from MRZ line 1
    mrz_name = re.search(r"P<PHL([A-Z<]+)<<([A-Z<]+)", text)
    if mrz_name:
        fields["surname"] = mrz_name.group(1).replace("<", " ").strip()
        fields["given_names"] = mrz_name.group(2).replace("<", " ").strip()

    # Fallback: printed name
    if not fields["surname"]:
        name_match = re.search(r"(?:Surname|Last Name)[:\s]+([A-Z][A-Z\s]+)", text, re.IGNORECASE)
        if name_match:
            fields["surname"] = _clean(name_match.group(1))

    if not fields["given_names"]:
        gn_match = re.search(r"(?:Given Names?|First Name)[:\s]+([A-Z][A-Z\s]+)", text, re.IGNORECASE)
        if gn_match:
            fields["given_names"] = _clean(gn_match.group(1))

    return fields


def _parse_drivers_license(text: str) -> dict:
    """Extract fields from a Philippine Land Transportation Office (LTO) driver's licence."""
    fields: dict = {
        "license_number": None,
        "name": None,
        "date_of_birth": None,
        "expiry_date": None,
        "restriction_codes": None,
        "raw_address": None,
    }

    # LTO licence number: XX-XX-XXXXXX
    lic_match = re.search(r"\b(\d{2}-\d{2}-\d{6})\b", text)
    if lic_match:
        fields["license_number"] = lic_match.group(1)

    name_match = _NAME_RE.search(text)
    if name_match:
        fields["name"] = _clean(name_match.group(1))

    dob_match = _DOB_RE.search(text)
    if dob_match:
        fields["date_of_birth"] = _clean(dob_match.group(1))

    exp_match = _EXPIRY_RE.search(text)
    if exp_match:
        fields["expiry_date"] = _clean(exp_match.group(1))

    # Restriction codes: digits 1-8 optionally separated by commas/spaces
    restr_match = re.search(
        r"(?:Restrictions?|Cond\.?|Conditions?)[:\s]+((?:[1-8][,\s]*)+)",
        text, re.IGNORECASE,
    )
    if restr_match:
        codes = re.findall(r"[1-8]", restr_match.group(1))
        fields["restriction_codes"] = codes if codes else None

    addr_match = _ADDR_RE.search(text)
    if addr_match:
        fields["raw_address"] = _clean(addr_match.group(1).replace("\n", " "))

    return fields


def _parse_philsys(text: str) -> dict:
    """Extract fields from a Philippine Statistics Authority (PhilSys / PSN) card."""
    fields: dict = {
        "pcn": None,
        "name": None,
        "sex": None,
        "date_of_birth": None,
        "raw_address": None,
    }

    # PCN format: XXXX-XXXX-XXXX-X (16 digits with hyphens)
    pcn_match = re.search(r"\b(\d{4}-\d{4}-\d{4}-\d)\b", text)
    if pcn_match:
        fields["pcn"] = pcn_match.group(1)

    name_match = _NAME_RE.search(text)
    if name_match:
        fields["name"] = _clean(name_match.group(1))

    sex_match = re.search(r"\b(MALE|FEMALE)\b", text, re.IGNORECASE)
    if sex_match:
        fields["sex"] = sex_match.group(1).upper()

    dob_match = _DOB_RE.search(text)
    if dob_match:
        fields["date_of_birth"] = _clean(dob_match.group(1))

    addr_match = _ADDR_RE.search(text)
    if addr_match:
        fields["raw_address"] = _clean(addr_match.group(1).replace("\n", " "))

    return fields


def _parse_prc(text: str) -> dict:
    """Extract fields from a Professional Regulation Commission (PRC) ID."""
    fields: dict = {
        "prc_number": None,
        "profession": None,
        "name": None,
        "date_of_birth": None,
        "registration_date": None,
        "expiry_date": None,
    }

    # PRC registration number: 7 consecutive digits
    prc_match = re.search(r"\b(\d{7})\b", text)
    if prc_match:
        fields["prc_number"] = prc_match.group(1)

    # Profession: first ALL-CAPS line that is not a form label
    for line in text.splitlines():
        stripped = line.strip()
        if (
            stripped.isupper()
            and len(stripped) > 3
            and not any(kw in stripped for kw in _PRC_LABEL_WORDS)
            and re.match(r"^[A-Z\s]+$", stripped)
        ):
            fields["profession"] = stripped
            break

    name_match = _NAME_RE.search(text)
    if name_match:
        fields["name"] = _clean(name_match.group(1))

    dob_match = _DOB_RE.search(text)
    if dob_match:
        fields["date_of_birth"] = _clean(dob_match.group(1))

    _REG_DATE_RE = re.compile(
        rf"(?:Registration Date|Date of Registration|Reg\.? Date)[:\s]+{_DATE_VAL}",
        re.IGNORECASE,
    )
    reg_match = _REG_DATE_RE.search(text)
    if reg_match:
        fields["registration_date"] = _clean(reg_match.group(1))

    exp_match = _EXPIRY_RE.search(text)
    if exp_match:
        fields["expiry_date"] = _clean(exp_match.group(1))

    return fields


def _parse_medical_license(text: str) -> dict:
    """Extract fields from a PRC Medical Professional Licence.

    Identical layout to a PRC ID but we also capture the profession
    explicitly for medical-profession validation in the flagging layer.
    """
    fields = _parse_prc(text)
    # Add medical-specific alias for the flagging layer
    fields["medical_profession"] = fields.get("profession")
    return fields


def _parse_organizational_certificate(text: str) -> dict:
    """Extract fields from a Philippine organisational registration certificate.

    Supports DTI Business Name Registration, CDA (Cooperative Development
    Authority), and DSWD-registered NGO certificates.
    """
    fields: dict = {
        "organization_name": None,
        "registration_number": None,
        "certificate_type": None,    # DTI | CDA | DSWD | OTHER
        "date_of_registration": None,
        "validity_date": None,
        "authorized_representative": None,
        "registered_address": None,
    }

    # --- Issuing authority / certificate type ---------------------------------
    if re.search(r"\bDTI\b|Department of Trade and Industry", text, re.IGNORECASE):
        fields["certificate_type"] = "DTI"
    elif re.search(r"\bCDA\b|Cooperative Development Authority", text, re.IGNORECASE):
        fields["certificate_type"] = "CDA"
    elif re.search(r"\bDSWD\b|Department of Social Welfare", text, re.IGNORECASE):
        fields["certificate_type"] = "DSWD"
    else:
        fields["certificate_type"] = "OTHER"

    # --- Registration / Business Name number ---------------------------------
    # DTI BN: BN-XXXXXXXXX-XX
    dti_match = re.search(r"\b(BN-\d{9}-\d{2})\b", text, re.IGNORECASE)
    if dti_match:
        fields["registration_number"] = dti_match.group(1)
    else:
        # Generic registration number (6-12 digits, possibly with hyphens)
        reg_match = re.search(
            r"(?:Reg(?:istration)?\.?\s*(?:No\.?|Number)[:\s]+)"
            r"([\dA-Z\-]{4,20})",
            text, re.IGNORECASE,
        )
        if reg_match:
            fields["registration_number"] = _clean(reg_match.group(1))

    # --- Organisation name ---------------------------------------------------
    org_match = re.search(
        r"(?:Name of (?:Business|Organization|NGO|Cooperative)|Business Name)[:\s]+"
        r"([A-Za-z0-9\s,\.\&\-\']+?)(?:\n|$)",
        text, re.IGNORECASE,
    )
    if org_match:
        fields["organization_name"] = _clean(org_match.group(1))

    # --- Dates ---------------------------------------------------------------
    reg_date_match = re.search(
        r"(?:Date of Registration|Registration Date|Issue Date|Date Issued)[:\s]+"
        r"(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}|\w+ \d{1,2},?\s*\d{4})",
        text, re.IGNORECASE,
    )
    if reg_date_match:
        fields["date_of_registration"] = _clean(reg_date_match.group(1))

    valid_match = re.search(
        r"(?:Valid(?:ity)?|Expiry|Expiration|Renewed Until|Valid Until)[:\s]+"
        r"(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}|\w+ \d{1,2},?\s*\d{4})",
        text, re.IGNORECASE,
    )
    if valid_match:
        fields["validity_date"] = _clean(valid_match.group(1))

    # --- Representative ------------------------------------------------------
    rep_match = re.search(
        r"(?:Authorized Representative|Owner|President|Signatory)[:\s]+"
        r"([A-Z][A-Za-z\s,\.]+?)(?:\n|$)",
        text, re.IGNORECASE,
    )
    if rep_match:
        fields["authorized_representative"] = _clean(rep_match.group(1))

    # --- Address -------------------------------------------------------------
    addr_match = re.search(
        r"(?:Business Address|Registered Address|Address)[:\s]+"
        r"(.+?)(?:\n\n|\Z)",
        text, re.IGNORECASE | re.DOTALL,
    )
    if addr_match:
        fields["registered_address"] = _clean(addr_match.group(1).replace("\n", " "))

    return fields


def _parse_sec_registration(text: str) -> dict:
    """Extract fields from a Philippine SEC Certificate of Registration.

    Covers domestic corporations, foundations, partnerships, and OPCs.
    """
    fields: dict = {
        "company_name": None,
        "sec_registration_number": None,
        "date_of_incorporation": None,
        "type_of_corporation": None,    # CORPORATION | FOUNDATION | OPC | PARTNERSHIP
        "registered_address": None,
        "authorized_capital_stock": None,
    }

    # --- SEC registration number: CS/OPC/A/F + 9–12 digits ----------------
    sec_num_match = re.search(
        r"\b((?:CS|OPC|A|F|FS|NA|LA|FC)\d{9,12})\b",
        text, re.IGNORECASE,
    )
    if sec_num_match:
        fields["sec_registration_number"] = sec_num_match.group(1).upper()
    else:
        # Fallback: labelled number
        sec_fallback = re.search(
            r"(?:SEC\s*Reg(?:istration)?\.?\s*(?:No\.?|Number)|Registration No\.?)[:\s]+"
            r"([\dA-Z\-]{6,20})",
            text, re.IGNORECASE,
        )
        if sec_fallback:
            fields["sec_registration_number"] = _clean(sec_fallback.group(1)).upper()

    # --- Company name -------------------------------------------------------
    comp_match = re.search(
        r"(?:Name of Corporation|Corporate Name|Company Name)[:\s]+"
        r"([A-Za-z0-9\s,\.\&\-\']+?)(?:\n|$)",
        text, re.IGNORECASE,
    )
    if comp_match:
        fields["company_name"] = _clean(comp_match.group(1))

    # --- Type of corporation -------------------------------------------------
    if re.search(r"\bFoundation\b", text, re.IGNORECASE):
        fields["type_of_corporation"] = "FOUNDATION"
    elif re.search(r"\bOne Person Corporation\b|\bOPC\b", text, re.IGNORECASE):
        fields["type_of_corporation"] = "OPC"
    elif re.search(r"\bPartnership\b", text, re.IGNORECASE):
        fields["type_of_corporation"] = "PARTNERSHIP"
    elif re.search(r"\bCorporation\b|\bInc\b|\bIncorporated\b", text, re.IGNORECASE):
        fields["type_of_corporation"] = "CORPORATION"

    # --- Date of incorporation -----------------------------------------------
    inc_match = re.search(
        r"(?:Date of Incorporation|Date of Registration|Incorporated)[:\s]+"
        r"(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}|\w+ \d{1,2},?\s*\d{4})",
        text, re.IGNORECASE,
    )
    if inc_match:
        fields["date_of_incorporation"] = _clean(inc_match.group(1))

    # --- Registered address --------------------------------------------------
    addr_match = re.search(
        r"(?:Principal Office|Registered Address|Address)[:\s]+(.+?)(?:\n\n|\Z)",
        text, re.IGNORECASE | re.DOTALL,
    )
    if addr_match:
        fields["registered_address"] = _clean(addr_match.group(1).replace("\n", " "))

    # --- Authorised capital stock -------------------------------------------
    cap_match = re.search(
        r"(?:Authorized Capital Stock|Capital Stock)[:\s]+PHP?\s*([\d,\.]+)",
        text, re.IGNORECASE,
    )
    if cap_match:
        fields["authorized_capital_stock"] = _clean(cap_match.group(1))

    return fields


def _parse_government_id(text: str) -> dict:
    """Fallback parser for the government_id umbrella type.

    Tries UMID, passport, PhilSys, driver's licence, and PRC parsers in
    sequence and returns the result with the most non-None fields.
    """
    candidates = {
        "umid": _parse_umid(text),
        "passport": _parse_passport(text),
        "philsys": _parse_philsys(text),
        "drivers_license": _parse_drivers_license(text),
        "prc": _parse_prc(text),
    }
    best_type, best_fields = max(
        candidates.items(),
        key=lambda kv: sum(1 for v in kv[1].values() if v is not None),
    )
    logger.debug(
        "government_id fallback parser: best match = %s (%d fields found)",
        best_type,
        sum(1 for v in best_fields.values() if v is not None),
    )
    best_fields["_detected_subtype"] = best_type
    return best_fields


# ---------------------------------------------------------------------------
# Field dispatcher
# ---------------------------------------------------------------------------

_PARSERS = {
    "umid": _parse_umid,
    "passport": _parse_passport,
    "drivers_license": _parse_drivers_license,
    "philsys": _parse_philsys,
    "prc": _parse_prc,
    "medical_license": _parse_medical_license,
    "organizational_certificate": _parse_organizational_certificate,
    "sec_registration": _parse_sec_registration,
    "government_id": _parse_government_id,
}


def parse_philippine_id_fields(raw_text: str, document_type: str) -> dict:
    """Dispatch raw OCR text to the correct document-type parser.

    Args:
        raw_text:      Full text returned by Tesseract.
        document_type: One of ``umid | passport | drivers_license | philsys | prc``.
                       Unrecognised types produce an empty dict.

    Returns:
        Dictionary of field names → extracted values (None when not found).
        All parsers also attempt to extract ``raw_address`` where applicable.
    """
    doc_type_lower = document_type.lower().strip()
    parser = _PARSERS.get(doc_type_lower)
    if parser is None:
        logger.warning("No parser registered for document type '%s'", document_type)
        return {}
    return parser(raw_text)


# ---------------------------------------------------------------------------
# Per-field confidence scoring
# ---------------------------------------------------------------------------


def calculate_field_confidence(extracted_data: dict, ocr_confidence: float) -> dict:
    """Compute a 0–1 confidence score for every field in *extracted_data*.

    Scoring logic
    -------------
    * If the field value is None → 0.0.
    * Otherwise the base score is ``ocr_confidence / 100``.
    * A small bonus (0.05) is awarded when the value looks like a well-formed
      pattern (date, ID number, etc.) and a small penalty (0.10) is applied
      when the value is suspiciously short (< 3 characters).

    Args:
        extracted_data: Dict returned by :func:`parse_philippine_id_fields`.
        ocr_confidence: Average Tesseract word confidence on the 0-100 scale.

    Returns:
        Dict mapping each field name to a float in [0.0, 1.0].
    """
    base = max(0.0, min(1.0, ocr_confidence / 100.0))

    # Simple heuristics to up/downweight specific fields
    _DATE_RE = re.compile(r"\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}|\w+ \d{1,2},?\s*\d{4}")
    _ID_RE = re.compile(r"\d[\d\-]+\d")  # contains digits and hyphens

    scores: dict = {}
    for field, value in extracted_data.items():
        if value is None:
            scores[field] = 0.0
            continue

        score = base
        value_str = str(value).strip()

        # Penalty for very short values
        if len(value_str) < 3:
            score -= 0.10

        # Bonus for well-structured values (date or ID number pattern)
        if _DATE_RE.fullmatch(value_str) or _ID_RE.search(value_str):
            score += 0.05

        scores[field] = round(max(0.0, min(1.0, score)), 4)

    return scores


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_ocr(image_bytes: bytes, document_type: str) -> dict:
    """Full OCR pipeline for a Philippine ID image.

    Execution order
    ---------------
    1. :func:`preprocess_image`           – clean and binarise.
    2. :func:`check_image_quality`        – blur / brightness metrics.
    3. :func:`extract_text_with_confidence` – Tesseract OCR.
    4. :func:`parse_philippine_id_fields` – structured field extraction.
    5. :func:`calculate_field_confidence` – per-field confidence scores.

    Args:
        image_bytes:   Raw bytes of the uploaded document image.
        document_type: One of ``umid | passport | drivers_license | philsys | prc``.

    Returns:
        On success::

            {
                "extracted_data":   dict,   # parsed fields
                "confidence_scores": dict,  # field → float 0-1
                "raw_text":         str,    # full Tesseract output
                "ocr_confidence":   float,  # avg word confidence 0-100
                "quality_metrics":  dict,   # blur / brightness info
            }

        On error::

            {
                "extracted_data":   {},
                "confidence_scores": {},
                "error":            str,
            }
    """
    try:
        logger.info("Starting OCR pipeline for document_type='%s'", document_type)

        processed = preprocess_image(image_bytes)
        quality_metrics = check_image_quality(processed)

        raw_text, ocr_confidence = extract_text_with_confidence(processed)
        logger.debug("raw_text preview: %s", raw_text[:200].replace("\n", " "))

        extracted_data = parse_philippine_id_fields(raw_text, document_type)
        confidence_scores = calculate_field_confidence(extracted_data, ocr_confidence)

        logger.info(
            "OCR pipeline complete – fields=%d ocr_conf=%.1f",
            len(extracted_data), ocr_confidence,
        )

        return {
            "extracted_data": extracted_data,
            "confidence_scores": confidence_scores,
            "raw_text": raw_text,
            "ocr_confidence": ocr_confidence,
            "quality_metrics": quality_metrics,
        }

    except Exception as exc:  # noqa: BLE001
        logger.exception("OCR pipeline failed: %s", exc)
        return {
            "extracted_data": {},
            "confidence_scores": {},
            "error": str(exc),
        }
