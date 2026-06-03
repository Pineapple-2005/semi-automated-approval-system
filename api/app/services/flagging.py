"""
Flag generation service for Philippine ID document verification.

CRITICAL RULE: All jobs require human review. The pipeline never approves or
rejects automatically. Flags and approval signals are evidence for the reviewer.

Both collections use a structured colon-separated key=value format so the admin
UI can parse context:
    "flag_type:detail=value:detail=value"

Negative flag examples (concerns for the reviewer):
    "name_mismatch:similarity=72%:extracted=DELA CRUZ JUAN:expected=JUAN DELA CRUZ"
    "blurry_image:laplacian=45.2"
    "expiry_check:expired=2023-01-15"

Positive signal examples (confirmations for the reviewer):
    "name_match:similarity=95%:extracted=JUAN DELA CRUZ:expected=JUAN DELA CRUZ"
    "good_image_quality:sharpness=245.3:brightness=128.0"
    "document_valid:expiry=2027-06-01"
"""

from __future__ import annotations

import re
import string
from datetime import date
from statistics import mean
from typing import Optional

from dateutil import parser as dateutil_parser
from rapidfuzz import fuzz

from app.services.system_requirements import MEDICAL_PROFESSIONS, is_medical_profession

# ---------------------------------------------------------------------------
# Required fields per document type
# ---------------------------------------------------------------------------

REQUIRED_FIELDS: dict[str, list[str]] = {
    # Government-issued personal IDs
    "umid":            ["crn", "name", "date_of_birth"],
    "passport":        ["passport_number", "surname", "given_names", "date_of_birth", "expiry_date"],
    "drivers_license": ["license_number", "name", "date_of_birth", "expiry_date"],
    "philsys":         ["pcn", "name", "date_of_birth"],
    "prc":             ["prc_number", "name", "date_of_birth", "expiry_date"],
    # Umbrella: resolved to a subtype before flagging; minimal check here
    "government_id":   ["name", "date_of_birth"],
    # Specialised
    "medical_license": ["prc_number", "profession", "name", "date_of_birth", "expiry_date"],
    "organizational_certificate": [
        "organization_name", "registration_number", "date_of_registration",
    ],
    "sec_registration": [
        "company_name", "sec_registration_number", "date_of_incorporation",
    ],
}

# ---------------------------------------------------------------------------
# Common Filipino name abbreviation expansions
# ---------------------------------------------------------------------------

_ABBREVIATION_MAP: dict[str, str] = {
    r"\bma\.?": "maria",
    r"\bdr\.?": "doctor",
    r"\bjr\.?": "junior",
    r"\bsr\.?": "senior",
    r"\bst\.?": "saint",
    r"\bgen\.?": "general",
}


# ---------------------------------------------------------------------------
# Name normalisation
# ---------------------------------------------------------------------------

def normalize_name(name: str) -> str:
    """Lowercase, expand abbreviations, strip punctuation except hyphens."""
    if not name:
        return ""
    normalized = name.lower().strip()
    for pattern, replacement in _ABBREVIATION_MAP.items():
        normalized = re.sub(pattern, replacement, normalized)
    allowed = set(string.ascii_lowercase + string.digits + " -")
    normalized = "".join(ch for ch in normalized if ch in allowed)
    return re.sub(r"\s+", " ", normalized).strip()


# ---------------------------------------------------------------------------
# Name comparison
# ---------------------------------------------------------------------------

def compare_names(
    extracted_name: Optional[str],
    expected_name: str,
) -> tuple[bool, float]:
    """Return (is_match, similarity 0-1).  None extracted → (True, 0.0)."""
    if not extracted_name:
        return (True, 0.0)
    norm_e = normalize_name(extracted_name)
    norm_x = normalize_name(expected_name)
    similarity = fuzz.token_set_ratio(norm_e, norm_x)
    return (similarity >= 80, similarity / 100.0)


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

def parse_date_flexible(date_str: str) -> Optional[date]:
    """Parse Philippine date strings including MRZ YYMMDD format."""
    if not date_str:
        return None
    date_str = date_str.strip()
    # MRZ YYMMDD
    if re.fullmatch(r"\d{6}", date_str):
        try:
            yy, mm, dd = int(date_str[:2]), int(date_str[2:4]), int(date_str[4:6])
            yyyy = (1900 + yy) if yy >= 30 else (2000 + yy)
            return date(yyyy, mm, dd)
        except ValueError:
            pass
    try:
        return dateutil_parser.parse(date_str, dayfirst=True).date()
    except (ValueError, OverflowError):
        pass
    for fmt in ("%m/%d/%Y", "%d-%m-%Y", "%Y-%m-%d"):
        try:
            from datetime import datetime
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    return None


def compare_dob(extracted: Optional[str], expected: Optional[str]) -> bool:
    """True = no mismatch (or unverifiable).  False = dates differ → flag."""
    if not extracted or not expected:
        return True
    d_ext = parse_date_flexible(extracted)
    d_exp = parse_date_flexible(expected)
    if d_ext is None or d_exp is None:
        return True
    return d_ext == d_exp


def check_document_expiry(extracted_data: dict) -> Optional[str]:
    """Return ISO expiry string if the document is expired, else None."""
    raw = extracted_data.get("expiry_date") or extracted_data.get("validity_date")
    if not raw:
        return None
    expiry = parse_date_flexible(str(raw))
    if expiry is None:
        return None
    return expiry.isoformat() if expiry < date.today() else None


# ---------------------------------------------------------------------------
# Quality flags  (image-level)
# ---------------------------------------------------------------------------

def evaluate_quality_flags(quality_metrics: dict) -> list[str]:
    """Return image-quality flags with numeric detail for reviewer context."""
    flags: list[str] = []
    if quality_metrics.get("is_blurry"):
        score = quality_metrics.get("blur_score", 0)
        flags.append(f"blurry_image:laplacian={score:.1f}:threshold=100")
    if quality_metrics.get("is_dark"):
        brightness = quality_metrics.get("mean_brightness", 0)
        flags.append(f"poor_lighting:brightness={brightness:.1f}:reason=too_dark")
    elif quality_metrics.get("is_overexposed"):
        brightness = quality_metrics.get("mean_brightness", 0)
        flags.append(f"poor_lighting:brightness={brightness:.1f}:reason=overexposed")
    return flags


# ---------------------------------------------------------------------------
# Extraction / OCR flags
# ---------------------------------------------------------------------------

def evaluate_extraction_flags(
    extracted_data: dict,
    confidence_scores: dict,
    document_type: str,
) -> list[str]:
    """Flag missing required fields and low OCR confidence."""
    flags: list[str] = []
    effective_type = document_type
    # When government_id umbrella was resolved, the OCR subtype may be stored
    detected_sub = extracted_data.get("_detected_subtype")
    if detected_sub:
        effective_type = detected_sub

    for field in REQUIRED_FIELDS.get(effective_type, []):
        if not extracted_data.get(field):
            flags.append(f"missing_required_field:{field}:doc_type={effective_type}")

    if confidence_scores and extracted_data:
        scores = [v for k, v in confidence_scores.items() if not k.startswith("_")]
        if scores:
            avg = mean(scores)
            if avg < 0.6:
                flags.append(f"low_ocr_confidence:avg={avg:.0%}:threshold=60%")

    return flags


# ---------------------------------------------------------------------------
# Document-type-specific semantic flags
# ---------------------------------------------------------------------------

def evaluate_medical_license_flags(extracted_data: dict) -> list[str]:
    """Ensure the profession on a medical_license is a recognised medical role."""
    profession = (
        extracted_data.get("medical_profession")
        or extracted_data.get("profession")
    )
    if not profession:
        # Absence is caught by missing_required_field
        return []
    if not is_medical_profession(profession):
        return [
            f"medical_profession_not_licensed"
            f":detected_profession={profession}"
            f":expected=PRC_medical_profession"
        ]
    return []


def evaluate_sec_registration_flags(extracted_data: dict) -> list[str]:
    """Validate SEC registration number format."""
    flags: list[str] = []
    sec_num = extracted_data.get("sec_registration_number")
    if sec_num:
        # Expected: CS/OPC/A/F prefix + 9-12 digits
        if not re.match(r"^(?:CS|OPC|A|F|FS|NA|LA|FC)\d{9,12}$", sec_num, re.IGNORECASE):
            flags.append(
                f"invalid_sec_number_format:extracted={sec_num}"
                f":expected=PREFIX+9-12digits"
            )
    return flags


def evaluate_org_certificate_flags(extracted_data: dict) -> list[str]:
    """Validate organisational certificate registration number format."""
    flags: list[str] = []
    reg_num = extracted_data.get("registration_number")
    cert_type = extracted_data.get("certificate_type", "OTHER")
    if reg_num and cert_type == "DTI":
        if not re.match(r"^BN-\d{9}-\d{2}$", reg_num, re.IGNORECASE):
            flags.append(
                f"invalid_dti_registration_format:extracted={reg_num}"
                f":expected=BN-XXXXXXXXX-XX"
            )
    return flags


# ---------------------------------------------------------------------------
# Organisation / company name comparison
# ---------------------------------------------------------------------------

def compare_org_name(
    extracted_data: dict,
    expected_name: str,
    document_type: str,
) -> list[str]:
    """For organisational documents compare organisation/company name."""
    if document_type == "sec_registration":
        extracted = extracted_data.get("company_name")
    elif document_type == "organizational_certificate":
        extracted = extracted_data.get("organization_name")
    else:
        return []

    if not extracted or not expected_name:
        return []

    is_match, sim = compare_names(extracted, expected_name)
    if not is_match:
        return [
            f"org_name_mismatch"
            f":similarity={sim:.0%}"
            f":extracted={extracted[:60]}"
            f":expected={expected_name[:60]}"
        ]
    return []


# ---------------------------------------------------------------------------
# Private helpers used only by generate_flags
# ---------------------------------------------------------------------------

_PERSONAL_TYPES = frozenset({
    "umid", "passport", "drivers_license", "philsys",
    "prc", "medical_license", "government_id",
})


def _name_flags(
    extracted_data: dict,
    expected_name: str,
    expected_type: str,
) -> list[str]:
    """Return name / org-name mismatch flags (step 4 in generate_flags)."""
    if expected_type in _PERSONAL_TYPES:
        extracted_name: Optional[str] = (
            extracted_data.get("name") or extracted_data.get("surname")
        )
        is_match, similarity = compare_names(extracted_name, expected_name)
        if not is_match:
            return [
                f"name_mismatch"
                f":similarity={similarity:.0%}"
                f":extracted={str(extracted_name or '')[:60]}"
                f":expected={expected_name[:60]}"
            ]
        return []
    return compare_org_name(extracted_data, expected_name, expected_type)


def _type_mismatch_flag(detected_type: str, expected_type: str) -> list[str]:
    """Return a document_type_mismatch flag when the classifier disagrees."""
    from app.services.system_requirements import GOVERNMENT_ID_SUBTYPES
    if detected_type in ("unknown", expected_type):
        return []
    # government_id umbrella: any recognised gov't-ID subtype is acceptable
    if expected_type == "government_id" and detected_type in GOVERNMENT_ID_SUBTYPES:
        return []
    return [
        f"document_type_mismatch"
        f":expected={expected_type}"
        f":detected={detected_type}"
    ]


# ---------------------------------------------------------------------------
# Master flag generator
# ---------------------------------------------------------------------------

def generate_flags(
    extracted_data: dict,
    confidence_scores: dict,
    quality_metrics: dict,
    detected_type: str,
    expected_type: str,
    expected_name: str,
    expected_dob: Optional[str],
    region_analysis: Optional[dict] = None,
) -> list[str]:
    """
    Return an ordered, deduplicated list of flag strings for a verification job.

    Flag evaluation order:
    1.  Image quality          (blurry_image, poor_lighting)
    2.  Extraction / OCR       (missing_required_field:*, low_ocr_confidence)
    3.  Doc-type semantic      (medical_profession_not_licensed, invalid_sec_*…)
    4.  Name / org comparison  (name_mismatch, org_name_mismatch)
    5.  Date-of-birth          (dob_mismatch)
    6.  Expiry                 (expiry_check)
    7.  Document type          (document_type_mismatch)
    8.  Coverage               (partial_document)
    9.  Unrecognised format    (unrecognized_format)

    CRITICAL: flags → needs_review. Zero flags → auto_approved.
    Rejection is ALWAYS a manual reviewer decision.
    """
    flags: list[str] = []

    # 1. Image quality
    flags.extend(evaluate_quality_flags(quality_metrics))

    # 2. Extraction completeness + OCR confidence
    flags.extend(evaluate_extraction_flags(extracted_data, confidence_scores, expected_type))

    # 3. Document-type-specific semantic validation
    if expected_type == "medical_license":
        flags.extend(evaluate_medical_license_flags(extracted_data))
    elif expected_type == "sec_registration":
        flags.extend(evaluate_sec_registration_flags(extracted_data))
    elif expected_type == "organizational_certificate":
        flags.extend(evaluate_org_certificate_flags(extracted_data))

    # 4. Name / org-name comparison
    flags.extend(_name_flags(extracted_data, expected_name, expected_type))

    # 5. Date-of-birth mismatch
    if not compare_dob(extracted_data.get("date_of_birth"), expected_dob):
        ext_dob = extracted_data.get("date_of_birth", "")
        flags.append(
            f"dob_mismatch"
            f":extracted={ext_dob}"
            f":expected={expected_dob or ''}"
        )

    # 6. Expiry check
    expired_on = check_document_expiry(extracted_data)
    if expired_on:
        flags.append(f"expiry_check:expired={expired_on}")

    # 7. Document type mismatch
    flags.extend(_type_mismatch_flag(detected_type, expected_type))

    # 8. Partial document coverage
    if region_analysis is not None:
        coverage = region_analysis.get("coverage_ratio", 1.0)
        if coverage < 0.6:
            flags.append(f"partial_document:coverage={coverage:.0%}:threshold=60%")

    # 9. Unrecognised format
    if detected_type == "unknown":
        flags.append("unrecognized_format:classifier_confidence=low")

    # Deduplicate preserving insertion order
    return list(dict.fromkeys(flags))


# ---------------------------------------------------------------------------
# Approval signal helpers  (positive counterparts to the flag evaluators)
# ---------------------------------------------------------------------------


def evaluate_quality_signals(quality_metrics: dict) -> list[str]:
    """Signal good image quality when none of the quality issues are present."""
    if (
        not quality_metrics.get("is_blurry")
        and not quality_metrics.get("is_dark")
        and not quality_metrics.get("is_overexposed")
    ):
        blur = quality_metrics.get("blur_score", 0)
        brightness = quality_metrics.get("mean_brightness", 0)
        return [f"good_image_quality:sharpness={blur:.1f}:brightness={brightness:.1f}"]
    return []


def evaluate_extraction_signals(
    extracted_data: dict,
    confidence_scores: dict,
    expected_type: str,
) -> list[str]:
    """Signal complete extraction and acceptable OCR confidence."""
    signals: list[str] = []

    effective_type = extracted_data.get("_detected_subtype") or expected_type
    required = REQUIRED_FIELDS.get(effective_type, [])
    if required and all(extracted_data.get(f) for f in required):
        signals.append(
            f"all_fields_extracted:doc_type={effective_type}:field_count={len(required)}"
        )

    if confidence_scores and extracted_data:
        scores = [v for k, v in confidence_scores.items() if not k.startswith("_")]
        if scores:
            avg = mean(scores)
            if avg >= 0.8:
                signals.append(f"high_ocr_confidence:avg={avg:.0%}")
            elif avg >= 0.6:
                signals.append(f"acceptable_ocr_confidence:avg={avg:.0%}")

    return signals


def evaluate_doc_type_semantic_signals(
    extracted_data: dict,
    expected_type: str,
) -> list[str]:
    """Signal validated doc-type-specific fields (medical profession, SEC/DTI numbers)."""
    signals: list[str] = []

    if expected_type == "medical_license":
        profession = (
            extracted_data.get("medical_profession") or extracted_data.get("profession")
        )
        if profession and is_medical_profession(profession):
            signals.append(f"licensed_medical_profession:profession={profession}")

    elif expected_type == "sec_registration":
        sec_num = extracted_data.get("sec_registration_number")
        if sec_num and re.match(
            r"^(?:CS|OPC|A|F|FS|NA|LA|FC)\d{9,12}$", sec_num, re.IGNORECASE
        ):
            signals.append(f"valid_sec_number_format:extracted={sec_num}")

    elif expected_type == "organizational_certificate":
        reg_num = extracted_data.get("registration_number")
        cert_type = extracted_data.get("certificate_type", "OTHER")
        if reg_num and cert_type == "DTI" and re.match(
            r"^BN-\d{9}-\d{2}$", reg_num, re.IGNORECASE
        ):
            signals.append(f"valid_dti_registration_format:extracted={reg_num}")

    return signals


def _org_name_signal(
    extracted_data: dict,
    expected_name: str,
    expected_type: str,
) -> list[str]:
    """Return an org_name_match signal for organisational document types."""
    field = "company_name" if expected_type == "sec_registration" else "organization_name"
    extracted_org = extracted_data.get(field)
    if not extracted_org:
        return []
    is_match, similarity = compare_names(extracted_org, expected_name)
    if not is_match:
        return []
    return [
        f"org_name_match:similarity={similarity:.0%}"
        f":extracted={str(extracted_org)[:60]}"
    ]


def _name_signals(
    extracted_data: dict,
    expected_name: str,
    expected_type: str,
) -> list[str]:
    """Return name/org-name match signals (step 4 in generate_approval_signals)."""
    if not expected_name:
        return []
    if expected_type in _PERSONAL_TYPES:
        extracted_name: Optional[str] = (
            extracted_data.get("name") or extracted_data.get("surname")
        )
        if not extracted_name:
            return []
        is_match, similarity = compare_names(extracted_name, expected_name)
        if not is_match:
            return []
        return [
            f"name_match:similarity={similarity:.0%}"
            f":extracted={str(extracted_name)[:60]}"
            f":expected={expected_name[:60]}"
        ]
    if expected_type in ("sec_registration", "organizational_certificate"):
        return _org_name_signal(extracted_data, expected_name, expected_type)
    return []


def evaluate_temporal_signals(
    extracted_data: dict,
    expected_dob: Optional[str],
) -> list[str]:
    """Signal matching DOB and non-expired document."""
    signals: list[str] = []

    if expected_dob and extracted_data.get("date_of_birth"):
        if compare_dob(extracted_data.get("date_of_birth"), expected_dob):
            signals.append(
                f"dob_match"
                f":extracted={extracted_data.get('date_of_birth', '')}"
                f":expected={expected_dob}"
            )

    raw = extracted_data.get("expiry_date") or extracted_data.get("validity_date")
    if raw:
        expiry = parse_date_flexible(str(raw))
        if expiry and expiry >= date.today():
            signals.append(f"document_valid:expiry={expiry.isoformat()}")

    return signals


def _type_confirmed_signal(detected_type: str, expected_type: str) -> list[str]:
    """Return a document_type_confirmed signal when the classifier agrees."""
    if detected_type == "unknown":
        return []
    from app.services.system_requirements import GOVERNMENT_ID_SUBTYPES
    type_ok = (
        detected_type == expected_type
        or (expected_type == "government_id" and detected_type in GOVERNMENT_ID_SUBTYPES)
    )
    if not type_ok:
        return []
    return [f"document_type_confirmed:detected={detected_type}:expected={expected_type}"]


# ---------------------------------------------------------------------------
# Master approval signal generator
# ---------------------------------------------------------------------------


def generate_approval_signals(
    extracted_data: dict,
    confidence_scores: dict,
    quality_metrics: dict,
    detected_type: str,
    expected_type: str,
    expected_name: str,
    expected_dob: Optional[str],
    region_analysis: Optional[dict] = None,
) -> list[str]:
    """
    Return an ordered, deduplicated list of positive confirmation signals.

    These complement flags so a human reviewer sees both what passed and
    what needs attention.  Signal evaluation mirrors generate_flags order:
    1.  Image quality          (good_image_quality)
    2.  Extraction / OCR       (all_fields_extracted, high/acceptable_ocr_confidence)
    3.  Doc-type semantic      (licensed_medical_profession, valid_sec/dti_format)
    4.  Name / org match       (name_match, org_name_match)
    5.  Date-of-birth match    (dob_match)
    6.  Expiry valid           (document_valid)
    7.  Document type match    (document_type_confirmed)
    8.  Coverage OK            (full_document_visible)
    """
    signals: list[str] = []

    signals.extend(evaluate_quality_signals(quality_metrics))
    signals.extend(evaluate_extraction_signals(extracted_data, confidence_scores, expected_type))
    signals.extend(evaluate_doc_type_semantic_signals(extracted_data, expected_type))
    signals.extend(_name_signals(extracted_data, expected_name, expected_type))
    signals.extend(evaluate_temporal_signals(extracted_data, expected_dob))
    signals.extend(_type_confirmed_signal(detected_type, expected_type))

    if region_analysis is not None:
        coverage = region_analysis.get("coverage_ratio", 1.0)
        if coverage >= 0.6:
            signals.append(f"full_document_visible:coverage={coverage:.0%}")

    return list(dict.fromkeys(signals))
