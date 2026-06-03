"""
System-level document-type requirements for the three integrated platforms.

Each system (identified by its system_id from the API key map) is only
permitted to submit certain document types.  The verification router
enforces this mapping so that, for example, BayaniHub cannot accidentally
(or maliciously) submit an SEC registration certificate.

System map (configured via API_KEY_SYSTEM_MAP in .env):
  system1  →  HopeCard  (charity e-commerce donation platform)
  system2  →  BayaniHub (volunteering & goods management)
  system3  →  Damayan   (DRRM emergency-response system)

"government_id" is an UMBRELLA type accepted by HopeCard and Damayan.
When a document is submitted as government_id, the template matcher
tries to resolve it to a specific subtype (umid, passport, philsys,
drivers_license, prc) before OCR.  The resolved subtype is stored in
document_type_detected while the submitted value stays in document_type.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Document-type allow-lists per system
# ---------------------------------------------------------------------------

SYSTEM_DOCUMENT_MAP: dict[str, list[str]] = {
    # HopeCard — charity e-commerce / donation platform
    # Requires proof of identity + organisational legitimacy
    "system1": [
        "government_id",              # any PH gov't-issued ID (umbrella)
        "organizational_certificate", # DTI / CDA / DSWD registration cert
        "sec_registration",           # SEC Certificate of Registration
    ],

    # BayaniHub — volunteering management & tangible goods
    # Requires transport authority + medical professional credentials
    "system2": [
        "drivers_license",    # LTO Driver's Licence
        "medical_license",    # PRC ID restricted to medical professions
    ],

    # Damayan — DRRM emergency-response for regions
    # Only needs basic proof of identity
    "system3": [
        "government_id",              # any PH gov't-issued ID (umbrella)
    ],
}

# Concrete subtypes that count as "government_id"
GOVERNMENT_ID_SUBTYPES: frozenset[str] = frozenset({
    "umid",
    "passport",
    "philsys",
    "drivers_license",
    "prc",
})

# All known document types (umbrella + subtypes + specialised)
ALL_DOCUMENT_TYPES: frozenset[str] = frozenset({
    "government_id",
    "organizational_certificate",
    "sec_registration",
    "medical_license",
    *GOVERNMENT_ID_SUBTYPES,
})

# Medical professions recognised by the Philippine PRC that qualify for
# medical_license submissions in BayaniHub.
MEDICAL_PROFESSIONS: frozenset[str] = frozenset({
    "PHYSICIAN",
    "MEDICAL DOCTOR",
    "DOCTOR OF MEDICINE",
    "REGISTERED NURSE",
    "NURSE",
    "DENTIST",
    "DENTAL SURGEON",
    "PHARMACIST",
    "PHYSICAL THERAPIST",
    "OCCUPATIONAL THERAPIST",
    "MEDICAL TECHNOLOGIST",
    "RADIOLOGIC TECHNOLOGIST",
    "MIDWIFE",
    "REGISTERED MIDWIFE",
    "OPTOMETRIST",
    "NUTRITIONIST-DIETITIAN",
    "RESPIRATORY THERAPIST",
    "PSYCHOLOGIST",
    "VETERINARIAN",
    "SANITARY ENGINEER",
    "MEDICAL PRACTITIONER",
})

# Keyword fragments used for fuzzy medical profession detection
MEDICAL_KEYWORDS: tuple[str, ...] = (
    "PHYSICIAN", "NURSE", "DOCTOR", "DENTIST", "PHARMA", "THERAPIST",
    "OPTOM", "MIDWIFE", "MEDIC", "RADIOLOG", "NUTRITIONIST", "VETERINARIAN",
    "PSYCHOLOG", "SANITARY ENGINEER",
)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def is_document_type_allowed(system_id: str, document_type: str) -> bool:
    """Return True when *system_id* is permitted to submit *document_type*.

    Handles the umbrella case: a subtype of ``government_id`` is allowed
    for a system that has ``government_id`` in its allow-list.

    Unknown system IDs (not in SYSTEM_DOCUMENT_MAP) are granted access to
    all document types for backwards compatibility.
    """
    allowed = SYSTEM_DOCUMENT_MAP.get(system_id)
    if allowed is None:
        return True  # unknown system → open access (backwards-compat)

    if document_type in allowed:
        return True

    # Subtype of government_id?  Accept if government_id is in the list.
    if document_type in GOVERNMENT_ID_SUBTYPES and "government_id" in allowed:
        return True

    return False


def get_system_label(system_id: str) -> str:
    """Return a human-readable label for a system_id."""
    return {
        "system1": "HopeCard",
        "system2": "BayaniHub",
        "system3": "Damayan",
    }.get(system_id, system_id)


def is_medical_profession(profession: str | None) -> bool:
    """Return True when *profession* belongs to the medical allow-list.

    Performs an exact match against MEDICAL_PROFESSIONS first, then
    falls back to keyword substring matching to handle minor OCR variants
    (e.g. "REGISTERED NURSE RN", "PHYSICIAN (SPECIALIST)").
    """
    if not profession:
        return False

    upper = profession.strip().upper()

    if upper in MEDICAL_PROFESSIONS:
        return True

    # Keyword-based fallback
    return any(kw in upper for kw in MEDICAL_KEYWORDS)


def get_effective_ocr_type(submitted_type: str, detected_type: str) -> str:
    """Resolve the doc type to use for OCR field extraction.

    When the submitted type is the ``government_id`` umbrella and the
    template matcher found a concrete subtype, return the subtype so
    that the correct field parser is applied.
    """
    if (
        submitted_type == "government_id"
        and detected_type
        and detected_type not in ("unknown", "government_id")
        and detected_type in GOVERNMENT_ID_SUBTYPES
    ):
        return detected_type
    return submitted_type
