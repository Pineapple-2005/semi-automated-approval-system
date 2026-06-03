# =============================================================================
# Philippine ID Document Verification API — Dockerfile
# =============================================================================
FROM python:3.11-slim

# ---------------------------------------------------------------------------
# System dependencies
# ---------------------------------------------------------------------------
# tesseract-ocr + tesseract-ocr-eng : OCR engine and English language data
# libgl1, libglib2.0-0              : OpenCV runtime requirements
# libsm6, libxrender1, libxext6     : Additional OpenCV/X11 dependencies
# libgomp1                          : OpenMP runtime (used by OpenCV internals)
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgomp1 \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Working directory
# ---------------------------------------------------------------------------
WORKDIR /app

# ---------------------------------------------------------------------------
# Python dependencies
# Install before copying application code to maximise Docker layer caching.
# ---------------------------------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Application source code
# ---------------------------------------------------------------------------
COPY app/ ./app/

# ORB template images and document-type configuration used at runtime.
COPY templates_config/ ./templates_config/

# ---------------------------------------------------------------------------
# Non-root user (security hardening)
# ---------------------------------------------------------------------------
RUN useradd --no-create-home --shell /bin/false appuser \
  && chown -R appuser:appuser /app
USER appuser

# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
