import json
from typing import Any, Optional
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # Database — either set DATABASE_URL or the individual DB_* vars.
    # Individual vars are preferred: the password is used as a plain string
    # so special characters, spaces, etc. never cause URL-parsing problems.
    DATABASE_URL: Optional[str] = None
    DB_HOST: Optional[str] = None
    DB_PORT: int = 5432
    DB_USER: Optional[str] = None
    DB_PASSWORD: Optional[str] = None
    DB_NAME: str = "postgres"

    # Redis / Celery
    REDIS_URL: str

    # MinIO
    MINIO_ENDPOINT: str
    MINIO_ACCESS_KEY: str
    MINIO_SECRET_KEY: str
    MINIO_BUCKET: str
    MINIO_SECURE: bool = False  # set True only if MinIO is behind HTTPS

    # Auth — raw JSON string coming from the environment
    # Using Any so pydantic-settings v2 doesn't attempt its own JSON coercion
    # before the model_validator runs; the validator handles both formats.
    API_KEY_SYSTEM_MAP: Any = {}
    ALLOWED_API_KEYS: Any = []

    # Limits
    MAX_IMAGE_SIZE_MB: float = 10.0
    WEBHOOK_TIMEOUT_SECONDS: int = 30

    # Debug mode — enables verbose error messages in responses
    DEBUG: bool = False

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @model_validator(mode="before")
    @classmethod
    def _parse_json_fields(cls, values: dict) -> dict:
        """Parse API_KEY_SYSTEM_MAP from a JSON string when needed."""
        raw_map = values.get("API_KEY_SYSTEM_MAP")
        if isinstance(raw_map, str):
            try:
                values["API_KEY_SYSTEM_MAP"] = json.loads(raw_map)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"API_KEY_SYSTEM_MAP must be valid JSON: {exc}"
                ) from exc

        raw_keys = values.get("ALLOWED_API_KEYS")
        if isinstance(raw_keys, str):
            try:
                values["ALLOWED_API_KEYS"] = json.loads(raw_keys)
            except json.JSONDecodeError:
                # Fallback: treat as comma-separated string
                values["ALLOWED_API_KEYS"] = [
                    k.strip() for k in raw_keys.split(",") if k.strip()
                ]

        return values

    @model_validator(mode="after")
    def _derive_allowed_keys(self) -> "Settings":
        """If ALLOWED_API_KEYS is empty, derive it from API_KEY_SYSTEM_MAP keys."""
        if not self.ALLOWED_API_KEYS and self.API_KEY_SYSTEM_MAP:
            self.ALLOWED_API_KEYS = list(self.API_KEY_SYSTEM_MAP.keys())
        return self

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def get_system_id_for_key(self, api_key: str) -> Optional[str]:
        """Return the system_id mapped to *api_key*, or None if not found."""
        return self.API_KEY_SYSTEM_MAP.get(api_key)

    def validate_api_key(self, api_key: str) -> tuple[bool, Optional[str]]:
        """Return (is_valid, system_id). system_id is None when key is invalid."""
        if api_key not in self.ALLOWED_API_KEYS:
            return False, None
        system_id = self.get_system_id_for_key(api_key)
        return True, system_id


# Singleton — import this throughout the application
settings = Settings()
