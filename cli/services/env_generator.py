"""Generate .env.local from .env.example based on CLI config choices."""

from __future__ import annotations

import re
import secrets
from pathlib import Path

from cli.models.config import InferenceBackend, MolebieConfig, SetupType
from cli.services.config_manager import get_project_root


# Default model names per backend
OLLAMA_THINKING_MODEL = "qwen3:8b"
OLLAMA_INSTANT_MODEL = "qwen3:4b"
MLX_THINKING_MODEL = "mlx-community/Qwen3.5-9B-MLX-4bit"
MLX_INSTANT_MODEL = "mlx-community/Qwen3.5-4B-4bit"


# Shared regex for KEY=VALUE lines
_KV_PATTERN = re.compile(r"^([A-Z_][A-Z0-9_]*)=(.*)")

# Section header patterns (matches the ── borders and ALL-CAPS titles in .env.example)
_SECTION_BORDER = re.compile(r"^#\s*[─━═]+")
_SECTION_TITLE = re.compile(r"^#\s+([A-Z][A-Z0-9 &/()\-.,]+)\s*$")


def get_env_key(key: str) -> str | None:
    """Read a single key's value from .env.local. Returns None if not found."""
    root = get_project_root()
    env_path = root / ".env.local"
    if not env_path.exists():
        return None
    pattern = re.compile(rf"^{re.escape(key)}=(.*)")
    for line in env_path.read_text().splitlines():
        m = pattern.match(line)
        if m:
            return m.group(1)
    return None


def get_valid_keys() -> set[str]:
    """Return the set of all KEY names defined in .env.example."""
    root = get_project_root()
    template = root / ".env.example"
    if not template.exists():
        return set()
    keys: set[str] = set()
    for line in template.read_text().splitlines():
        m = _KV_PATTERN.match(line)
        if m:
            keys.add(m.group(1))
    return keys


def parse_env_file(env_path: Path) -> list[tuple[str, str, str | None]]:
    """Parse a .env file into structured entries.

    Returns a list of tuples:
        ("section", "SECTION TITLE", None)
        ("var", "KEY", "value")
        ("comment", "# text", None)       -- blank/comment lines (skipped in display)
    """
    if not env_path.exists():
        return []
    lines = env_path.read_text().splitlines()
    entries: list[tuple[str, str, str | None]] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # Detect 3-line section header: border / TITLE / border
        if _SECTION_BORDER.match(line):
            if i + 2 < len(lines):
                title_match = _SECTION_TITLE.match(lines[i + 1])
                if title_match and _SECTION_BORDER.match(lines[i + 2]):
                    entries.append(("section", title_match.group(1).strip(), None))
                    i += 3
                    continue
            # Standalone border — skip
            entries.append(("comment", line, None))
            i += 1
            continue
        m = _KV_PATTERN.match(line)
        if m:
            entries.append(("var", m.group(1), m.group(2)))
        else:
            entries.append(("comment", line, None))
        i += 1
    return entries


def init_env_local(force: bool = False) -> Path:
    """Generate a default .env.local from .env.example with a real JWT secret.

    Unlike generate_env_local(), this doesn't need a MolebieConfig — it just
    copies the template and replaces the JWT_SECRET placeholder.
    """
    root = get_project_root()
    template = root / ".env.example"
    output = root / ".env.local"

    if output.exists() and not force:
        return output

    if not template.exists():
        raise FileNotFoundError(f"Template not found: {template}")

    # Preserve existing JWT_SECRET across regeneration
    _PLACEHOLDER_SECRETS = {"", "CHANGE_ME_TO_A_RANDOM_SECRET"}
    existing_jwt = get_env_key("JWT_SECRET") if output.exists() else None
    if existing_jwt and existing_jwt not in _PLACEHOLDER_SECRETS:
        jwt_secret = existing_jwt
    else:
        jwt_secret = secrets.token_urlsafe(48)

    lines = template.read_text().splitlines()
    result: list[str] = []

    for line in lines:
        m = _KV_PATTERN.match(line)
        if m and m.group(1) == "JWT_SECRET":
            result.append(f"JWT_SECRET={jwt_secret}")
        else:
            result.append(line)

    output.write_text("\n".join(result) + "\n")
    return output


def _build_overrides(config: MolebieConfig) -> dict[str, str]:
    """Build a key→value override map from the config."""
    overrides: dict[str, str] = {}
    gpu = config.gpu_ip
    server = config.server_ip

    # --- Setup type: IP replacements ---
    if config.setup_type == SetupType.TWO_MACHINE:
        # Inference URLs point to GPU machine
        overrides["INFERENCE_THINKING_URL"] = f"http://{gpu}:8080"
        overrides["INFERENCE_INSTANT_URL"] = f"http://{gpu}:8081"
        # Gateway/Webapp URLs point to server machine
        overrides["NEXT_PUBLIC_GATEWAY_URL"] = f"http://{server}:8000"
        overrides["CORS_ORIGINS"] = (
            f"http://localhost:3000,http://127.0.0.1:3000,http://{server}:3000"
        )

    # --- Inference backend ---
    if config.inference_backend == InferenceBackend.OLLAMA:
        overrides["INFERENCE_THINKING_URL"] = f"http://{gpu}:11434"
        overrides["INFERENCE_INSTANT_URL"] = f"http://{gpu}:11434"
        overrides["INFERENCE_THINKING_API_PREFIX"] = "/v1"
        overrides["INFERENCE_INSTANT_API_PREFIX"] = "/v1"
        overrides["INFERENCE_THINKING_MODEL"] = (
            config.thinking_model or OLLAMA_THINKING_MODEL
        )
        overrides["INFERENCE_INSTANT_MODEL"] = (
            config.instant_model or OLLAMA_INSTANT_MODEL
        )
        overrides["INFERENCE_THINKING_ENABLE_THINKING"] = "true"
        overrides["INFERENCE_INSTANT_ENABLE_THINKING"] = "false"

    elif config.inference_backend == InferenceBackend.OPENAI_COMPATIBLE:
        if config.inference_url:
            overrides["INFERENCE_THINKING_URL"] = config.inference_url
            overrides["INFERENCE_INSTANT_URL"] = config.inference_url
        overrides["INFERENCE_THINKING_API_PREFIX"] = "/v1"
        overrides["INFERENCE_INSTANT_API_PREFIX"] = "/v1"
        if config.inference_api_key:
            overrides["INFERENCE_API_KEY"] = config.inference_api_key
        if config.thinking_model:
            overrides["INFERENCE_THINKING_MODEL"] = config.thinking_model
        if config.instant_model:
            overrides["INFERENCE_INSTANT_MODEL"] = config.instant_model

    elif config.inference_backend == InferenceBackend.MLX:
        if config.setup_type == SetupType.TWO_MACHINE:
            overrides["INFERENCE_THINKING_URL"] = f"http://{gpu}:8080"
            overrides["INFERENCE_INSTANT_URL"] = f"http://{gpu}:8081"
        overrides["INFERENCE_THINKING_API_PREFIX"] = ""
        overrides["INFERENCE_INSTANT_API_PREFIX"] = ""
        overrides["INFERENCE_THINKING_MODEL"] = (
            config.thinking_model or MLX_THINKING_MODEL
        )
        overrides["INFERENCE_INSTANT_MODEL"] = (
            config.instant_model or MLX_INSTANT_MODEL
        )

    # --- Feature flags ---
    overrides["WEB_SEARCH_ENABLED"] = str(config.search_enabled).lower()
    overrides["RAG_ENABLED"] = str(config.rag_enabled).lower()

    return overrides


def generate_env_local(config: MolebieConfig, force: bool = False) -> Path:
    """Generate .env.local from .env.example with config-driven overrides.

    Preserves all comments and section structure from the template.
    Returns the path to the generated file.
    """
    root = get_project_root()
    template = root / ".env.example"
    output = root / ".env.local"

    if output.exists() and not force:
        raise FileExistsError(
            f"{output} already exists. Use force=True to overwrite."
        )

    if not template.exists():
        raise FileNotFoundError(f"Template not found: {template}")

    overrides = _build_overrides(config)

    # Preserve existing .env.local values that aren't explicitly overridden.
    # This protects manual edits (INFERENCE_API_KEY, CORS_ORIGINS, etc.) and
    # secrets (JWT_SECRET) across reinstalls.  Config-driven values win;
    # everything else the user set is kept.
    _PLACEHOLDER_SECRETS = {"", "CHANGE_ME_TO_A_RANDOM_SECRET"}
    kv_pattern = re.compile(r"^([A-Z_][A-Z0-9_]*)=(.*)")

    if output.exists():
        for line in output.read_text().splitlines():
            m = kv_pattern.match(line)
            if m:
                key, val = m.group(1), m.group(2)
                if key not in overrides:
                    overrides[key] = val

    # JWT_SECRET: preserve real secret, only generate if missing or placeholder
    existing_jwt = overrides.get("JWT_SECRET")
    if not existing_jwt or existing_jwt in _PLACEHOLDER_SECRETS:
        overrides["JWT_SECRET"] = secrets.token_urlsafe(48)

    lines = template.read_text().splitlines()
    result: list[str] = []

    for line in lines:
        m = kv_pattern.match(line)
        if m and m.group(1) in overrides:
            key = m.group(1)
            result.append(f"{key}={overrides[key]}")
        else:
            result.append(line)

    output.write_text("\n".join(result) + "\n")

    return output


def update_env_key(key: str, value: str) -> bool:
    """Update a single key in .env.local. Returns True if key was found and updated."""
    root = get_project_root()
    env_path = root / ".env.local"
    if not env_path.exists():
        return False

    lines = env_path.read_text().splitlines()
    pattern = re.compile(rf"^{re.escape(key)}=(.*)")
    updated = False

    for i, line in enumerate(lines):
        if pattern.match(line):
            lines[i] = f"{key}={value}"
            updated = True
            break

    if updated:
        env_path.write_text("\n".join(lines) + "\n")
    return updated
