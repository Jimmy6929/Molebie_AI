"""Manage Supabase: start, extract keys, inject into .env.local."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from cli.services.config_manager import get_project_root
from cli.services.env_generator import update_env_key
from cli.ui.console import print_fail, print_ok, print_warn


@dataclass
class SupabaseKeys:
    anon_key: str
    service_role_key: str
    jwt_secret: str


def start_supabase() -> bool:
    """Start Supabase local. Returns True on success."""
    root = get_project_root()
    supabase_dir = root / "supabase"
    if not supabase_dir.exists():
        print_fail("supabase/ directory not found")
        return False

    result = subprocess.run(
        ["supabase", "start"],
        cwd=str(supabase_dir),
        capture_output=False,
        timeout=300,
    )
    return result.returncode == 0


def extract_keys() -> SupabaseKeys | None:
    """Extract Supabase keys from `supabase status -o env`."""
    root = get_project_root()
    supabase_dir = root / "supabase"

    try:
        result = subprocess.run(
            ["supabase", "status", "-o", "env"],
            cwd=str(supabase_dir),
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None

    if result.returncode != 0:
        return None

    output = result.stdout

    def _extract(key: str) -> str:
        match = re.search(rf'^{key}="?([^"\n]+)"?', output, re.MULTILINE)
        return match.group(1) if match else ""

    anon = _extract("ANON_KEY")
    service = _extract("SERVICE_ROLE_KEY")
    jwt = _extract("JWT_SECRET")

    if not all([anon, service, jwt]):
        return None

    return SupabaseKeys(anon_key=anon, service_role_key=service, jwt_secret=jwt)


def inject_keys(keys: SupabaseKeys) -> bool:
    """Inject Supabase keys into .env.local."""
    updates = {
        "SUPABASE_ANON_KEY": keys.anon_key,
        "SUPABASE_SERVICE_ROLE_KEY": keys.service_role_key,
        "NEXT_PUBLIC_SUPABASE_ANON_KEY": keys.anon_key,
        "JWT_SECRET": keys.jwt_secret,
    }
    all_ok = True
    for key, value in updates.items():
        if not update_env_key(key, value):
            all_ok = False
    return all_ok


def setup_supabase() -> bool:
    """Full Supabase setup: start, extract keys, inject."""
    print_ok("Starting Supabase (may take a minute on first run)...")
    if not start_supabase():
        print_fail("Failed to start Supabase")
        return False
    print_ok("Supabase is running")

    keys = extract_keys()
    if keys is None:
        print_warn("Could not extract Supabase keys automatically")
        print_warn("Run: cd supabase && supabase status  — then update .env.local manually")
        return True  # Supabase is running, just keys missing

    if inject_keys(keys):
        print_ok("Supabase keys injected into .env.local")
    else:
        print_warn("Some keys could not be injected — check .env.local manually")

    return True
