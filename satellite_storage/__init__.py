"""Molebie AI satellite-side blob storage service.

Standalone FastAPI app run on a satellite machine. Exposes a small
content-addressable storage surface on ``:8090`` that the primary's
TieredStorageService (slice 9.3) calls over Tailscale to offload cold
files like uploaded documents and images.

This is a separate process from the gateway — its own dependencies,
its own data directory, its own port. Run via ``python -m satellite_storage``.
"""

__version__ = "0.1.0"
