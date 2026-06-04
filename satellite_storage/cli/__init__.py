"""molebie-satellite CLI — standalone installer + operator surface.

This module ships with the ``molebie-satellite`` package. It is intentionally
*not* shared with the primary's ``cli/`` package: the satellite is meant to
be installable on a new machine via ``pipx install`` without dragging in the
full Molebie repo. Small helper duplication (Tailscale discovery, rich
console wrappers) is the price for keeping the two installables independent —
the same call we made for the gateway's ``tailscale_outbound.py``.
"""
