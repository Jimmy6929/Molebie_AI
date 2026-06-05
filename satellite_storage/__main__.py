"""Entry point for ``python -m satellite_storage``.

Delegates to ``satellite_storage.cli.serve.run`` so this invocation and
``molebie-satellite serve`` share one code path. Honors
``MOLEBIE_STORAGE_PORT`` and ``MOLEBIE_STORAGE_DATA_DIR``.
"""

from __future__ import annotations


def main() -> None:
    from satellite_storage.cli.serve import run

    run()


if __name__ == "__main__":
    main()
