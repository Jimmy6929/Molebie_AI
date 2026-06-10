"""Guard against test files that pytest silently never collects.

``verify_phase12.py`` sat in this directory for months with 26 passing
tests that CI never ran, because discovery only matches ``test_*.py``
(see ``[tool.pytest.ini_options] python_files`` in pyproject.toml).
This meta-test fails loudly if a sibling file defines test functions
under a filename discovery would skip.
"""

import re
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
TEST_DEF_RE = re.compile(r"^(?:(?:async )?def test_|class Test)", re.MULTILINE)


def test_no_test_functions_outside_collection_pattern():
    offenders = []
    for path in TESTS_DIR.rglob("*.py"):
        if path.name.startswith("test_") or path.name == "conftest.py":
            continue
        if TEST_DEF_RE.search(path.read_text(encoding="utf-8", errors="replace")):
            offenders.append(str(path.relative_to(TESTS_DIR)))
    assert not offenders, (
        "These files define test functions but their names don't match the "
        f"'test_*.py' discovery pattern, so pytest/CI silently skips them: {offenders}. "
        "Rename them to test_<name>.py."
    )
