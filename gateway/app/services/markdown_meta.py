"""
Markdown metadata extraction for Obsidian-style notes.

Pulls `[[wikilinks]]` and `#tags` out of markdown text so they can be
attached to chunk metadata for graph-aware retrieval. Code fences and
inline code are stripped before matching so that `# python comment`
inside a code block is not mistaken for a tag.
"""

import re
from typing import TypedDict


class MarkdownMeta(TypedDict):
    wikilinks: list[str]
    tags: list[str]


_WIKILINK_RE = re.compile(r"\[\[([^\]\|#]+?)(?:#[^\]\|]+)?(?:\|[^\]]+)?\]\]")
_TAG_RE = re.compile(r"(?:^|\s)#([a-zA-Z][a-zA-Z0-9_/-]{0,63})")
_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`]+`")


def extract_md_metadata(text: str) -> MarkdownMeta:
    """Extract wikilinks and tags from markdown text, code-fence aware."""
    if not text:
        return {"wikilinks": [], "tags": []}

    cleaned = _CODE_FENCE_RE.sub("", text)
    cleaned = _INLINE_CODE_RE.sub("", cleaned)

    seen_links: set[str] = set()
    wikilinks: list[str] = []
    for m in _WIKILINK_RE.finditer(cleaned):
        target = m.group(1).strip()
        if target and target not in seen_links:
            seen_links.add(target)
            wikilinks.append(target)

    seen_tags: set[str] = set()
    tags: list[str] = []
    for m in _TAG_RE.finditer(cleaned):
        tag = m.group(1)
        if tag not in seen_tags:
            seen_tags.add(tag)
            tags.append(tag)

    return {"wikilinks": wikilinks, "tags": tags}
