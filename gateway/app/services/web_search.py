"""
Web search service using self-hosted SearXNG.

Queries SearXNG's JSON API, extracts top results, and formats them
for injection into the LLM context so the model can cite up-to-date
information.  Optionally fetches full-page content via trafilatura
for the top N results to give the model richer evidence.
"""

import asyncio
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx

try:
    import trafilatura
except ImportError:  # graceful fallback if not installed
    trafilatura = None  # type: ignore[assignment]

from app.config import Settings, get_settings

_TRIVIAL_PATTERNS = re.compile(
    r"^("
    r"h(i|ey|ello|owdy)"
    r"|yo"
    r"|sup"
    r"|thanks?( you)?"
    r"|thx"
    r"|ok(ay)?"
    r"|sure"
    r"|yep|yeah|yes|no|nope"
    r"|bye|goodbye|see ya"
    r"|good (morning|afternoon|evening|night)"
    r"|gm|gn"
    r"|lol|lmao|haha"
    r"|\\?"   # lone question mark
    r"|\\!"
    r")$",
    re.IGNORECASE,
)

# Messages that definitely do NOT need a web search
_SKIP_PATTERNS = re.compile(
    r"("
    # Code / programming
    r"write (a |me a )?(function|program|script|class|code|test|regex)"
    r"|implement\b|refactor\b|debug\b|compile\b|syntax\b"
    r"|python|javascript|typescript|java\b|rust\b|golang|html|css|sql\b"
    r"|\bcode\b|snippet|algorithm|data structure"
    # Math
    r"|calculate\b|solve\b|equation|integral|derivative|factorial"
    r"|what is \d+\s*[\+\-\*\/\^]\s*\d+"
    # Creative writing
    r"|write me a\b|poem\b|story\b|essay\b|haiku\b|limerick"
    r"|email draft|cover letter|resignation letter"
    # Explanations / definitions
    r"|explain\b|how does .+ work|difference between\b|define\b|meaning of\b"
    r"|what is a\b|what are\b|ELI5|in simple terms"
    # Meta / assistant
    r"|who are you|what can you do|your name|summarize this|translate\b"
    # Follow-ups (short, no temporal words)
    r"|^(continue|go on|elaborate|keep going|what about|and also)\b"
    # Opinion / advice
    r"|what do you think|which is better|recommend\b|pros and cons"
    r"|should i\b|advice\b|opinion\b|suggest\b"
    r")",
    re.IGNORECASE,
)

# Messages that definitely DO need a web search
_SEARCH_PATTERNS = re.compile(
    r"("
    # Temporal / recency
    r"\btoday\b|\bright now\b|\bcurrently\b|\blatest\b|\brecent(ly)?\b"
    r"|\bthis (week|month|year)\b|\b202[4-9]\b|\b203\d\b"
    # News / events
    r"|\bnews\b|\bhappened\b|\bupdate on\b|\bannounce[ds]?\b|\breleased\b|\blaunched\b"
    r"|\belection\b|\bbreaking\b"
    # Real-time data
    r"|\bweather\b|\bstock price\b|\bexchange rate\b|\bscore\b|\bstandings\b|\bforecast\b"
    # Commerce / lookup
    r"|\bprice of\b|\bcost of\b|\bwhere to buy\b"
    r"|\bhours of\b|\baddress of\b|\bopen now\b|\bdirections to\b"
    # Explicit search intent
    r"|\bsearch for\b|\blook up\b|\bfind me\b|\bgoogle\b"
    r")",
    re.IGNORECASE,
)

# Domain patterns for source classification
_OFFICIAL_PATTERNS = (
    ".gov", ".edu", ".mil",
    "docs.", "developer.", "devdocs.",
    "mozilla.org/en-US/docs",
)
_REFERENCE_DOMAINS = (
    "wikipedia.org", "wikimedia.org", "wikidata.org",
    "britannica.com",
)
_FORUM_DOMAINS = (
    "stackoverflow.com", "stackexchange.com", "superuser.com",
    "serverfault.com", "askubuntu.com",
    "reddit.com", "quora.com",
    "github.com/issues", "github.com/discussions",
    "discourse.",
)
_NEWS_DOMAINS = (
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk",
    "nytimes.com", "theguardian.com", "washingtonpost.com",
    "arstechnica.com", "techcrunch.com", "theverge.com",
    "wired.com", "bloomberg.com",
)


def _extract_domain(url: str) -> str:
    """Pull hostname from a URL, stripping www. prefix."""
    try:
        host = urlparse(url).hostname or ""
        return host.removeprefix("www.")
    except Exception:
        return ""


def _classify_source(url: str) -> str:
    """Categorise a URL as official / reference / forum / news / web."""
    lower = url.lower()
    domain = _extract_domain(url).lower()
    for pat in _OFFICIAL_PATTERNS:
        if pat in domain or pat in lower:
            return "official"
    for pat in _REFERENCE_DOMAINS:
        if pat in domain:
            return "reference"
    for pat in _FORUM_DOMAINS:
        if pat in lower:
            return "forum"
    for pat in _NEWS_DOMAINS:
        if pat in domain:
            return "news"
    return "web"


_TRUST_LABELS: Dict[str, str] = {
    "official": "high trust",
    "reference": "high trust, may lag",
    "news": "high trust for events",
    "forum": "moderate trust",
    "web": "low trust, cross-check",
}


def _is_duplicate_content(new_snippet: str, existing_snippets: List[str], threshold: float = 0.6) -> bool:
    """Check if new_snippet is too similar to any existing snippet (Jaccard on word sets)."""
    new_words = set(new_snippet.lower().split())
    if not new_words:
        return False
    for existing in existing_snippets:
        existing_words = set(existing.lower().split())
        if not existing_words:
            continue
        intersection = len(new_words & existing_words)
        union = len(new_words | existing_words)
        if union > 0 and intersection / union >= threshold:
            return True
    return False


class WebSearchService:
    """Queries SearXNG for web results and formats them for LLM context."""

    def __init__(self, settings: Settings):
        self.base_url = settings.searxng_url.rstrip("/")
        self.enabled = settings.web_search_enabled
        self.timeout = settings.web_search_timeout
        self.max_results = settings.web_search_max_results
        self.snippet_max_chars = settings.web_search_snippet_max_chars
        self.fetch_full_content = settings.web_search_fetch_full_content
        self.full_content_count = settings.web_search_full_content_count
        self.full_content_max_chars = settings.web_search_full_content_max_chars
        self.full_content_timeout = settings.web_search_full_content_timeout
        self.llm_classify_enabled = settings.web_search_llm_classify
        self.classify_timeout = settings.web_search_classify_timeout
        self.classify_max_tokens = settings.web_search_classify_max_tokens

    async def _classify_with_llm(self, message: str) -> bool:
        """Ask the instant model whether this message needs a web search."""
        try:
            from app.services.inference import get_inference_service

            inference = get_inference_service()
            prompt = (
                "Decide if this message needs a live web search.\n"
                "Answer SEARCH if it asks about current events, recent news, "
                "real-time data, or facts that change over time.\n"
                "Answer SKIP if it asks about timeless knowledge, creative tasks, "
                "coding, math, personal advice, or explanations.\n\n"
                f'Message: "{message}"\n\n'
                "One word answer:"
            )
            result = await asyncio.wait_for(
                inference.generate_response(
                    messages=[{"role": "user", "content": prompt}],
                    mode="instant",
                    max_tokens=self.classify_max_tokens,
                    temperature=0.0,
                ),
                timeout=self.classify_timeout,
            )
            answer = result.get("content", "").upper()
            needs_search = "SEARCH" in answer
            print(f"[web_search] LLM classify: '{message[:60]}' → {answer.strip()} → search={needs_search}")
            return needs_search
        except Exception as exc:
            print(f"[web_search] LLM classify failed ({exc}), defaulting to no search")
            return False

    async def should_search(self, message: str) -> bool:
        """Classify whether a message needs web search (rules + LLM fallback)."""
        if not self.enabled:
            return False
        cleaned = message.strip().rstrip("?!., ")
        if len(cleaned) < 2:
            return False
        lower = cleaned.lower()

        # Tier 1: Trivial messages (greetings, acks) — always skip
        if _TRIVIAL_PATTERNS.match(cleaned):
            print(f"[web_search] SKIP (trivial): '{cleaned[:60]}'")
            return False

        # Tier 2: Definite search (temporal, news, real-time data) — checked first
        # so explicit search intent ("search for", "latest") beats topic keywords
        if _SEARCH_PATTERNS.search(lower):
            print(f"[web_search] SEARCH (rule): '{cleaned[:60]}'")
            return True

        # Tier 3: Definite skip (code, math, creative, explanations, etc.)
        if _SKIP_PATTERNS.search(lower):
            print(f"[web_search] SKIP (rule): '{cleaned[:60]}'")
            return False

        # Tier 3: Ambiguous — ask the LLM
        if self.llm_classify_enabled:
            return await self._classify_with_llm(cleaned)

        print(f"[web_search] No match, LLM classify off → skip: '{cleaned[:60]}'")
        return False

    async def search(self, query: str, num_results: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Query SearXNG and return a list of result dicts with
        keys: title, url, content, domain, source_type, engines, published_date.

        Returns an empty list on any failure so chat can proceed without search.
        """
        limit = num_results or self.max_results
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(
                    f"{self.base_url}/search",
                    params={
                        "q": query,
                        "format": "json",
                        "engines": "google,duckduckgo,brave,wikipedia",
                    },
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            print(f"[web_search] SearXNG query failed: {exc}")
            return []

        raw_results = data.get("results", [])
        seen_urls: set[str] = set()
        existing_snippets: List[str] = []
        results: List[Dict[str, Any]] = []

        for item in raw_results:
            url = item.get("url", "")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            snippet = (item.get("content") or "").strip()
            if not snippet:
                continue
            # Deduplicate near-identical snippets from different engines
            if _is_duplicate_content(snippet, existing_snippets):
                continue
            if len(snippet) > self.snippet_max_chars:
                snippet = snippet[:self.snippet_max_chars].rsplit(" ", 1)[0] + "..."
            existing_snippets.append(snippet)
            results.append({
                "title": (item.get("title") or "Untitled").strip(),
                "url": url,
                "content": snippet,
                "domain": _extract_domain(url),
                "source_type": _classify_source(url),
                "engines": item.get("engines", []),
                "published_date": item.get("publishedDate", ""),
            })
            if len(results) >= limit:
                break

        print(f"[web_search] Found {len(results)} results for: {query[:80]}")
        return results

    async def _fetch_page_content(self, url: str) -> Optional[str]:
        """Fetch full-page content from *url* and extract clean text."""
        if trafilatura is None:
            return None
        try:
            async with httpx.AsyncClient(timeout=self.full_content_timeout) as client:
                resp = await client.get(
                    url,
                    headers={"User-Agent": "Mozilla/5.0 (compatible; ChatBot/1.0)"},
                    follow_redirects=True,
                )
                resp.raise_for_status()
                html = resp.text
        except Exception as exc:
            print(f"[web_search] Full-page fetch failed for {url}: {exc}")
            return None

        text = trafilatura.extract(
            html,
            include_links=False,
            include_tables=True,
            include_comments=False,
        )
        if not text:
            return None
        if len(text) > self.full_content_max_chars:
            text = text[: self.full_content_max_chars].rsplit(" ", 1)[0]
        return text

    async def enrich_with_full_content(
        self, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Fetch full-page text for the top N results (concurrent)."""
        if not self.fetch_full_content or not results:
            return results

        count = min(self.full_content_count, len(results))
        tasks = [self._fetch_page_content(results[i]["url"]) for i in range(count)]
        fetched = await asyncio.gather(*tasks, return_exceptions=True)

        for i, content in enumerate(fetched):
            if isinstance(content, BaseException) or content is None:
                results[i]["full_content"] = None
                results[i]["content_source"] = "snippet"
            else:
                results[i]["full_content"] = content
                results[i]["content_source"] = "full_page"

        for i in range(count, len(results)):
            results[i]["full_content"] = None
            results[i]["content_source"] = "snippet"

        full_ok = sum(1 for r in results if r.get("content_source") == "full_page")
        print(f"[web_search] Full-page enrichment: {full_ok}/{count} succeeded")
        return results

    def format_results_for_context(self, results: List[Dict[str, Any]]) -> str:
        """Format search results into a text block for the system message."""
        if not results:
            return ""
        lines = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "Untitled")
            url = r.get("url", "")
            domain = r.get("domain", "")
            source_type = r.get("source_type", "web")
            trust = _TRUST_LABELS.get(source_type, "unknown")

            # Prefer full-page content when available
            body = r.get("full_content") or r.get("content", "")
            content_tag = "[Full page content]" if r.get("content_source") == "full_page" else "[Snippet only]"

            header = (
                f"[{i}] {title}\n"
                f"    URL: {url} | Type: {source_type} ({trust}) | Domain: {domain}\n"
                f"    {content_tag}"
            )
            lines.append(f"{header}\n    {body}")
        return "\n\n".join(lines)


# ── Singleton ────────────────────────────────────────────────────

_web_search_service: Optional[WebSearchService] = None


def get_web_search_service() -> WebSearchService:
    """Get cached WebSearchService instance."""
    global _web_search_service
    if _web_search_service is None:
        _web_search_service = WebSearchService(get_settings())
    return _web_search_service
