"""
Inference service for calling LLM endpoints.

Supports two-tier inference (Instant + Thinking) via OpenAI-compatible
APIs. Each tier can use a different server type:
  - mlx_vlm.server  → VLMs like Qwen3.5-9B (endpoints at /chat/completions)
  - mlx_lm.server   → text LLMs           (endpoints at /v1/chat/completions)

The API prefix ("/v1" or "") is configurable per tier so both server
types work transparently through the same gateway code.
"""

import json
import re
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from app.config import Settings, get_settings


# Defensive strip: prior `<think>...</think>` blocks must never re-enter the
# prompt. The chat route already strips on persist, but multi-process pipelines
# (background tasks, future replay) can re-introduce them — clean once here so
# the inference layer is always safe regardless of how messages were assembled.
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def _apply_sampling_params(
    payload: dict[str, Any],
    presence_penalty: float,
    repetition_penalty: float,
    flavor: str = "auto",
) -> None:
    """Add sampling penalty fields to ``payload`` only when non-default.

    Default values (``presence_penalty=0.0``, ``repetition_penalty=1.0``)
    are sampler no-ops on every OpenAI-compatible backend, so omitting them
    keeps the wire format identical to a request without penalties — which
    matters because some backends (notably mlx_vlm with ``enable_thinking``
    + ``thinking_budget``) flip sampler / streaming-buffer paths on field
    presence, not just value.

    The ``flavor`` argument controls naming for repetition penalty:
      * ``auto``                    → emit ``repetition_penalty`` AND
                                      ``repeat_penalty`` (covers all backends)
      * ``mlx`` / ``vllm`` / ``ollama`` → ``repetition_penalty``
      * ``llamacpp``                → ``repeat_penalty``

    Idempotent and safe to call once per payload.
    """
    if presence_penalty:                                # 0.0 → falsy → omit
        payload["presence_penalty"] = presence_penalty
    if repetition_penalty and repetition_penalty != 1.0:
        if flavor in ("auto", "mlx", "vllm", "ollama"):
            payload["repetition_penalty"] = repetition_penalty
        if flavor in ("auto", "llamacpp"):
            payload["repeat_penalty"] = repetition_penalty


def _strip_think_in_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return a copy of ``messages`` with any stray ``<think>`` blocks removed
    from string and multimodal-text content. Other parts (images) untouched."""
    cleaned: list[dict[str, Any]] = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            new_content = _THINK_BLOCK_RE.sub("", content)
            if new_content != content:
                msg = {**msg, "content": new_content}
        elif isinstance(content, list):
            new_parts = []
            mutated = False
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text = part.get("text", "")
                    new_text = _THINK_BLOCK_RE.sub("", text)
                    if new_text != text:
                        mutated = True
                        new_parts.append({**part, "text": new_text})
                        continue
                new_parts.append(part)
            if mutated:
                msg = {**msg, "content": new_parts}
        cleaned.append(msg)
    return cleaned


class InferenceService:
    """
    Service for LLM inference calls via OpenAI-compatible API.

    Supports any open-source model served through MLX, vLLM, TGI,
    llama.cpp server, or any OpenAI-compatible endpoint. The service
    is model-agnostic -- model name is purely a config value passed
    through to the server.
    """

    def __init__(self, settings: Settings):
        self.settings = settings

        # Per-mode endpoint URLs
        self.instant_url = settings.inference_instant_url or None
        self.thinking_url = settings.inference_thinking_url or None

        # Per-mode model names (fall back to shared setting)
        self.instant_model = settings.get_model_for_mode("instant")
        self.thinking_model = settings.get_model_for_mode("thinking")

        # Per-mode API prefix ("" for mlx_vlm, "/v1" for mlx_lm)
        self.instant_api_prefix = settings.get_api_prefix_for_mode("instant")
        self.thinking_api_prefix = settings.get_api_prefix_for_mode("thinking")

        # Fallback & routing config
        self.fallback_to_instant = settings.routing_thinking_fallback_to_instant
        self.cold_start_timeout = settings.routing_cold_start_timeout

        # API key for commercial backends (OpenAI, etc.)
        self.api_key = settings.inference_api_key or ""

    # ==================== Shared Helpers ====================

    def _get_headers(self) -> dict:
        """Build HTTP headers, including Authorization if an API key is set."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    # ==================== Mode Config Helpers ====================

    def _get_endpoint(self, mode: str) -> str | None:
        """Get the inference endpoint URL for the given mode."""
        if mode in ("thinking", "thinking_harder"):
            return self.thinking_url
        return self.instant_url

    def _get_api_prefix(self, mode: str) -> str:
        """Get the API path prefix for the given mode ('' or '/v1')."""
        if mode in ("thinking", "thinking_harder"):
            return self.thinking_api_prefix
        return self.instant_api_prefix

    def _get_model(self, mode: str) -> str:
        """Get the model name for the given mode."""
        if mode in ("thinking", "thinking_harder"):
            return self.thinking_model
        return self.instant_model

    def _get_max_tokens(self, mode: str) -> int:
        """Get the default max_tokens for the given mode."""
        return self.settings.get_max_tokens_for_mode(mode)

    def _get_temperature(self, mode: str) -> float:
        """Get the default temperature for the given mode."""
        return self.settings.get_temperature_for_mode(mode)

    def _get_timeout(self, mode: str) -> float:
        """Get the HTTP timeout for the given mode."""
        return self.settings.get_timeout_for_mode(mode)

    def _get_top_p(self, mode: str) -> float:
        """Get the default top_p for the given mode."""
        return self.settings.get_top_p_for_mode(mode)

    def _get_top_k(self, mode: str) -> int:
        """Get the default top_k for the given mode."""
        return self.settings.get_top_k_for_mode(mode)

    def _get_presence_penalty(self, mode: str) -> float:
        """Get presence_penalty for the given mode (clamped at safe ceiling)."""
        return self.settings.get_presence_penalty_for_mode(mode)

    def _get_repetition_penalty(self, mode: str) -> float:
        """Get repetition_penalty for the given mode."""
        return self.settings.get_repetition_penalty_for_mode(mode)

    def _get_backend_flavor(self) -> str:
        """Get the configured backend flavor for parameter naming."""
        return self.settings.inference_backend_flavor

    def _get_enable_thinking(self, mode: str) -> bool:
        """Get whether chain-of-thought is enabled for the given mode."""
        return self.settings.get_enable_thinking_for_mode(mode)

    def _get_thinking_budget(self, mode: str) -> int | None:
        """Get the thinking token budget for the given mode, or None."""
        return self.settings.get_thinking_budget_for_mode(mode)

    # ==================== Health Check ====================

    async def check_health(self, mode: str = "instant") -> dict[str, Any]:
        """
        Check if the inference endpoint is healthy.
        Returns status info including model name and cold-start readiness.
        """
        endpoint = self._get_endpoint(mode)
        model = self._get_model(mode)

        if not endpoint:
            return {
                "status": "not_configured",
                "mode": mode,
                "url": None,
                "model": model,
                "message": f"No {mode} inference endpoint configured. Using mock responses.",
            }

        try:
            prefix = self._get_api_prefix(mode)
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Try /health first, then {prefix}/models (adapts to server type)
                for path in ["/health", f"{prefix}/models"]:
                    try:
                        response = await client.get(f"{endpoint}{path}")
                        if response.status_code == 200:
                            result = {
                                "status": "healthy",
                                "mode": mode,
                                "url": endpoint,
                                "endpoint_checked": path,
                                "model": model,
                                "max_tokens": self._get_max_tokens(mode),
                                "temperature": self._get_temperature(mode),
                                "timeout": self._get_timeout(mode),
                            }
                            if "models" in path:
                                try:
                                    data = response.json()
                                    models_list = data.get("data", [])
                                    if isinstance(models_list, list):
                                        result["available_models"] = [
                                            m.get("id") for m in models_list
                                        ]
                                except Exception:
                                    pass
                            return result
                    except Exception:
                        continue

                return {
                    "status": "unhealthy",
                    "mode": mode,
                    "url": endpoint,
                    "model": model,
                    "message": "Endpoint reachable but no health endpoint responded",
                }
        except httpx.ConnectError:
            return {
                "status": "unreachable",
                "mode": mode,
                "url": endpoint,
                "model": model,
                "message": (
                    f"Cannot connect to {endpoint}. "
                    + ("Serverless pod may be scaled to zero." if mode in ("thinking", "thinking_harder")
                       else "Is the GPU pod running?")
                ),
            }
        except Exception as e:
            return {
                "status": "error",
                "mode": mode,
                "url": endpoint,
                "model": model,
                "message": str(e),
            }

    # ==================== Non-Streaming ====================

    async def generate_response(
        self,
        messages: list[dict[str, Any]],
        mode: str = "instant",
        max_tokens: int | None = None,
        temperature: float | None = None,
        enable_thinking: bool | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
    ) -> dict[str, Any]:
        """
        Generate a complete response from the LLM.

        Works with any open-source model served via an OpenAI-compatible
        API. The model name, token limits, and temperature are resolved
        per-mode from config so both tiers are fully independent.

        If thinking mode fails and fallback is enabled, automatically
        retries with the instant tier.

        Args:
            messages: Conversation history with 'role' and 'content'
            mode: 'instant' or 'thinking'
            max_tokens: Override default max tokens for this mode
            temperature: Override default temperature for this mode
            enable_thinking: Force CoT on/off, overriding the mode default.
                Pass ``False`` for RAG / tool calls (Qwen3.5 9B otherwise
                burns thousands of thinking tokens on routine retrievals).

        Returns:
            Dict with 'content', 'tokens_used', 'mode_used', 'model',
            'fallback_used', 'latency_ms'
        """
        start_time = time.monotonic()
        endpoint = self._get_endpoint(mode)
        resolved_enable_thinking = (
            self._get_enable_thinking(mode) if enable_thinking is None
            else enable_thinking
        )

        if not endpoint:
            # If thinking not configured, try fallback to instant
            if mode in ("thinking", "thinking_harder") and self.fallback_to_instant and self.instant_url:
                print("[inference] Thinking endpoint not configured -- falling back to instant")
                result = await self._call_endpoint(
                    endpoint=self.instant_url,
                    model=self.instant_model,
                    messages=messages,
                    mode="instant",
                    max_tokens=max_tokens or self._get_max_tokens("instant"),
                    temperature=(temperature if temperature is not None
                                 else self._get_temperature("instant")),
                    top_p=self._get_top_p("instant"),
                    top_k=self._get_top_k("instant"),
                    timeout=self._get_timeout("instant"),
                    api_prefix=self._get_api_prefix("instant"),
                    enable_thinking=self._get_enable_thinking("instant"),
                    presence_penalty=self._get_presence_penalty("instant"),
                    repetition_penalty=self._get_repetition_penalty("instant"),
                )
                result["fallback_used"] = True
                result["original_mode"] = "thinking"
                result["latency_ms"] = int((time.monotonic() - start_time) * 1000)
                return result

            result = await self._mock_response(messages, mode)
            result["latency_ms"] = int((time.monotonic() - start_time) * 1000)
            return result

        # Call the endpoint for the requested mode
        # When CoT is forced off, also drop the thinking_budget — sending
        # one alongside enable_thinking=False produces an empty <think>
        # block on some backends that the chat route then has to strip.
        result = await self._call_endpoint(
            endpoint=endpoint,
            model=self._get_model(mode),
            messages=messages,
            mode=mode,
            max_tokens=max_tokens or self._get_max_tokens(mode),
            temperature=(temperature if temperature is not None
                         else self._get_temperature(mode)),
            top_p=self._get_top_p(mode),
            top_k=self._get_top_k(mode),
            timeout=self._get_timeout(mode),
            api_prefix=self._get_api_prefix(mode),
            enable_thinking=resolved_enable_thinking,
            thinking_budget=(self._get_thinking_budget(mode)
                             if resolved_enable_thinking else None),
            presence_penalty=self._get_presence_penalty(mode),
            repetition_penalty=self._get_repetition_penalty(mode),
            tools=tools,
            tool_choice=tool_choice,
        )

        # If thinking call failed and fallback is enabled, try instant
        if (result.get("_error")
                and mode in ("thinking", "thinking_harder")
                and self.fallback_to_instant
                and self.instant_url):
            print("[inference] Thinking endpoint failed -- falling back to instant")
            result = await self._call_endpoint(
                endpoint=self.instant_url,
                model=self.instant_model,
                messages=messages,
                mode="instant",
                max_tokens=max_tokens or self._get_max_tokens("instant"),
                temperature=(temperature if temperature is not None
                             else self._get_temperature("instant")),
                top_p=self._get_top_p("instant"),
                top_k=self._get_top_k("instant"),
                timeout=self._get_timeout("instant"),
                api_prefix=self._get_api_prefix("instant"),
                enable_thinking=self._get_enable_thinking("instant"),
                presence_penalty=self._get_presence_penalty("instant"),
                repetition_penalty=self._get_repetition_penalty("instant"),
            )
            result["fallback_used"] = True
            result["original_mode"] = "thinking"

        result["latency_ms"] = int((time.monotonic() - start_time) * 1000)
        result.pop("_error", None)
        return result

    async def _call_endpoint(
        self,
        endpoint: str,
        model: str,
        messages: list[dict[str, Any]],
        mode: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        timeout: float,
        api_prefix: str = "/v1",
        enable_thinking: bool = False,
        thinking_budget: int | None = None,
        presence_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
    ) -> dict[str, Any]:
        """
        Make the actual HTTP call to the inference endpoint.

        Per-backend notes for the sampling fields:
          * ``presence_penalty`` — accepted by vLLM, mlx_vlm/mlx_lm, Ollama
            (via /v1), llama.cpp server. Standard OpenAI field.
          * ``repetition_penalty`` — Qwen-recommended name; accepted by
            mlx_vlm/mlx_lm and vLLM. llama.cpp server expects
            ``repeat_penalty`` (set server-side via ``--repeat-penalty``).
            Backends that don't recognise the field ignore it harmlessly.
          * ``enable_thinking`` / ``thinking_budget`` — Qwen3.5 + mlx_vlm
            extension. vLLM exposes via ``chat_template_kwargs``;
            llama.cpp via ``--chat-template-kwargs``.
        """
        messages = _strip_think_in_messages(messages)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                payload: dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "stream": False,
                    "enable_thinking": enable_thinking,
                }
                _apply_sampling_params(
                    payload,
                    presence_penalty=presence_penalty,
                    repetition_penalty=repetition_penalty,
                    flavor=self._get_backend_flavor(),
                )
                if thinking_budget is not None:
                    payload["thinking_budget"] = thinking_budget
                    payload["thinking_start_token"] = "<think>"
                    payload["thinking_end_token"] = "</think>"
                if tools:
                    payload["tools"] = tools
                    payload["tool_choice"] = tool_choice or "auto"
                response = await client.post(
                    f"{endpoint}{api_prefix}/chat/completions",
                    json=payload,
                    headers=self._get_headers(),
                )
                response.raise_for_status()
                data = response.json()

                choice = data["choices"][0]
                usage = data.get("usage", {})

                return {
                    "content": choice["message"].get("content") or "",
                    "reasoning_content": choice["message"].get("reasoning_content"),
                    "tool_calls": choice["message"].get("tool_calls") or [],
                    "tokens_used": usage.get("total_tokens"),
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "mode_used": mode,
                    "model": data.get("model", model),
                    "finish_reason": choice.get("finish_reason"),
                    "fallback_used": False,
                }
        except httpx.ConnectError:
            ctx = "cold start?" if mode in ("thinking", "thinking_harder") else "is pod running?"
            print(f"[inference] Cannot connect to {endpoint} ({mode}) -- {ctx}")
            return self._error_response(
                mode=mode, endpoint=endpoint,
                reason="connection refused",
                detail=f"Cannot reach the inference server ({ctx}).",
            )
        except httpx.TimeoutException:
            print(f"[inference] Timeout calling {endpoint} ({mode}, {timeout}s)")
            return self._error_response(
                mode=mode, endpoint=endpoint,
                reason=f"timeout after {int(timeout)}s",
                detail="The inference server accepted the request but didn't respond in time.",
            )
        except httpx.HTTPStatusError as e:
            body = e.response.text[:200]
            print(f"[inference] HTTP {e.response.status_code} from {endpoint} "
                  f"({mode}): {body}")
            return self._error_response(
                mode=mode, endpoint=endpoint,
                reason=f"HTTP {e.response.status_code}",
                detail=body,
            )
        except Exception as e:
            print(f"[inference] Error ({mode}): {e}")
            return self._error_response(
                mode=mode, endpoint=endpoint,
                reason=type(e).__name__,
                detail=str(e),
            )

    # ==================== Streaming ====================

    async def generate_response_stream(
        self,
        messages: list[dict[str, Any]],
        mode: str = "instant",
        max_tokens: int | None = None,
        temperature: float | None = None,
        enable_thinking: bool | None = None,
    ) -> AsyncIterator[str]:
        """
        Generate a streaming response from the LLM (SSE format).

        Model-agnostic -- works with any open-source model behind an
        OpenAI-compatible streaming endpoint. Uses per-mode settings
        for model name, token limits, and timeout.

        If the requested mode's endpoint is unavailable and fallback
        is enabled, streams from the fallback tier instead.

        Yields:
            Server-sent event strings in OpenAI format
        """
        messages = _strip_think_in_messages(messages)
        endpoint = self._get_endpoint(mode)
        model = self._get_model(mode)
        resolved_max_tokens = max_tokens or self._get_max_tokens(mode)
        resolved_temperature = (temperature if temperature is not None
                                else self._get_temperature(mode))
        resolved_top_p = self._get_top_p(mode)
        resolved_top_k = self._get_top_k(mode)
        resolved_timeout = self._get_timeout(mode)
        fallback_used = False

        # If endpoint not configured, try fallback or mock
        if not endpoint:
            if mode in ("thinking", "thinking_harder") and self.fallback_to_instant and self.instant_url:
                endpoint = self.instant_url
                model = self.instant_model
                resolved_max_tokens = max_tokens or self._get_max_tokens("instant")
                resolved_temperature = (temperature if temperature is not None
                                        else self._get_temperature("instant"))
                resolved_top_p = self._get_top_p("instant")
                resolved_top_k = self._get_top_k("instant")
                resolved_timeout = self._get_timeout("instant")
                mode = "instant"
                fallback_used = True
                print("[inference] Thinking stream not configured -- falling back to instant")
            else:
                async for chunk in self._mock_stream(messages, mode):
                    yield chunk
                return

        prefix = self._get_api_prefix(mode)
        resolved_enable_thinking = (
            self._get_enable_thinking(mode) if enable_thinking is None
            else enable_thinking
        )
        resolved_thinking_budget = (
            self._get_thinking_budget(mode) if resolved_enable_thinking else None
        )
        resolved_presence_penalty = self._get_presence_penalty(mode)
        resolved_repetition_penalty = self._get_repetition_penalty(mode)

        try:
            # Emit mode metadata as first event. ``enable_thinking`` here
            # reflects what we actually sent to the backend — distinct from
            # the user's UI mode, which can be "thinking" while CoT is
            # disabled (e.g. RAG auto-disable). The frontend uses this to
            # decide whether mid-stream content might be inside a <think>
            # block whose opener was stripped by the chat template.
            meta = {
                "mode": mode,
                "model": model,
                "fallback_used": fallback_used,
                "enable_thinking": resolved_enable_thinking,
            }
            yield f"data: {json.dumps({'metadata': meta})}\n\n"

            stream_payload: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "max_tokens": resolved_max_tokens,
                "temperature": resolved_temperature,
                "top_p": resolved_top_p,
                "top_k": resolved_top_k,
                "stream": True,
                "enable_thinking": resolved_enable_thinking,
            }
            _apply_sampling_params(
                stream_payload,
                presence_penalty=resolved_presence_penalty,
                repetition_penalty=resolved_repetition_penalty,
                flavor=self._get_backend_flavor(),
            )
            if resolved_thinking_budget is not None:
                stream_payload["thinking_budget"] = resolved_thinking_budget
                stream_payload["thinking_start_token"] = "<think>"
                stream_payload["thinking_end_token"] = "</think>"

            async with httpx.AsyncClient(timeout=resolved_timeout) as client:
                async with client.stream(
                    "POST",
                    f"{endpoint}{prefix}/chat/completions",
                    json=stream_payload,
                    headers=self._get_headers(),
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line.strip():
                            yield f"{line}\n\n"
        except Exception as e:
            error_context = f"{mode}, model={model}"
            print(f"[inference] Stream error ({error_context}): {e}")
            primary_endpoint, primary_reason, primary_detail = (
                endpoint, *_classify_stream_error(e),
            )

            # If thinking failed, try fallback stream
            if mode in ("thinking", "thinking_harder") and self.fallback_to_instant and self.instant_url:
                print("[inference] Falling back to instant stream")
                fb_prefix = self._get_api_prefix("instant")
                fallback_meta = {
                    "mode": "instant",
                    "model": self.instant_model,
                    "fallback_used": True,
                    "enable_thinking": self._get_enable_thinking("instant"),
                }
                yield f"data: {json.dumps({'metadata': fallback_meta})}\n\n"

                try:
                    fb_payload: dict[str, Any] = {
                        "model": self.instant_model,
                        "messages": messages,
                        "max_tokens": (max_tokens
                                       or self._get_max_tokens("instant")),
                        "temperature": (
                            temperature if temperature is not None
                            else self._get_temperature("instant")
                        ),
                        "top_p": self._get_top_p("instant"),
                        "top_k": self._get_top_k("instant"),
                        "stream": True,
                        "enable_thinking": self._get_enable_thinking("instant"),
                    }
                    _apply_sampling_params(
                        fb_payload,
                        presence_penalty=self._get_presence_penalty("instant"),
                        repetition_penalty=self._get_repetition_penalty("instant"),
                        flavor=self._get_backend_flavor(),
                    )
                    async with httpx.AsyncClient(
                        timeout=self._get_timeout("instant")
                    ) as client:
                        async with client.stream(
                            "POST",
                            f"{self.instant_url}{fb_prefix}/chat/completions",
                            json=fb_payload,
                            headers=self._get_headers(),
                        ) as response:
                            response.raise_for_status()
                            async for line in response.aiter_lines():
                                if line.strip():
                                    yield f"{line}\n\n"
                    return
                except Exception as fallback_err:
                    print(f"[inference] Fallback stream also failed: {fallback_err}")
                    # Report the *fallback* failure — it's what the user ultimately saw.
                    primary_endpoint = self.instant_url
                    primary_reason, primary_detail = _classify_stream_error(fallback_err)

            # Surface a real error to the user — don't pretend the endpoint is unconfigured.
            async for chunk in self._error_stream(
                mode=mode, endpoint=primary_endpoint,
                reason=primary_reason, detail=primary_detail,
            ):
                yield chunk

    # ==================== Error (configured-but-failed) ====================

    def _error_response(
        self,
        mode: str,
        endpoint: str,
        reason: str,
        detail: str = "",
    ) -> dict[str, Any]:
        """Build a response shape identical to a real chat message, but whose
        content tells the user what broke and where to look.

        Distinct from ``_mock_response``: this is for endpoints that ARE
        configured but failed at call time, where the mock script's
        "configure your endpoints" text would be actively misleading.
        """
        mode_label = "Thinking" if mode in ("thinking", "thinking_harder") else "Instant"
        log_hint = _mlx_log_hint(endpoint)

        lines = [
            f"⚠️ **Inference error** — the **{mode_label}** tier at "
            f"`{endpoint}` returned `{reason}`.",
        ]
        if detail:
            first_line = detail.strip().splitlines()[0][:240]
            if first_line:
                lines.append(f"> `{first_line}`")
        if log_hint:
            lines.append(
                f"Check `{log_hint}` for the full traceback, then restart "
                f"the stack with `molebie-ai run`."
            )
        else:
            lines.append(
                "Check the inference server's logs, then restart with `molebie-ai run`."
            )

        content = "\n\n".join(lines)

        return {
            "content": content,
            "tokens_used": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "mode_used": mode,
            "model": "error",
            "finish_reason": "stop",
            "fallback_used": False,
            "_error": reason,
        }

    async def _error_stream(
        self,
        mode: str,
        endpoint: str,
        reason: str,
        detail: str = "",
    ) -> AsyncIterator[str]:
        """SSE variant of ``_error_response``. Mirrors ``_mock_stream`` shape so
        the frontend renders it the same way as a normal assistant message."""
        err = self._error_response(mode, endpoint, reason, detail)

        meta = {"mode": mode, "model": "error", "fallback_used": False, "enable_thinking": False}
        yield f"data: {json.dumps({'metadata': meta})}\n\n"

        words = err["content"].split(" ")
        for i, word in enumerate(words):
            chunk = {
                "choices": [{
                    "delta": {
                        "content": word + (" " if i < len(words) - 1 else ""),
                    },
                    "index": 0,
                    "finish_reason": None if i < len(words) - 1 else "stop",
                }],
                "model": "error",
            }
            yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"

    # ==================== Mock ====================

    async def _mock_response(
        self,
        messages: list[dict[str, Any]],
        mode: str,
    ) -> dict[str, Any]:
        """Generate a mock response for testing when no endpoint is configured."""
        raw_content = messages[-1]["content"] if messages else ""
        last_message = raw_content if isinstance(raw_content, str) else (
            next((p["text"] for p in raw_content if isinstance(p, dict) and p.get("type") == "text"), "")
        )

        mode_label = "Thinking" if mode in ("thinking", "thinking_harder") else "Instant"

        if "hello" in last_message.lower() or "hi" in last_message.lower():
            content = "Hello! I'm your AI assistant. How can I help you today?"
        elif "2+2" in last_message or "2 + 2" in last_message:
            content = "2 + 2 equals 4."
        elif "?" in last_message:
            content = (
                f"That's a great question! I'm currently running in **mock mode** "
                f"({mode_label}). Once the inference endpoints are configured, I'll "
                f"be able to provide real AI-powered responses.\n\n"
                f"To connect a real model:\n"
                f"- **Instant**: Set `INFERENCE_INSTANT_URL` and `INFERENCE_INSTANT_MODEL`\n"
                f"- **Thinking**: Set `INFERENCE_THINKING_URL` and `INFERENCE_THINKING_MODEL`\n\n"
                f"Any OpenAI-compatible server works (MLX, vLLM, TGI, llama.cpp)."
            )
        else:
            content = (
                f"I received your message. I'm currently running in **mock mode** "
                f"({mode_label}). Configure your `.env` to connect a real model server.\n\n"
                f"See `GPU_SETUP_GUIDE.md` (Instant) or "
                f"`GPU_SETUP_GUIDE_THINKING.md` (Thinking) for setup instructions."
            )

        return {
            "content": content,
            "tokens_used": len(content.split()) * 2,
            "prompt_tokens": sum(len(m["content"].split()) for m in messages),
            "completion_tokens": len(content.split()),
            "mode_used": mode,
            "model": "mock",
            "finish_reason": "stop",
            "fallback_used": False,
        }

    async def _mock_stream(
        self,
        messages: list[dict[str, Any]],
        mode: str,
    ) -> AsyncIterator[str]:
        """Stream a mock response word by word."""
        mock = await self._mock_response(messages, mode)

        # Emit metadata
        meta = {"mode": mode, "model": "mock", "fallback_used": False, "enable_thinking": False}
        yield f"data: {json.dumps({'metadata': meta})}\n\n"

        words = mock["content"].split()
        for i, word in enumerate(words):
            chunk = {
                "choices": [{
                    "delta": {
                        "content": word + (" " if i < len(words) - 1 else ""),
                    },
                    "index": 0,
                    "finish_reason": None if i < len(words) - 1 else "stop",
                }],
                "model": "mock",
            }
            yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"


def _classify_stream_error(exc: BaseException) -> tuple[str, str]:
    """Turn an exception raised while streaming into (reason, detail) strings
    suitable for _error_response / _error_stream."""
    if isinstance(exc, httpx.HTTPStatusError):
        body = exc.response.text[:240] if exc.response is not None else ""
        return f"HTTP {exc.response.status_code}", body
    if isinstance(exc, httpx.ConnectError):
        return "connection refused", "Cannot reach the inference server."
    if isinstance(exc, httpx.TimeoutException):
        return "timeout", "The inference server didn't respond in time."
    return type(exc).__name__, str(exc)


def _mlx_log_hint(endpoint: str) -> str | None:
    """If the endpoint looks like a locally-managed MLX server, return the
    per-service log file path that the CLI writes via ServiceRunner. Returns
    None for remote / non-MLX endpoints where we can't reliably guess the log."""
    if not endpoint:
        return None
    low = endpoint.lower()
    if "://localhost" not in low and "://127.0.0.1" not in low:
        return None
    if ":8080" in low:
        return "data/logs/mlx-thinking.log"
    if ":8081" in low:
        return "data/logs/mlx-instant.log"
    return None


# Singleton instance
_inference_service: InferenceService | None = None


def get_inference_service() -> InferenceService:
    """Get inference service instance."""
    global _inference_service
    if _inference_service is None:
        _inference_service = InferenceService(get_settings())
    return _inference_service
