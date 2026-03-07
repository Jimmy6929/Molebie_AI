"""
Inference service for calling LLM endpoints.

Supports two-tier inference (Instant + Thinking) via OpenAI-compatible
APIs. Each tier can use a different server type:
  - mlx_vlm.server  → VLMs like Qwen3.5 (endpoints at /chat/completions)
  - mlx_lm.server   → text LLMs like Qwen3  (endpoints at /v1/chat/completions)

The API prefix ("/v1" or "") is configurable per tier so both server
types work transparently through the same gateway code.
"""

from typing import Optional, List, Dict, Any, AsyncIterator
import json
import time
import httpx

from app.config import Settings, get_settings


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

    # ==================== Mode Config Helpers ====================

    def _get_endpoint(self, mode: str) -> Optional[str]:
        """Get the inference endpoint URL for the given mode."""
        if mode == "thinking":
            return self.thinking_url
        return self.instant_url

    def _get_api_prefix(self, mode: str) -> str:
        """Get the API path prefix for the given mode ('' or '/v1')."""
        if mode == "thinking":
            return self.thinking_api_prefix
        return self.instant_api_prefix

    def _get_model(self, mode: str) -> str:
        """Get the model name for the given mode."""
        if mode == "thinking":
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

    # ==================== Health Check ====================

    async def check_health(self, mode: str = "instant") -> Dict[str, Any]:
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
                    + ("Serverless pod may be scaled to zero." if mode == "thinking"
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
        messages: List[Dict[str, str]],
        mode: str = "instant",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
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

        Returns:
            Dict with 'content', 'tokens_used', 'mode_used', 'model',
            'fallback_used', 'latency_ms'
        """
        start_time = time.monotonic()
        endpoint = self._get_endpoint(mode)

        if not endpoint:
            # If thinking not configured, try fallback to instant
            if mode == "thinking" and self.fallback_to_instant and self.instant_url:
                print("[inference] Thinking endpoint not configured -- falling back to instant")
                result = await self._call_endpoint(
                    endpoint=self.instant_url,
                    model=self.instant_model,
                    messages=messages,
                    mode="instant",
                    max_tokens=max_tokens or self._get_max_tokens("instant"),
                    temperature=(temperature if temperature is not None
                                 else self._get_temperature("instant")),
                    timeout=self._get_timeout("instant"),
                    api_prefix=self._get_api_prefix("instant"),
                )
                result["fallback_used"] = True
                result["original_mode"] = "thinking"
                result["latency_ms"] = int((time.monotonic() - start_time) * 1000)
                return result

            result = await self._mock_response(messages, mode)
            result["latency_ms"] = int((time.monotonic() - start_time) * 1000)
            return result

        # Call the endpoint for the requested mode
        result = await self._call_endpoint(
            endpoint=endpoint,
            model=self._get_model(mode),
            messages=messages,
            mode=mode,
            max_tokens=max_tokens or self._get_max_tokens(mode),
            temperature=(temperature if temperature is not None
                         else self._get_temperature(mode)),
            timeout=self._get_timeout(mode),
            api_prefix=self._get_api_prefix(mode),
        )

        # If thinking call failed and fallback is enabled, try instant
        if (result.get("_error")
                and mode == "thinking"
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
                timeout=self._get_timeout("instant"),
                api_prefix=self._get_api_prefix("instant"),
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
        messages: List[Dict[str, str]],
        mode: str,
        max_tokens: int,
        temperature: float,
        timeout: float,
        api_prefix: str = "/v1",
    ) -> Dict[str, Any]:
        """
        Make the actual HTTP call to the inference endpoint.
        Works with both mlx_lm (/v1/...) and mlx_vlm (/...) servers.
        """
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                payload: Dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stream": False,
                    "enable_thinking": True,
                }
                response = await client.post(
                    f"{endpoint}{api_prefix}/chat/completions",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()
                data = response.json()

                choice = data["choices"][0]
                usage = data.get("usage", {})

                return {
                    "content": choice["message"]["content"],
                    "tokens_used": usage.get("total_tokens"),
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "mode_used": mode,
                    "model": data.get("model", model),
                    "finish_reason": choice.get("finish_reason"),
                    "fallback_used": False,
                }
        except httpx.ConnectError:
            ctx = "cold start?" if mode == "thinking" else "is pod running?"
            print(f"[inference] Cannot connect to {endpoint} ({mode}) -- {ctx}")
            mock = await self._mock_response(messages, mode)
            mock["_error"] = "connect_error"
            return mock
        except httpx.TimeoutException:
            print(f"[inference] Timeout calling {endpoint} ({mode}, {timeout}s)")
            mock = await self._mock_response(messages, mode)
            mock["_error"] = "timeout"
            return mock
        except httpx.HTTPStatusError as e:
            print(f"[inference] HTTP {e.response.status_code} from {endpoint} "
                  f"({mode}): {e.response.text[:200]}")
            mock = await self._mock_response(messages, mode)
            mock["_error"] = f"http_{e.response.status_code}"
            return mock
        except Exception as e:
            print(f"[inference] Error ({mode}): {e}")
            mock = await self._mock_response(messages, mode)
            mock["_error"] = str(e)
            return mock

    # ==================== Streaming ====================

    async def generate_response_stream(
        self,
        messages: List[Dict[str, str]],
        mode: str = "instant",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
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
        endpoint = self._get_endpoint(mode)
        model = self._get_model(mode)
        resolved_max_tokens = max_tokens or self._get_max_tokens(mode)
        resolved_temperature = (temperature if temperature is not None
                                else self._get_temperature(mode))
        resolved_timeout = self._get_timeout(mode)
        fallback_used = False

        # If endpoint not configured, try fallback or mock
        if not endpoint:
            if mode == "thinking" and self.fallback_to_instant and self.instant_url:
                endpoint = self.instant_url
                model = self.instant_model
                resolved_max_tokens = max_tokens or self._get_max_tokens("instant")
                resolved_temperature = (temperature if temperature is not None
                                        else self._get_temperature("instant"))
                resolved_timeout = self._get_timeout("instant")
                mode = "instant"
                fallback_used = True
                print("[inference] Thinking stream not configured -- falling back to instant")
            else:
                async for chunk in self._mock_stream(messages, mode):
                    yield chunk
                return

        prefix = self._get_api_prefix(mode)

        try:
            # Emit mode metadata as first event
            meta = {"mode": mode, "model": model, "fallback_used": fallback_used}
            yield f"data: {json.dumps({'metadata': meta})}\n\n"

            async with httpx.AsyncClient(timeout=resolved_timeout) as client:
                async with client.stream(
                    "POST",
                    f"{endpoint}{prefix}/chat/completions",
                    json={
                        "model": model,
                        "messages": messages,
                        "max_tokens": resolved_max_tokens,
                        "temperature": resolved_temperature,
                        "stream": True,
                        "enable_thinking": True,
                    },
                    headers={"Content-Type": "application/json"},
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line.strip():
                            yield f"{line}\n\n"
        except Exception as e:
            error_context = f"{mode}, model={model}"
            print(f"[inference] Stream error ({error_context}): {e}")

            # If thinking failed, try fallback stream
            if mode == "thinking" and self.fallback_to_instant and self.instant_url:
                print("[inference] Falling back to instant stream")
                fb_prefix = self._get_api_prefix("instant")
                fallback_meta = {
                    "mode": "instant",
                    "model": self.instant_model,
                    "fallback_used": True,
                }
                yield f"data: {json.dumps({'metadata': fallback_meta})}\n\n"

                try:
                    async with httpx.AsyncClient(
                        timeout=self._get_timeout("instant")
                    ) as client:
                        async with client.stream(
                            "POST",
                            f"{self.instant_url}{fb_prefix}/chat/completions",
                            json={
                                "model": self.instant_model,
                                "messages": messages,
                                "max_tokens": (max_tokens
                                               or self._get_max_tokens("instant")),
                                "temperature": (
                                    temperature if temperature is not None
                                    else self._get_temperature("instant")
                                ),
                                "stream": True,
                                "enable_thinking": True,
                            },
                            headers={"Content-Type": "application/json"},
                        ) as response:
                            response.raise_for_status()
                            async for line in response.aiter_lines():
                                if line.strip():
                                    yield f"{line}\n\n"
                    return
                except Exception as fallback_err:
                    print(f"[inference] Fallback stream also failed: {fallback_err}")

            # Last resort: mock
            async for chunk in self._mock_stream(messages, mode):
                yield chunk

    # ==================== Mock ====================

    async def _mock_response(
        self,
        messages: List[Dict[str, str]],
        mode: str,
    ) -> Dict[str, Any]:
        """Generate a mock response for testing when no endpoint is configured."""
        last_message = messages[-1]["content"] if messages else ""

        mode_label = "Thinking" if mode == "thinking" else "Instant"

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
        messages: List[Dict[str, str]],
        mode: str,
    ) -> AsyncIterator[str]:
        """Stream a mock response word by word."""
        mock = await self._mock_response(messages, mode)

        # Emit metadata
        meta = {"mode": mode, "model": "mock", "fallback_used": False}
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


# Singleton instance
_inference_service: Optional[InferenceService] = None


def get_inference_service() -> InferenceService:
    """Get inference service instance."""
    global _inference_service
    if _inference_service is None:
        _inference_service = InferenceService(get_settings())
    return _inference_service
