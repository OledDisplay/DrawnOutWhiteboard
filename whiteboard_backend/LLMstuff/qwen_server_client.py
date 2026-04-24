from __future__ import annotations

from typing import Any, Optional, Sequence


class QwenServerClient:
    """
    Lightweight HTTP client for the external Qwen worker server.
    This module is safe to import on the Windows-side pipeline because it has
    no dependency on vLLM, torch, or the server runtime.
    """

    def __init__(self, base_url: str) -> None:
        self.base_url = str(base_url or "").rstrip("/")
        try:
            import requests  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("requests is required for QwenServerClient") from exc
        self._requests = requests

    def endpoints(self) -> dict[str, str]:
        return {
            "base": self.base_url,
            "status": f"{self.base_url}/status",
            "load_text": f"{self.base_url}/load/text",
            "reload_text": f"{self.base_url}/reload/text",
            "wake_text": f"{self.base_url}/wake/text",
            "load_vision": f"{self.base_url}/load/vision",
            "reload_vision": f"{self.base_url}/reload/vision",
            "wake_vision": f"{self.base_url}/wake/vision",
            "sleep": f"{self.base_url}/sleep",
            "unload": f"{self.base_url}/unload",
            "generate_text": f"{self.base_url}/generate/text",
            "generate_vision": f"{self.base_url}/generate/vision",
            "health": f"{self.base_url}/health",
            "shutdown": f"{self.base_url}/shutdown",
        }

    def load_text(self, **kwargs: Any) -> dict[str, Any]:
        return self._post("/load/text", kwargs)

    def reload_text(self, **kwargs: Any) -> dict[str, Any]:
        return self._post("/reload/text", kwargs)

    def wake_text(self, **kwargs: Any) -> dict[str, Any]:
        return self._post("/wake/text", kwargs)

    def load_vision(self, **kwargs: Any) -> dict[str, Any]:
        return self._post("/load/vision", kwargs)

    def reload_vision(self, **kwargs: Any) -> dict[str, Any]:
        return self._post("/reload/vision", kwargs)

    def wake_vision(self, **kwargs: Any) -> dict[str, Any]:
        return self._post("/wake/vision", kwargs)

    def sleep(self) -> dict[str, Any]:
        return self._post("/sleep", {})

    def unload(self) -> dict[str, Any]:
        return self._post("/unload", {})

    def status(self) -> dict[str, Any]:
        return self._get("/status")

    def generate_text_batch(
        self,
        prompts: Sequence[str],
        *,
        system_prompt: Optional[str] = None,
        generation: Optional[dict[str, Any]] = None,
        use_tqdm: bool = False,
    ) -> dict[str, Any]:
        return self._post(
            "/generate/text",
            {
                "prompts": list(prompts),
                "system_prompt": system_prompt,
                "generation": generation or {},
                "use_tqdm": use_tqdm,
            },
        )

    def generate_vision_batch(
        self,
        requests: Sequence[dict[str, Any]],
        *,
        system_prompt: Optional[str] = None,
        generation: Optional[dict[str, Any]] = None,
        use_tqdm: bool = False,
    ) -> dict[str, Any]:
        return self._post(
            "/generate/vision",
            {
                "requests": list(requests),
                "system_prompt": system_prompt,
                "generation": generation or {},
                "use_tqdm": use_tqdm,
            },
        )

    def text_endpoint(self) -> str:
        return f"{self.base_url}/generate/text"

    def vision_endpoint(self) -> str:
        return f"{self.base_url}/generate/vision"

    def _get(self, path: str) -> dict[str, Any]:
        resp = self._requests.get(f"{self.base_url}{path}", timeout=60)
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        resp = self._requests.post(f"{self.base_url}{path}", json=payload, timeout=600)
        resp.raise_for_status()
        return resp.json()
