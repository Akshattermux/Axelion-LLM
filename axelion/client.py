"""Small dependency-free client for the Axelion OpenAI-compatible API."""

import json
from typing import Any, Dict, Iterable, List, Optional
from urllib.request import Request, urlopen


class AxelionClient:
    """Connect an application or agent to a running Axelion server."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def _request(self, path: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8") if payload is not None else None,
            headers=headers,
            method="POST" if payload is not None else "GET",
        )
        with urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))

    def health(self) -> Dict[str, Any]:
        return self._request("/health")

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = "axelion",
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Dict[str, Any]:
        return self._request(
            "/v1/chat/completions",
            {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            },
        )

    def agent(self, query: str, max_steps: int = 3) -> Dict[str, Any]:
        return self._request("/v1/agent/run", {"query": query, "max_steps": max_steps})

    def complete(self, prompt: str, **options: Any) -> str:
        response = self._request("/v1/completions", {"prompt": prompt, **options})
        return response["choices"][0]["text"]
