"""Small asynchronous client for the existing BCG HTTP contract.

No graph-model credentials or extraction implementation belong in FrontierAgent.
Unknown write outcomes are never retried against the same graph by this client.
"""

from __future__ import annotations

from typing import Any, Protocol

import httpx


class GraphError(RuntimeError):
    """A graph operation was not acknowledged or its result was invalid."""


class GraphBackend(Protocol):
    async def push(self, problem_id: str, turns: list[dict[str, str]]) -> dict[str, Any]: ...
    async def get(self, problem_id: str) -> dict[str, Any] | None: ...
    async def select(
        self, problem_id: str, query: str, question: str, budget: int
    ) -> dict[str, Any]: ...
    async def close(self) -> None: ...


class HttpGraphBackend:
    def __init__(
        self,
        url: str,
        timeout: float,
        api_key: str = "",
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._owns_client = client is None
        self.client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=min(3.0, timeout)),
            follow_redirects=False,
        )

    async def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any] | None:
        try:
            response = await self.client.request(
                method,
                self.url + path,
                headers=self.headers,
                timeout=httpx.Timeout(self.timeout, connect=min(3.0, self.timeout)),
                **kwargs,
            )
            if method == "GET" and response.status_code == 404:
                return None
            response.raise_for_status()
            value = response.json()
            if not isinstance(value, dict):
                raise GraphError("invalid_graph_response")
            return value
        except (httpx.HTTPError, ValueError) as exc:
            # Do not include key-bearing URLs, response bodies, or headers.
            raise GraphError("graph_request_failed") from exc

    async def push(self, problem_id: str, turns: list[dict[str, str]]) -> dict[str, Any]:
        payload = [
            {**turn, "problem_id": problem_id, "is_message_end": True, "is_trajectory_end": False}
            for turn in turns
        ]
        result = await self._request("POST", "/turns", json=payload)
        latest = result.get("latest") if result else None
        snapshot = latest.get(problem_id) if isinstance(latest, dict) else None
        if (
            result is None
            or type(result.get("pushed")) is not int
            or result["pushed"] != len(turns)
            or not isinstance(snapshot, dict)
            or snapshot.get("problem_id") != problem_id
        ):
            raise GraphError("unacknowledged_graph_write")
        return snapshot

    async def get(self, problem_id: str) -> dict[str, Any] | None:
        result = await self._request("GET", "/graph", params={"problem_id": problem_id})
        if result is not None and result.get("problem_id") != problem_id:
            raise GraphError("wrong_graph_session")
        return result

    async def select(
        self, problem_id: str, query: str, question: str, budget: int
    ) -> dict[str, Any]:
        result = await self._request(
            "POST",
            "/context-selection",
            json={
                "problem_id": problem_id,
                "strategy": "focused",
                "query": query,
                "focus_query": query,
                "question": question,
                "node_char_budget": budget,
                "max_depth": 4,
            },
        )
        if result is None or result.get("problem_id") != problem_id:
            raise GraphError("wrong_selection_session")
        return result

    async def close(self) -> None:
        if self._owns_client:
            await self.client.aclose()
