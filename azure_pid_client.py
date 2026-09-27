"""Thin client for Microsoft's P&ID digitization reference inference service.

The service itself must be deployed by the user's Azure/IT team. This module
does not create resources and never stores an API key.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from urllib.parse import urljoin


DEFAULT_ROUTES = {
    "symbol": "/api/pid-digitization/symbol-detection/{pid_id}",
    "text": "/api/pid-digitization/text-detection/{pid_id}",
    "graph": "/api/pid-digitization/graph-construction/{pid_id}",
    "persist": "/api/pid-digitization/graph-persistence/{pid_id}",
}


def _safe_pid_id(value: str) -> str:
    value = value.strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", value):
        raise ValueError("pid_id may contain only letters, numbers, dot, dash and underscore")
    return value


class AzurePidClient:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: int = 180,
        routes: dict | None = None,
    ) -> None:
        self.base_url = (base_url or os.getenv("AZURE_PID_SERVICE_URL", "")).strip().rstrip("/")
        if not self.base_url.startswith("https://"):
            raise ValueError("AZURE_PID_SERVICE_URL must be an approved HTTPS endpoint")
        self.api_key = api_key if api_key is not None else os.getenv("AZURE_PID_API_KEY", "")
        self.timeout = int(timeout)
        self.routes = {**DEFAULT_ROUTES, **(routes or {})}

    def _headers(self) -> dict[str, str]:
        header = os.getenv("AZURE_PID_API_KEY_HEADER", "X-API-Key")
        return {header: self.api_key} if self.api_key else {}

    def _url(self, stage: str, pid_id: str) -> str:
        pid_id = _safe_pid_id(pid_id)
        route = self.routes[stage].format(pid_id=pid_id)
        return urljoin(self.base_url + "/", route.lstrip("/"))

    @staticmethod
    def _decode(response):
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        return response.json() if "json" in content_type else {"text": response.text, "status_code": response.status_code}

    def symbol_detection(self, pid_id: str, image_path: str | Path, bounding_box: dict | None = None) -> dict:
        import requests

        image = Path(image_path)
        if image.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            raise ValueError("Microsoft reference symbol endpoint accepts PNG or JPEG")
        if not image.is_file():
            raise FileNotFoundError(image)
        box = bounding_box or {"topX": 0.0, "topY": 0.0, "bottomX": 1.0, "bottomY": 1.0}
        if set(box) != {"topX", "topY", "bottomX", "bottomY"} or not all(0 <= float(v) <= 1 for v in box.values()):
            raise ValueError("bounding_box must contain normalized topX, topY, bottomX, bottomY")
        with image.open("rb") as handle:
            response = requests.post(
                self._url("symbol", pid_id), headers=self._headers(),
                files={"file": (image.name, handle, "image/png" if image.suffix.lower() == ".png" else "image/jpeg")},
                data={"bounding_box_inclusive_str": json.dumps(box)}, timeout=self.timeout,
            )
        return self._decode(response)

    def post_stage(self, stage: str, pid_id: str, payload: dict | list) -> dict:
        import requests

        if stage not in {"text", "graph", "persist"}:
            raise ValueError("stage must be text, graph or persist")
        response = requests.post(
            self._url(stage, pid_id), headers={**self._headers(), "Content-Type": "application/json"},
            json=payload, timeout=self.timeout,
        )
        return self._decode(response)

    def save_response(self, stage: str, pid_id: str, response: dict, output_dir: str | Path) -> Path:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        target = output / f"{_safe_pid_id(pid_id)}_{stage}_response.json"
        target.write_text(json.dumps(response, indent=2), encoding="utf-8")
        return target
