"""Best-effort advisory metadata; never grants execution authority."""

import json
import os
import threading
from typing import Any
from urllib.parse import quote

import httpx

_VALUE_KEYS = {"default", "examples", "example", "const", "enum"}


def sanitize_schema(value: Any, depth: int = 0, named_map: bool = False) -> Any:
    if depth > 32:
        raise ValueError("schema too deep")
    if isinstance(value, dict):
        return {
            key: sanitize_schema(item, depth + 1, not named_map and key in {"properties", "patternProperties", "$defs", "definitions"})
            for key, item in value.items()
            if named_map or key not in _VALUE_KEYS
        }
    if isinstance(value, list):
        return [sanitize_schema(item, depth + 1) for item in value]
    return value


def _publish(agent: Any, tenant_id: str, agent_id: str, base: str, key: str | None) -> None:
    try:
        tools = getattr(agent, "tools", [])
        metadata = []
        for tool in tools if isinstance(tools, list) else []:
            name = ""
            try:
                name = getattr(tool, "name", "")
                if not isinstance(name, str) or not name:
                    continue
                schema = getattr(tool, "args_schema", None)
                builder = getattr(schema, "model_json_schema", None)
                if callable(builder):
                    schema = builder()
                schema = sanitize_schema(schema) if isinstance(schema, dict) else {}
                json.dumps(schema, allow_nan=False)
            except Exception:
                if not isinstance(name, str) or not name:
                    continue
                schema = {}
            metadata.append({"name": name, "description": str(getattr(tool, "description", "") or "")[:4096], "input_schema": schema})
        with httpx.Client(timeout=2.0) as client:
            for offset in range(0, len(metadata), 100):
                client.post(
                    f"{base}/{quote(tenant_id, safe='')}/thoth/tool-inventory",
                    headers={"x-api-key": key} if key else {},
                    json={"agent_id": agent_id, "source": "python_sdk", "tools": metadata[offset : offset + 100]},
                )
    except Exception:
        # Advisory collection must never affect agent startup or governance.
        pass


def publish_tool_inventory(agent: Any, *, tenant_id: str, agent_id: str, api_url: str | None, api_key: str | None) -> None:
    base = (api_url or os.getenv("THOTH_API_URL") or "").rstrip("/")
    if not base or os.getenv("THOTH_TOOL_INVENTORY", "").lower() == "off":
        return
    try:
        threading.Thread(target=_publish, args=(agent, tenant_id, agent_id, base, api_key or os.getenv("THOTH_API_KEY")), daemon=True).start()
    except Exception:
        pass
