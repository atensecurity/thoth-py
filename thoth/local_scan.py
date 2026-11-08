"""Scan tool arguments on this side and report what was found, never the values.

When ``thoth.data_handling`` withholds tool-argument values, Thoth cannot look
at them. This module scans the raw arguments in your process first and builds
the ``client_findings`` field of the request: counts, categories, hostnames and
recipient domains. It implements the shared detector spec
(``detectors.v1.json``, vendored next to this file; ``docs/integrations/local-scan-detectors.md``
in the repository explains it).

The detectors are heuristic regular expressions. A finding is a signal, not
proof that data is sensitive, and no finding is not proof that it is not. This
is not a compliance statement of any kind.

What leaves the process: counts, categories, hostnames and recipient domains.
Never values, paths, local parts of addresses, display names or message text.

Trust is the workspace's call. Thoth cannot verify a client scan; it relies on
one only when the workspace opted in and listed this ruleset's hash. Findings
only ever raise a decision.

Failing closed: a check that cannot be completed (a node bound was hit, the
check raised, the report would not fit) is left out of ``scanned`` and has no
findings, so Thoth sees it as missing rather than clean.

This module imports nothing from Thoth's server code; it needs only the
standard library.
"""

from __future__ import annotations

from collections.abc import Callable
import functools
import hashlib
from importlib.metadata import version as package_version
import json
import logging
from pathlib import Path
import re
from typing import Any

logger = logging.getLogger("thoth.local_scan")

SPEC_PATH = Path(__file__).resolve().parent / "detectors.v1.json"
CHECKS = ("dlp", "sensitive_path", "egress_host", "recipient")
REPORT_VERSION = 1
SCANNER_NAME = "atensec-thoth"
# Keep in step with the server's client_findings.py (StringConstraints and MAX_PAYLOAD_BYTES).
_VERSION_PATTERN = re.compile(r"[^A-Za-z0-9._+-]")
_MAX_VERSION_CHARS = 32
_ASCII_WHITESPACE = "\t\n\v\f\r"
# Stands for a recipient that could not be read; it never matches an address.
_UNPARSED = "<unparsed>"
_MAX_DOMAIN = 253
_MAX_LABEL = 63


class LocalScanError(Exception):
    """The detector spec could not be loaded or does not match its pinned hash."""


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def compute_ruleset_hash(spec: dict[str, Any]) -> str:
    """sha256 hex of the canonical ``detectors`` object, as the spec's ``hash_rule`` defines."""
    return hashlib.sha256(canonical(spec["detectors"]).encode("utf-8")).hexdigest()


def _compile(pattern: str, flags: list[str]) -> re.Pattern[str]:
    # Python's default (Unicode) semantics, as the server's. Never re.ASCII.
    return re.compile(pattern, re.IGNORECASE if "i" in flags else 0)


def _sdk_version() -> str:
    try:
        raw = package_version("atensec-thoth")
    except Exception:
        raw = "0"
    cleaned = _VERSION_PATTERN.sub("", raw)[:_MAX_VERSION_CHARS].lstrip("._+-")
    return cleaned or "0"


def _plain(text: str) -> bool:
    """Printable ASCII and ASCII whitespace only (detectors.recipient.ambiguous_input)."""
    return all(0x20 <= ord(ch) <= 0x7E or ch in _ASCII_WHITESPACE for ch in text)


class LocalScanner:
    """A scanner built from the vendored spec. Immutable and safe to share across threads."""

    def __init__(self, spec: dict[str, Any], *, version: str | None = None) -> None:
        actual = compute_ruleset_hash(spec)
        if spec.get("ruleset_hash") != actual:
            raise LocalScanError("detector spec does not match its ruleset_hash")
        self.ruleset_id: str = spec["ruleset_id"]
        self.ruleset_hash: str = actual
        self.version = version or _sdk_version()
        d = spec["detectors"]
        self._d = d
        bounds = d["bounds"]
        self._max_depth: int = bounds["max_depth"]
        self._max_values: int = bounds["max_values"]
        self._max_count: int = bounds["max_count"]
        self._max_findings: int = bounds["max_findings"]
        self._max_bytes: int = bounds["max_payload_bytes"]
        self._steps = [(step, _compile(step["pattern"], step["flags"])) for step in d["dlp"]["steps"]]
        self._categories: list[str] = d["dlp"]["categories"]
        self._path_regex = _compile(d["sensitive_path"]["pattern"], d["sensitive_path"]["flags"])
        self._path_keys: list[str] = d["sensitive_path"]["keys"]
        grammar = d["egress_host"]["host_grammar"]
        self._host_regex = _compile(grammar["pattern"], grammar["flags"])
        self._host_max: int = grammar["max_length"]
        rule = d["recipient"]
        self._recipient = rule
        self._fields = frozenset(rule["fields"])
        self._extra_fields = frozenset(rule["extra_fields_for_sending_tools"])
        self._sending = _compile(rule["sending_tool_pattern"]["pattern"], rule["sending_tool_pattern"]["flags"])
        self._move_tool = _compile(rule["data_move"]["tool_name_hint"]["pattern"], rule["data_move"]["tool_name_hint"]["flags"])
        self._move_fields = frozenset(rule["data_move"]["fields"])
        self._positional = _compile(rule["positional_keys"]["pattern"], rule["positional_keys"]["flags"])
        self._address = _compile(rule["address"]["pattern"], rule["address"]["flags"])
        self._text_domain = re.compile(r"(?<=\S)@(" + rule["text_domain_pattern"]["pattern"] + ")")
        self._scanned_order: list[str] = d["report"]["scanned_order"]

    # ------------------------------------------------------------------ #
    # Traversal                                                           #
    # ------------------------------------------------------------------ #

    def _walk(self, root: Any) -> tuple[list[str], bool]:
        """String leaves visited under the bounds, and whether any node was left unvisited."""
        budget = self._max_values
        leaves: list[str] = []
        truncated = False
        stack: list[tuple[Any, int]] = [(root, 0)]
        # Pre-order without recursion: push children reversed so the first is visited first.
        while stack:
            node, depth = stack.pop()
            if depth > self._max_depth or budget <= 0:
                truncated = True
                continue
            budget -= 1
            if isinstance(node, dict):
                stack.extend((item, depth + 1) for item in reversed(list(node.values())))
            elif isinstance(node, list):
                stack.extend((item, depth + 1) for item in reversed(node))
            elif isinstance(node, str):
                leaves.append(node)
        return leaves, truncated

    # ------------------------------------------------------------------ #
    # Checks                                                              #
    # ------------------------------------------------------------------ #

    def _dlp(self, leaves: list[str]) -> list[dict[str, Any]]:
        counts: dict[str, int] = {}
        for leaf in leaves:
            text = leaf
            for step, regex in self._steps:
                replacement = step["replacement"]

                # A function replacement keeps the text literal: no backslash or group escapes.
                def substitute(m: re.Match[str], r: str = replacement) -> str:
                    return r.replace("{group1}", m.group(1) if m.re.groups else "")

                text, n = regex.subn(substitute, text)
                if n:
                    counts[step["category"]] = counts.get(step["category"], 0) + n
        return [{"kind": "dlp", "category": c, "count": min(counts[c], self._max_count)} for c in self._categories if counts.get(c, 0) > 0]

    def _sensitive_path(self, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        candidates: list[str] = []
        first, second = self._path_keys[0], self._path_keys[1]
        if isinstance(arguments.get(first), str) and arguments[first]:
            candidates.append(arguments[first])
        if isinstance(arguments.get(second), list):
            candidates += [item for item in arguments[second] if isinstance(item, str) and item]
        count = sum(1 for candidate in candidates if self._path_regex.search(candidate))
        return [{"kind": "sensitive_path", "count": min(count, self._max_count)}] if count else []

    @staticmethod
    def _first_string(source: Any, keys: list[str]) -> str | None:
        if not isinstance(source, dict):
            return None
        for key in keys:
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return None

    @staticmethod
    def _host_from_value(value: str) -> tuple[str, bool]:
        """(host, unparseable) by the spec's extraction steps; no URL library, so no parser differential."""
        text = value.strip(" ")
        if any(ch == "\\" or ord(ch) < 0x20 or ord(ch) == 0x7F or ch.isspace() for ch in text):
            return "", True
        if "://" not in text:
            text = "https://" + text
        authority = re.split(r"[/?#]", text.split("://", 1)[1], maxsplit=1)[0]
        if authority.count("[") != authority.count("]") or ("[" in authority and "]" not in authority):
            return "", True
        hostinfo = authority.rpartition("@")[2]
        host = hostinfo[1:].split("]", 1)[0] if hostinfo.startswith("[") else hostinfo.split(":", 1)[0]
        host = host.lower()
        original = value.strip(" ")
        if not host and ("://" in original or original.startswith("//") or "@" in original):
            return "", True
        return host, False

    def _egress(self, arguments: dict[str, Any], metadata: dict[str, Any] | None) -> list[dict[str, Any]]:
        host, unparseable = "", False
        for source in self._d["egress_host"]["sources"]:
            if source["from"] == "metadata":
                if metadata is None:
                    continue
                if "key" in source:
                    broker = metadata.get("secrets_broker")
                    target = broker.get("target_host") if isinstance(broker, dict) and isinstance(broker.get("target_host"), str) else None
                    value = target if target and target.strip() else None
                else:
                    value = self._first_string(metadata, source["first_non_empty_of"])
            else:
                value = self._first_string(arguments, source["first_non_empty_of"])
            if value is not None:
                host, unparseable = self._host_from_value(value)
                if host or unparseable:
                    break
        if unparseable:
            return [{"kind": "egress_host", "verdict": "unresolved"}]
        if not host:
            return []
        if len(host) <= self._host_max and self._host_regex.match(host):
            return [{"kind": "egress_host", "verdict": "not_allowlisted", "host": host}]
        return [{"kind": "egress_host", "verdict": "unresolved"}]

    # Recipient ----------------------------------------------------------

    @staticmethod
    def _normal(key: object) -> str:
        return re.sub(r"[_\-\s]", "", str(key)).lower()

    def _strings(self, value: object) -> list[str]:
        """Every recipient in a field's value; anything unreadable is _UNPARSED."""
        if value is None:
            return []
        if isinstance(value, str):
            out: list[str] = []
            for raw in re.split(r"[,;]", value):
                entry = raw.strip()
                if not entry:
                    continue
                bracketed = re.findall(r"<([^<>]*)>", entry)
                rest = re.sub(r"<([^<>]*)>", " ", entry).split()
                loose = [word for word in rest if "@" in word]
                if bracketed or loose:
                    out += [*(b.strip() for b in bracketed), *loose]
                else:
                    out += rest
            return out
        if isinstance(value, list):
            return [item for element in value for item in self._strings(element)]
        if isinstance(value, dict):
            address_keys = {"email", "address", "emailaddress", "value", "mail"}
            display_keys = {"name", "displayname", "firstname", "lastname", "fullname", "type", "role", "label"}
            addresses: list[str] = []
            other: list[str] = []
            for key, element in value.items():
                name = self._normal(key)
                if name in address_keys:
                    addresses += self._strings(element)
                elif name not in display_keys or not isinstance(element, str) or "@" in element:
                    other += self._strings(element)
            found = [item for item in addresses if item != _UNPARSED] + [item for item in other if "@" in item]
            unreadable = not any("@" in item for item in addresses) or any("@" not in item for item in [*addresses, *other])
            return found + ([_UNPARSED] if unreadable else [])
        return [_UNPARSED]

    def _addresses_in_text(self, value: object) -> list[str]:
        if isinstance(value, str):
            if "@" not in value:
                return []
            out = []
            for domain in self._text_domain.findall(value):
                too_long = len(domain) > _MAX_DOMAIN or any(len(label) > _MAX_LABEL for label in domain.split("."))
                out.append(_UNPARSED if too_long else f"recipient@{domain}")
            return out
        if isinstance(value, list):
            return [address for item in value for address in self._addresses_in_text(item)]
        return []

    def _ambiguous_recipient_input(self, tool_name: str, arguments: Any, fields: frozenset[str], sending: bool) -> bool:
        """detectors.recipient.ambiguous_input: non-ASCII text the Unicode and ASCII engines read differently."""
        if not _plain(tool_name):
            return True

        def keys_plain(value: Any) -> bool:
            if isinstance(value, dict):
                return all(_plain(str(key)) and keys_plain(item) for key, item in value.items())
            if isinstance(value, list):
                return all(keys_plain(item) for item in value)
            return True

        def text_plain(value: Any) -> bool:
            if isinstance(value, str):
                return "@" not in value or _plain(value)
            if isinstance(value, list):
                return all(text_plain(item) for item in value)
            return True

        def ambiguous(value: Any) -> bool:
            if isinstance(value, dict):
                for key, item in value.items():
                    if not _plain(str(key)):
                        return True
                    name = self._normal(key)
                    if name in fields:
                        if not keys_plain(item):
                            return True
                        continue
                    if sending and self._positional.match(name) and not text_plain(item):
                        return True
                    if ambiguous(item):
                        return True
            elif isinstance(value, list):
                return any(ambiguous(item) for item in value)
            return False

        return ambiguous(arguments)

    def _collect_recipients(self, tool_name: str, arguments: Any, fields: frozenset[str], sending: bool) -> tuple[list[str], bool]:
        found: list[str] = []
        data_field = False

        def walk(value: object) -> None:
            nonlocal data_field
            if isinstance(value, dict):
                for key, item in value.items():
                    name = self._normal(key)
                    if name in fields:
                        found.extend(self._strings(item))
                        data_field = data_field or name in self._move_fields
                    else:
                        if sending and self._positional.match(name):
                            found.extend(self._addresses_in_text(item))
                        walk(item)
            elif isinstance(value, list):
                for item in value:
                    walk(item)

        walk(arguments)
        return found, data_field

    def _recipient_findings(self, tool_name: str, arguments: Any) -> list[dict[str, Any]] | None:
        """Findings, or None when the check must be left out (ambiguous input)."""
        sending = bool(self._sending.search(tool_name or ""))
        fields = self._fields | (self._extra_fields if sending else frozenset())
        if self._ambiguous_recipient_input(tool_name or "", arguments, fields, sending):
            return None
        values, data_field = self._collect_recipients(tool_name or "", arguments, fields, sending)
        data_move = bool(self._move_tool.search(tool_name or "")) or data_field
        rule = self._recipient
        domains: list[str] = []
        unresolved = 0
        for value in values:
            match = self._address.match(value.strip()) if value != _UNPARSED else None
            domain = match.group(1).lower() if match else ""
            if domain and len(domain) <= self._host_max and "." in domain and self._host_regex.match(domain):
                if domain not in domains:
                    domains.append(domain)
            else:
                unresolved += 1
        out: list[dict[str, Any]] = []
        if domains:
            out.append({"kind": "recipient", "status": "outside", "data_move": data_move, "domains": domains[: rule["max_domains"]]})
        extra = (1 if len(domains) > rule["max_domains"] else 0) + unresolved
        out += [{"kind": "recipient", "status": "unresolved", "data_move": data_move} for _ in range(min(extra, rule["max_unresolved_findings"]))]
        return out

    # ------------------------------------------------------------------ #
    # Report                                                              #
    # ------------------------------------------------------------------ #

    def scan(self, tool_name: str, arguments: Any, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        """``{"scanned": [...], "findings": [...]}`` for the raw arguments.

        A check that raises or cannot be answered is not in ``scanned`` and has
        no findings. Nothing here raises.
        """
        if not isinstance(arguments, dict):
            return {"scanned": [], "findings": []}
        args: dict[str, Any] = arguments
        meta = metadata if isinstance(metadata, dict) else None
        results: dict[str, list[dict[str, Any]]] = {}

        def run(check: str, fn: Callable[[], list[dict[str, Any]] | None]) -> None:
            try:
                found = fn()
            except Exception:
                # Type only: an exception message could quote a value.
                logger.warning("thoth: local scan check %s failed; reporting it as not scanned", check)
                return
            if found is not None:
                results[check] = found

        bounded: list[tuple[list[str], bool]] = []

        def leaves() -> tuple[list[str], bool]:
            if not bounded:
                bounded.append(self._walk(arguments))
            return bounded[0]

        def dlp() -> list[dict[str, Any]] | None:
            texts, truncated = leaves()
            return None if truncated else self._dlp(texts)

        def recipient() -> list[dict[str, Any]] | None:
            _, truncated = leaves()
            return None if truncated else self._recipient_findings(tool_name, args)

        run("dlp", dlp)
        run("sensitive_path", lambda: self._sensitive_path(args))
        run("egress_host", lambda: self._egress(args, meta))
        run("recipient", recipient)
        scanned = [check for check in self._scanned_order if check in results]
        findings = [finding for check in self._scanned_order for finding in results.get(check, [])]
        return {"scanned": scanned, "findings": findings}

    def _size(self, payload: dict[str, Any]) -> int:
        return len(json.dumps(payload, separators=(",", ":")).encode("utf-8"))

    def client_findings(self, tool_name: str, arguments: Any, metadata: dict[str, Any] | None = None) -> dict[str, Any] | None:
        """The ``client_findings`` request field, or None when nothing was scanned.

        If the report would exceed the server's size or count bound, checks are
        left out (recipient first, then egress_host) rather than sending a
        report the server would reject.
        """
        try:
            result = self.scan(tool_name, arguments, metadata)
        except Exception:
            logger.warning("thoth: local scan failed; sending no findings")
            return None
        scanner = {"name": SCANNER_NAME, "version": self.version, "ruleset_id": self.ruleset_id, "ruleset_hash": self.ruleset_hash}
        scanned, findings = list(result["scanned"]), list(result["findings"])
        for drop in (None, "recipient", "egress_host"):
            if drop is not None:
                scanned = [check for check in scanned if check != drop]
                findings = [finding for finding in findings if finding["kind"] != drop]
            payload = {"version": REPORT_VERSION, "scanner": scanner, "scanned": scanned, "findings": findings}
            if len(findings) <= self._max_findings and self._size(payload) <= self._max_bytes:
                return payload if scanned else None
        return None


@functools.lru_cache(maxsize=1)
def load_scanner() -> LocalScanner | None:
    """The scanner built from the vendored spec, or None (logged once) when it cannot be loaded."""
    try:
        spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
        return LocalScanner(spec)
    except Exception:
        logger.warning("thoth: local scan is unavailable (detector spec could not be loaded); sending no findings")
        return None
