"""Tool handlers — the code that runs when the LLM calls each tool."""

import json
from pathlib import Path

from . import scanner


def artifact_scan(args: dict, **kwargs) -> str:
    """Scan one artifact file. Always returns a JSON string."""
    path = (args or {}).get("path", "")
    if not path:
        return json.dumps({"error": "No path provided"})
    p = Path(path)
    if not p.exists():
        return json.dumps({"error": f"Path does not exist: {path}"})
    if p.is_dir():
        return json.dumps({
            "error": "Path is a directory; use artifact_audit_plugin for directories",
            "path": path,
        })
    try:
        report = scanner.scan_file(p)
        return json.dumps(report.to_dict(), indent=2)
    except Exception as exc:  # handlers never raise
        return json.dumps({"error": f"Scan failed: {exc}", "path": path})


def artifact_audit_plugin(args: dict, **kwargs) -> str:
    """Recursively audit a directory for risky artifacts."""
    path = (args or {}).get("path", "")
    if not path:
        return json.dumps({"error": "No path provided"})
    p = Path(path)
    if not p.exists():
        return json.dumps({"error": f"Path does not exist: {path}"})
    try:
        return json.dumps(scanner.scan_directory(p), indent=2)
    except Exception as exc:  # handlers never raise
        return json.dumps({"error": f"Audit failed: {exc}", "path": path})
