"""ArtifactGuard plugin — registration and load-time guard hook.

ArtifactGuard scans the *data* plane: serialized model artifacts
(pickle / torch / joblib / numpy / safetensors) are disassembled at the
opcode level and checked for the globals they would resolve and invoke
when loaded — before any loader gets the chance.
"""

import logging
import re
from pathlib import Path

from . import scanner, schemas, tools

logger = logging.getLogger(__name__)

# Tool calls whose text can carry a deserialization call.
_CODE_TOOLS = {"terminal", "execute_code", "run_code", "python", "shell"}
# Deserialization entry points we watch for in code/command text.
_LOAD_CALL_RE = re.compile(
    r"(?:pickle\.loads?|torch\.load|joblib\.load|dill\.loads?|"
    r"marshal\.loads?|shelve\.open|cloudpickle\.loads?|np\.load|"
    r"numpy\.load|yaml\.load\s*\([^)]*Loader\s*=\s*Loader)"
)
# Candidate path tokens inside code/command text (split on quotes/space).
_TOKEN_RE = re.compile(r"""[^\s'"]+""")

_settings: dict = {"block_on": "dangerous", "max_bytes": 1 << 31}


def _candidate_paths(args: dict) -> list[str]:
    """Extract existing artifact file paths referenced by a tool call."""
    found: list[str] = []

    def consider(text: str) -> None:
        if Path(text).suffix.lower() in scanner.ARTIFACT_SUFFIXES and Path(text).is_file():
            found.append(text)

    def walk(value) -> None:
        if isinstance(value, str):
            consider(value)
        elif isinstance(value, dict):
            for v in value.values():
                walk(v)
        elif isinstance(value, (list, tuple)):
            for v in value:
                walk(v)

    walk(args)
    return found


def _code_paths(args: dict) -> list[str]:
    """Artifact paths named inside a load call in code/command text."""
    texts = [v for v in args.values() if isinstance(v, str)]
    out: list[str] = []
    for text in texts:
        if not _LOAD_CALL_RE.search(text):
            continue
        for token in _TOKEN_RE.findall(text):
            token = token.strip(",();[]{}")
            if Path(token).suffix.lower() in scanner.ARTIFACT_SUFFIXES and Path(token).is_file():
                out.append(token)
    return out


def _guard_pre_tool_call(tool_name, args, task_id="", **kwargs):
    """pre_tool_call policy hook: block loads of dangerous artifacts.

    Fires before every tool call. Only calls that would actually
    deserialize an artifact are inspected:
      * code/command tools whose text calls a known loader on an
        artifact path, and
      * any tool call that passes an artifact file path while its name
        or arguments indicate loading.
    A "dangerous" verdict blocks the call (configurable via the
    ``block_on`` setting: "dangerous" or "caution").
    """
    try:
        args = args or {}
        paths: list[str] = []
        if tool_name in _CODE_TOOLS:
            paths = _code_paths(args)
        elif "load" in tool_name:
            paths = _candidate_paths(args)
        if not paths:
            return None
        threshold = _settings.get("block_on", "dangerous")
        for path in paths:
            report = scanner.scan_file(path, max_bytes=int(_settings.get("max_bytes", 1 << 31)))
            verdict = report.verdict
            if verdict == "dangerous" or (threshold == "caution" and verdict == "caution"):
                crit: list[str] = []
                globs: list[str] = []

                def collect(findings, globals_found) -> None:
                    for f in findings:
                        sev = f["severity"] if isinstance(f, dict) else f.severity
                        det = f["detail"] if isinstance(f, dict) else f.detail
                        if sev in ("critical", "high"):
                            crit.append(det)
                    for g in globals_found:
                        if g["severity"] in ("critical", "high"):
                            globs.append(f"{g['module']}.{g['name']}")

                collect(report.findings, report.globals_found)
                for member in report.members:
                    collect(member.get("findings", []), member.get("globals", []))
                detail = "; ".join(crit + globs) or verdict
                logger.warning("ArtifactGuard blocked load of %s (%s)", path, detail)
                return {
                    "action": "block",
                    "message": (
                        f"ArtifactGuard blocked this call: {path} scanned as "
                        f"{verdict.upper()} — {detail}. The artifact was NOT "
                        f"loaded. Run artifact_scan for the full report, or "
                        f"move the file aside if this is unexpected."
                    ),
                }
            if verdict in ("caution", "unknown"):
                logger.info("ArtifactGuard: %s scanned as %s (allowed)", path, verdict)
        return None
    except Exception as exc:  # a guard bug must not wedge the agent loop
        logger.warning("ArtifactGuard hook error (failing open): %s", exc)
        return None


def register(ctx):
    """Wire schemas to handlers and register the load guard hook."""
    try:
        _settings["block_on"] = ctx.get_config("block_on", default="dangerous")
    except Exception:
        pass
    ctx.register_tool(name="artifact_scan", toolset="artifactguard",
                      schema=schemas.ARTIFACT_SCAN, handler=tools.artifact_scan)
    ctx.register_tool(name="artifact_audit_plugin", toolset="artifactguard",
                      schema=schemas.ARTIFACT_AUDIT_PLUGIN, handler=tools.artifact_audit_plugin)
    ctx.register_hook("pre_tool_call", _guard_pre_tool_call)
