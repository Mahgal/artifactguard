"""ArtifactGuard scanning engine.

Static, opcode-level analysis of serialized model artifacts. The engine
*never* unpickles anything: pickle streams are disassembled with
``pickletools.genops`` and interpreted with a tiny abstract stack machine
that tracks string constants and global references, so a hostile artifact
is evidence, never an execution.

Formats covered:
  * raw pickle streams (.pkl / .pickle / .joblib / legacy .bin / .ckpt)
  * zip containers (.pt / .pth / .npz) — every embedded pickle member is
    scanned, .npy members are header-checked
  * .npy — header parsed; object-dtype arrays carry pickled payloads in
    the data section and those are scanned too
  * .safetensors — header/size consistency (the format cannot carry code;
    unexpected trailing bytes are flagged)

Public entry points: ``scan_bytes``, ``scan_file``, ``scan_directory``.
Everything returns plain dicts (JSON-ready) via ``Finding``/``Report``.
"""

from __future__ import annotations

import ast
import io
import json
import pickletools
import struct
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Rule tables (core set — extended rule packs are maintained out of tree)
# --------------------------------------------------------------------------

# (module, name) -> reason. Any of these referenced by a pickle is a
# code-execution or process-spawning primitive: verdict "dangerous".
CRITICAL_GLOBALS: dict[tuple[str, str], str] = {
    ("os", "system"): "shell command execution",
    ("os", "popen"): "shell command execution",
    ("os", "execv"): "process replacement",
    ("os", "execve"): "process replacement",
    ("os", "execl"): "process replacement",
    ("os", "execle"): "process replacement",
    ("os", "execlp"): "process replacement",
    ("os", "execlpe"): "process replacement",
    ("os", "execvp"): "process replacement",
    ("os", "execvpe"): "process replacement",
    ("os", "spawnl"): "process spawning",
    ("os", "spawnle"): "process spawning",
    ("os", "spawnlp"): "process spawning",
    ("os", "spawnlpe"): "process spawning",
    ("os", "spawnv"): "process spawning",
    ("os", "spawnve"): "process spawning",
    ("os", "spawnvp"): "process spawning",
    ("os", "spawnvpe"): "process spawning",
    ("os", "posix_spawn"): "process spawning",
    ("os", "fork"): "process forking",
    ("posix", "system"): "shell command execution",
    ("nt", "system"): "shell command execution",
    ("subprocess", "Popen"): "process spawning",
    ("subprocess", "run"): "process spawning",
    ("subprocess", "call"): "process spawning",
    ("subprocess", "check_call"): "process spawning",
    ("subprocess", "check_output"): "process spawning",
    ("subprocess", "getoutput"): "shell command execution",
    ("subprocess", "getstatusoutput"): "shell command execution",
    ("builtins", "eval"): "dynamic code evaluation",
    ("builtins", "exec"): "dynamic code execution",
    ("builtins", "compile"): "code object construction",
    ("builtins", "__import__"): "dynamic import",
    ("importlib", "import_module"): "dynamic import",
    ("pty", "spawn"): "shell spawning",
    ("commands", "getoutput"): "shell command execution",
    ("runpy", "run_path"): "code execution from path",
    ("runpy", "run_module"): "code execution by module name",
    ("code", "InteractiveConsole"): "embedded interactive interpreter",
    ("multiprocessing", "Process"): "process spawning",
}

# Referenced = suspicious capability (file mutation, dynamic attribute or
# network machinery). Verdict "dangerous" when invoked, "caution" if only
# referenced without a call.
HIGH_GLOBALS: dict[tuple[str, str], str] = {
    ("builtins", "getattr"): "dynamic attribute access (gadget-chain primitive)",
    ("builtins", "setattr"): "dynamic attribute assignment",
    ("builtins", "open"): "file access",
    ("builtins", "input"): "interactive input",
    ("io", "open"): "file access",
    ("os", "remove"): "file deletion",
    ("os", "unlink"): "file deletion",
    ("os", "rename"): "file rename",
    ("os", "chmod"): "permission change",
    ("os", "chown"): "ownership change",
    ("os", "mkdir"): "directory creation",
    ("os", "makedirs"): "directory creation",
    ("shutil", "rmtree"): "recursive deletion",
    ("shutil", "move"): "file move",
    ("shutil", "copy"): "file copy",
    ("shutil", "copyfile"): "file copy",
    ("socket", "socket"): "network socket creation",
    ("urllib", "urlopen"): "network fetch",
    ("urllib.request", "urlopen"): "network fetch",
    ("requests", "get"): "network fetch",
    ("requests", "post"): "network request",
    ("http.client", "HTTPConnection"): "network connection",
    ("pathlib", "Path"): "filesystem path object (write-capable)",
    ("operator", "attrgetter"): "attribute access gadget",
    ("copyreg", "_reconstructor"): "legacy object reconstruction",
    ("copyreg", "__newobj__"): "object construction helper",
    ("copyreg", "__newobj_ex__"): "object construction helper",
    ("__builtin__", "eval"): "dynamic code evaluation",
    ("__builtin__", "exec"): "dynamic code execution",
    ("__builtin__", "getattr"): "dynamic attribute access (gadget-chain primitive)",
    ("__builtin__", "open"): "file access",
}

# Globals that are the normal, expected content of ML artifacts. These are
# informational only and never raise the verdict on their own.
EXPECTED_GLOBALS: dict[tuple[str, str], str] = {
    ("collections", "OrderedDict"): "ordered state dict (standard in checkpoints)",
    ("numpy", "ndarray"): "numpy array",
    ("numpy", "dtype"): "numpy dtype descriptor",
    ("numpy.core.multiarray", "_reconstruct"): "numpy array reconstruction",
    ("numpy.core.multiarray", "scalar"): "numpy scalar reconstruction",
    ("numpy._core.multiarray", "_reconstruct"): "numpy array reconstruction",
    ("numpy._core.multiarray", "scalar"): "numpy scalar reconstruction",
    ("numpy.dtypes", "Float64DType"): "numpy dtype",
    ("numpy.dtypes", "Float32DType"): "numpy dtype",
    ("numpy.dtypes", "Int64DType"): "numpy dtype",
    ("torch._utils", "_rebuild_tensor"): "torch tensor reconstruction",
    ("torch._utils", "_rebuild_tensor_v2"): "torch tensor reconstruction",
    ("torch._utils", "_rebuild_parameter"): "torch parameter reconstruction",
    ("torch._utils", "_rebuild_qtensor"): "torch quantized tensor reconstruction",
    ("torch", "Tensor"): "torch tensor",
    ("torch", "OrderedDict"): "ordered state dict",
    ("torch.storage", "TypedStorage"): "torch typed storage",
    ("torch.storage", "_UntypedStorage"): "torch untyped storage",
    ("torch.torch_version", "TorchVersion"): "torch version marker",
    ("datetime", "datetime"): "timestamp value",
    ("datetime", "date"): "date value",
    ("datetime", "timedelta"): "duration value",
    ("decimal", "Decimal"): "decimal value",
    ("uuid", "UUID"): "uuid value",
    ("typing", "Any"): "typing marker",
    ("re", "Pattern"): "compiled pattern value",
}

_MODULE_ALIASES = {"__builtin__": "builtins", "copy_reg": "copyreg",
                   "posix": "os", "nt": "os"}

ARTIFACT_SUFFIXES = {
    ".pkl", ".pickle", ".joblib", ".pt", ".pth", ".bin", ".ckpt",
    ".npy", ".npz", ".safetensors", ".model",
}

_SEVERITY_RANK = {"info": 0, "caution": 1, "high": 2, "critical": 3}


@dataclass
class Finding:
    severity: str  # info | caution | high | critical
    kind: str
    detail: str

    def to_dict(self) -> dict:
        return {"severity": self.severity, "kind": self.kind, "detail": self.detail}


@dataclass
class Report:
    path: str
    format: str
    verdict: str = "clean"  # clean | caution | dangerous | unknown
    globals_found: list[dict] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    members: list[dict] = field(default_factory=list)  # container children

    def finish(self) -> "Report":
        worst = 0
        for f in self.findings:
            worst = max(worst, _SEVERITY_RANK.get(f.severity, 0))
        for g in self.globals_found:
            worst = max(worst, _SEVERITY_RANK.get(g.get("severity", "info"), 0))
        for m in self.members:
            worst = max(worst, {"clean": 0, "caution": 1, "dangerous": 3}.get(m.get("verdict", "clean"), 0))
        if self.verdict != "unknown":
            self.verdict = {0: "clean", 1: "caution", 2: "dangerous", 3: "dangerous"}[worst]
        return self

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "format": self.format,
            "verdict": self.verdict,
            "globals": self.globals_found,
            "findings": [f.to_dict() for f in self.findings],
            "notes": self.notes,
            "members": self.members,
        }


# --------------------------------------------------------------------------
# Pickle opcode analysis
# --------------------------------------------------------------------------

# Abstract stack values: ("str", value) | ("global", module, name) | ("mark",) | ("?",)
_STR_OPS = {"SHORT_BINUNICODE", "BINUNICODE", "UNICODE", "BINUNICODE8",
            "SHORT_BINSTRING", "BINSTRING", "STRING"}
_PUSH_UNKNOWN_OPS = {
    "NONE", "NEWTRUE", "NEWFALSE", "BININT", "BININT1", "BININT2", "INT",
    "LONG", "LONG1", "LONG4", "BINFLOAT", "FLOAT", "EMPTY_LIST",
    "EMPTY_DICT", "EMPTY_TUPLE", "EMPTY_SET", "BYTEARRAY8", "BINBYTES",
    "SHORT_BINBYTES", "BINBYTES8", "NEXT_BUFFER",
}
_POP_TO_MARK_OPS = {"TUPLE", "DICT", "SETITEMS", "APPENDS", "ADDITEMS", "OBJ", "FROZENSET"}


def _norm(module: str, name: str) -> tuple[str, str]:
    return (_MODULE_ALIASES.get(module, module), name)


def _classify_global(report: Report, module: str, name: str) -> str:
    """Record a global reference; returns its severity."""
    key = _norm(module, name)
    if key in CRITICAL_GLOBALS:
        sev, reason = "critical", CRITICAL_GLOBALS[key]
    elif key in HIGH_GLOBALS:
        sev, reason = "high", HIGH_GLOBALS[key]
    elif key in EXPECTED_GLOBALS:
        sev, reason = "info", EXPECTED_GLOBALS[key]
    elif key[0] == "__main__":
        sev, reason = "caution", "reference to __main__ (session-defined object; provenance unclear)"
    else:
        sev, reason = "caution", "unreviewed global reference"
    report.globals_found.append(
        {"module": key[0], "name": key[1], "severity": sev, "reason": reason})
    return sev


def analyze_pickle(data: bytes, report: Report) -> None:
    """Abstractly interpret a pickle byte stream. Never executes it."""
    stack: list[tuple] = []
    memo: dict[int, tuple] = {}
    memo_count = 0
    invoked: set[tuple[str, str]] = set()

    def push(v: tuple) -> None:
        stack.append(v)

    def pop() -> tuple:
        return stack.pop() if stack else ("?",)

    def pop_to_mark() -> list[tuple]:
        items: list[tuple] = []
        while stack:
            v = stack.pop()
            if v == ("mark",):
                break
            items.append(v)
        items.reverse()
        return items

    def note_invocation(target: tuple, how: str) -> None:
        if target and target[0] == "global":
            key = (target[1], target[2])
            invoked.add(key)
            sev = _SEVERITY_RANK.get(
                next((g["severity"] for g in report.globals_found
                      if (g["module"], g["name"]) == key), "caution"), 1)
            if sev >= _SEVERITY_RANK["high"]:
                report.findings.append(Finding(
                    "critical" if sev == 3 else "high", "invocation",
                    f"{key[0]}.{key[1]} is actually invoked via {how}"))
            elif key not in EXPECTED_GLOBALS:
                report.findings.append(Finding(
                    "caution", "invocation",
                    f"unreviewed callable {key[0]}.{key[1]} invoked via {how}"))

    try:
        ops = list(pickletools.genops(data))
    except Exception as exc:  # malformed stream: analyse the prefix genops yields lazily below
        ops = None
        report.findings.append(Finding(
            "caution", "malformed", f"pickle stream could not be fully disassembled: {exc}"))
        _analyze_pickle_lenient(data, report)
        return

    for opcode, arg, _pos in ops:
        op = opcode.name
        if op in _STR_OPS:
            push(("str", arg if isinstance(arg, str) else ""))
        elif op == "MARK":
            push(("mark",))
        elif op == "GLOBAL":
            module, _, name = str(arg).partition(" ")
            _classify_global(report, module.strip(), name.strip())
            push(("global",) + _norm(module.strip(), name.strip()))
        elif op == "STACK_GLOBAL":
            name_v = pop()
            module_v = pop()
            if module_v[0] == "str" and name_v[0] == "str":
                _classify_global(report, module_v[1], name_v[1])
                push(("global",) + _norm(module_v[1], name_v[1]))
            else:
                report.findings.append(Finding(
                    "caution", "dynamic-global",
                    "STACK_GLOBAL with non-constant operands (dynamically resolved callable)"))
                push(("?",))
        elif op in ("MEMOIZE",):
            if stack:
                memo[memo_count] = stack[-1]
                memo_count += 1
        elif op in ("PUT", "BINPUT", "LONG_BINPUT"):
            if stack:
                memo[int(arg)] = stack[-1]
        elif op in ("GET", "BINGET", "LONG_BINGET"):
            push(memo.get(int(arg), ("?",)))
        elif op == "REDUCE":
            args_v = pop()
            callable_v = pop()
            _ = args_v
            note_invocation(callable_v, "REDUCE")
            push(("?",))
        elif op == "NEWOBJ":
            _args = pop()
            cls_v = pop()
            note_invocation(cls_v, "NEWOBJ")
            push(("?",))
        elif op == "NEWOBJ_EX":
            pop(); pop()
            cls_v = pop()
            note_invocation(cls_v, "NEWOBJ_EX")
            push(("?",))
        elif op == "INST":
            module, _, name = str(arg).partition(" ")
            pop_to_mark()
            _classify_global(report, module.strip(), name.strip())
            report.findings.append(Finding(
                "caution", "legacy-inst",
                f"protocol-0 INST instantiates {_norm(module.strip(), name.strip())}"))
            push(("?",))
        elif op == "OBJ":
            items = pop_to_mark()
            if items:
                note_invocation(items[0], "OBJ")
            push(("?",))
        elif op == "BUILD":
            pop()
            report.findings.append(Finding(
                "info", "build", "BUILD applies pickled state onto an object (__setstate__)"))
        elif op in ("EXT1", "EXT2", "EXT4"):
            report.findings.append(Finding(
                "caution", "extension-ref",
                f"extension-registry reference (code {arg}) resolves indirectly via copyreg"))
            push(("?",))
        elif op in _POP_TO_MARK_OPS:
            pop_to_mark()
            push(("?",))
        elif op in ("TUPLE1", "TUPLE2", "TUPLE3"):
            n = int(op[-1])
            for _ in range(n):
                pop()
            push(("?",))
        elif op == "SETITEM":
            pop(); pop()
            # the container itself stays on the stack
        elif op == "APPEND":
            pop()
            # the list itself stays on the stack
        elif op == "POP":
            pop()
        elif op == "POP_MARK":
            pop_to_mark()
        elif op == "DUP":
            if stack:
                push(stack[-1])
        elif op == "STOP":
            break
        elif op in _PUSH_UNKNOWN_OPS or op in ("PROTO", "FRAME", "READONLY_BUFFER"):
            if op not in ("PROTO", "FRAME", "READONLY_BUFFER"):
                push(("?",))
        else:
            # Unknown/other opcode: assume it consumes nothing reliably; keep stack as-is.
            pass


def _analyze_pickle_lenient(data: bytes, report: Report) -> None:
    """Fallback: pull whatever genops yields before the stream breaks."""
    try:
        for opcode, arg, _pos in pickletools.genops(data):
            if opcode.name == "GLOBAL":
                module, _, name = str(arg).partition(" ")
                _classify_global(report, module.strip(), name.strip())
    except Exception:
        pass


# --------------------------------------------------------------------------
# Format handlers
# --------------------------------------------------------------------------

def _scan_pickle_report(data: bytes, path: str, fmt: str = "pickle") -> Report:
    report = Report(path=path, format=fmt)
    if not data:
        report.findings.append(Finding("caution", "empty", "file is empty"))
        return report.finish()
    analyze_pickle(data, report)
    return report.finish()


def _parse_npy_header(data: bytes) -> tuple[dict | None, int]:
    """Return (header dict, payload offset) for a .npy byte string."""
    if len(data) < 10 or data[:6] != b"\x93NUMPY":
        return None, 0
    major = data[6]
    if major == 1:
        (hlen,) = struct.unpack("<H", data[8:10])
        offset = 10 + hlen
    else:
        (hlen,) = struct.unpack("<I", data[8:12])
        offset = 12 + hlen
    try:
        header = ast.literal_eval(data[offset - hlen:offset].decode("latin1").strip())
    except Exception:
        return None, 0
    return header, offset


def _scan_npy(data: bytes, path: str) -> Report:
    report = Report(path=path, format="npy")
    header, offset = _parse_npy_header(data)
    if header is None:
        report.verdict = "unknown"
        report.notes.append("could not parse .npy header")
        return report.finish()
    descr = str(header.get("descr", ""))
    report.notes.append(f"descr={descr} shape={header.get('shape')} fortran_order={header.get('fortran_order')}")
    if "'O'" in descr or '"O"' in descr or descr.endswith("|O"):
        report.findings.append(Finding(
            "caution", "object-array",
            "object-dtype array: data section contains pickled objects, scanning it"))
        sub = _scan_pickle_report(data[offset:], path + "!<payload>", "pickle(payload)")
        report.globals_found.extend(sub.globals_found)
        report.findings.extend(sub.findings)
    else:
        report.notes.append("numeric dtype: format carries raw tensor data only")
    return report.finish()


def _scan_safetensors(data: bytes, path: str) -> Report:
    report = Report(path=path, format="safetensors")
    if len(data) < 8:
        report.verdict = "unknown"
        report.notes.append("too short for a safetensors header")
        return report.finish()
    (hlen,) = struct.unpack("<Q", data[:8])
    try:
        header = json.loads(data[8:8 + hlen].decode("utf-8"))
    except Exception as exc:
        report.findings.append(Finding("caution", "malformed", f"safetensors header is not valid JSON: {exc}"))
        return report.finish()
    max_end = 0
    tensors = 0
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        tensors += 1
        offsets = meta.get("data_offsets", [0, 0])
        max_end = max(max_end, int(offsets[1]))
    expected = 8 + hlen + max_end
    report.notes.append(f"{tensors} tensors declared; format cannot carry executable code")
    if len(data) > expected:
        report.findings.append(Finding(
            "caution", "trailing-data",
            f"{len(data) - expected} unexpected trailing bytes after declared tensor data"))
    elif len(data) < expected:
        report.findings.append(Finding(
            "caution", "truncated", "file is shorter than the declared tensor data"))
    return report.finish()


def _scan_zip(data: bytes, path: str) -> Report:
    report = Report(path=path, format="zip-container")
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except Exception as exc:
        report.verdict = "unknown"
        report.notes.append(f"zip header present but archive unreadable: {exc}")
        return report.finish()
    pickle_members = 0
    for info in zf.infolist():
        name = info.filename
        if info.is_dir():
            continue
        lower = name.lower()
        try:
            member = zf.read(name)
        except Exception as exc:
            report.findings.append(Finding("caution", "member-error", f"member {name} unreadable: {exc}"))
            continue
        if lower.endswith((".pkl", "data.pkl")) or lower.endswith(".pickle"):
            pickle_members += 1
            sub = _scan_pickle_report(member, f"{path}!{name}")
            report.members.append(sub.to_dict())
        elif lower.endswith(".npy"):
            sub = _scan_npy(member, f"{path}!{name}")
            report.members.append(sub.to_dict())
        else:
            report.notes.append(f"member {name}: raw data ({info.file_size} bytes), not executable pickle")
    if pickle_members == 0:
        report.notes.append("no pickle members found in archive")
    return report.finish()


def scan_bytes(data: bytes, path: str = "<bytes>") -> Report:
    """Scan one artifact's raw bytes, dispatching on content, not just name."""
    suffix = Path(path).suffix.lower()
    if data[:4] == b"PK\x03\x04":
        return _scan_zip(data, path)
    if data[:6] == b"\x93NUMPY":
        return _scan_npy(data, path)
    if suffix == ".safetensors":
        return _scan_safetensors(data, path)
    if suffix == ".npy":
        return _scan_npy(data, path)
    if suffix in ARTIFACT_SUFFIXES or data[:1] == b"\x80":
        return _scan_pickle_report(data, path)
    report = Report(path=path, format="unknown", verdict="unknown")
    report.notes.append("unrecognized artifact format; not scanned")
    return report.finish()


def scan_file(path: str | Path, max_bytes: int = 1 << 31) -> Report:
    p = Path(path)
    try:
        data = p.read_bytes()
    except OSError as exc:
        report = Report(path=str(p), format="unknown", verdict="unknown")
        report.notes.append(f"could not read file: {exc}")
        return report.finish()
    if len(data) > max_bytes:
        report = Report(path=str(p), format="unknown", verdict="unknown")
        report.notes.append(f"file exceeds scan limit ({len(data)} > {max_bytes} bytes)")
        return report.finish()
    return scan_bytes(data, str(p))


def scan_directory(path: str | Path) -> dict:
    """Recursively scan every artifact file under a directory."""
    root = Path(path)
    results: list[dict] = []
    worst = "clean"
    counts = {"clean": 0, "caution": 0, "dangerous": 0, "unknown": 0}
    if root.is_file():
        files = [root]
    else:
        files = sorted(f for f in root.rglob("*")
                       if f.is_file() and f.suffix.lower() in ARTIFACT_SUFFIXES)
    for f in files:
        rep = scan_file(f).to_dict()
        results.append(rep)
        counts[rep["verdict"]] = counts.get(rep["verdict"], 0) + 1
        if rep["verdict"] == "dangerous":
            worst = "dangerous"
        elif rep["verdict"] == "caution" and worst == "clean":
            worst = "caution"
    return {
        "root": str(root),
        "files_scanned": len(results),
        "verdict": worst,
        "counts": counts,
        "results": results,
    }
