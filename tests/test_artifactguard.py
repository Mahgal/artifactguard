"""Tests for ArtifactGuard.

SAFETY: fixtures are generated in memory at test time and are only ever
*scanned* — nothing in this file calls pickle.load/loads on them. The
repository intentionally ships no serialized artifact files.
"""

import importlib.util
import io
import json
import os
import pickle
import struct
import sys
import tempfile
import unittest
import zipfile
from collections import OrderedDict
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parent.parent


def _load_package():
    spec = importlib.util.spec_from_file_location(
        "artifactguard", PLUGIN_DIR / "__init__.py",
        submodule_search_locations=[str(PLUGIN_DIR)])
    module = importlib.util.module_from_spec(spec)
    sys.modules["artifactguard"] = module
    spec.loader.exec_module(module)
    return module


pkg = _load_package()
scanner = pkg.scanner


# ---------------------------------------------------------------------------
# Fixture builders — bytes only, never loaded
# ---------------------------------------------------------------------------

class _ReduceTo:
    """An object whose pickle stream invokes a chosen callable when loaded."""

    def __init__(self, func, arg):
        self.func = func
        self.arg = arg

    def __reduce__(self):
        return (self.func, (self.arg,))


def benign_pickle() -> bytes:
    return pickle.dumps({"weights": [1, 2, 3], "meta": ("v1", 4.5)}, protocol=4)


def benign_ml_pickle() -> bytes:
    return pickle.dumps(OrderedDict([("layer.weight", [0.1, 0.2])]), protocol=4)


def evil_system_pickle() -> bytes:
    return pickle.dumps(_ReduceTo(os.system, "echo pwned"), protocol=2)


def evil_eval_pickle() -> bytes:
    return pickle.dumps(_ReduceTo(eval, "1+1"), protocol=4)


def subprocess_pickle() -> bytes:
    import subprocess
    return pickle.dumps(_ReduceTo(subprocess.Popen, "id"), protocol=4)


def unreviewed_callable_pickle() -> bytes:
    return pickle.dumps(_ReduceTo(json.loads, "{}"), protocol=4)


def stack_global_ref_bytes() -> bytes:
    # PROTO 4, SHORT_BINUNICODE "os", SHORT_BINUNICODE "system", STACK_GLOBAL, STOP
    return b"\x80\x04\x8c\x02os\x8c\x06system\x93."


def torch_zip(payload: bytes) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("archive/data.pkl", payload)
        zf.writestr("archive/data/0", b"\x00\x01\x02\x03")
    return buf.getvalue()


def npy_bytes(descr: str, payload: bytes) -> bytes:
    header = "{'descr': '%s', 'fortran_order': False, 'shape': (1,), }" % descr
    pad = 64 - ((10 + len(header) + 1) % 64)
    header = header + " " * pad + "\n"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header.encode("latin1") + payload


def safetensors_bytes(trailing: bytes = b"") -> bytes:
    header = json.dumps({"w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    return struct.pack("<Q", len(header)) + header + b"\x00\x00\x00\x00" + trailing


# ---------------------------------------------------------------------------
# Scanner tests
# ---------------------------------------------------------------------------

class TestPickleScanner(unittest.TestCase):
    def scan(self, data, name="x.pkl"):
        return scanner.scan_bytes(data, name).to_dict()

    def test_benign_dict_is_clean(self):
        rep = self.scan(benign_pickle())
        self.assertEqual(rep["verdict"], "clean", rep)

    def test_ordered_dict_checkpoint_is_clean(self):
        rep = self.scan(benign_ml_pickle())
        self.assertEqual(rep["verdict"], "clean", rep)
        self.assertTrue(any(g["module"] == "collections" for g in rep["globals"]))

    def test_os_system_reduce_is_dangerous(self):
        rep = self.scan(evil_system_pickle())
        self.assertEqual(rep["verdict"], "dangerous", rep)
        self.assertTrue(any(g["module"] == "os" and g["name"] == "system"
                            and g["severity"] == "critical" for g in rep["globals"]))
        self.assertTrue(any(f["kind"] == "invocation" for f in rep["findings"]))

    def test_eval_via_stack_global_is_dangerous(self):
        rep = self.scan(evil_eval_pickle())
        self.assertEqual(rep["verdict"], "dangerous", rep)
        self.assertTrue(any(g["name"] == "eval" and g["severity"] == "critical"
                            for g in rep["globals"]))

    def test_subprocess_popen_is_dangerous(self):
        rep = self.scan(subprocess_pickle())
        self.assertEqual(rep["verdict"], "dangerous", rep)

    def test_bare_stack_global_reference_is_flagged(self):
        rep = self.scan(stack_global_ref_bytes())
        self.assertEqual(rep["verdict"], "dangerous", rep)
        self.assertTrue(any(g["module"] == "os" and g["name"] == "system"
                            for g in rep["globals"]))
        # referenced but never invoked: no invocation finding
        self.assertFalse(any(f["kind"] == "invocation" for f in rep["findings"]))

    def test_unreviewed_callable_is_caution(self):
        rep = self.scan(unreviewed_callable_pickle())
        self.assertEqual(rep["verdict"], "caution", rep)

    def test_truncated_pickle_is_caution_not_crash(self):
        rep = self.scan(benign_pickle()[:5])
        self.assertIn(rep["verdict"], ("caution", "clean", "unknown"))

    def test_empty_file_is_caution(self):
        rep = self.scan(b"")
        self.assertEqual(rep["verdict"], "caution", rep)


class TestContainerFormats(unittest.TestCase):
    def test_torch_zip_with_evil_member_is_dangerous(self):
        rep = scanner.scan_bytes(torch_zip(evil_system_pickle()), "model.pt").to_dict()
        self.assertEqual(rep["verdict"], "dangerous", rep)
        self.assertTrue(rep["members"])

    def test_torch_zip_benign_is_clean(self):
        rep = scanner.scan_bytes(torch_zip(benign_ml_pickle()), "model.pt").to_dict()
        self.assertEqual(rep["verdict"], "clean", rep)

    def test_numeric_npy_is_clean(self):
        rep = scanner.scan_bytes(npy_bytes("<f8", b"\x00" * 8), "a.npy").to_dict()
        self.assertEqual(rep["verdict"], "clean", rep)

    def test_object_npy_with_evil_payload_is_dangerous(self):
        rep = scanner.scan_bytes(npy_bytes("|O", evil_eval_pickle()), "a.npy").to_dict()
        self.assertEqual(rep["verdict"], "dangerous", rep)

    def test_safetensors_clean(self):
        rep = scanner.scan_bytes(safetensors_bytes(), "m.safetensors").to_dict()
        self.assertEqual(rep["verdict"], "clean", rep)

    def test_safetensors_trailing_data_is_caution(self):
        rep = scanner.scan_bytes(safetensors_bytes(b"EXTRA"), "m.safetensors").to_dict()
        self.assertEqual(rep["verdict"], "caution", rep)


class TestDirectoryScan(unittest.TestCase):
    def test_directory_aggregation(self):
        with tempfile.TemporaryDirectory() as td:
            Path(td, "good.pkl").write_bytes(benign_pickle())
            Path(td, "bad.pkl").write_bytes(evil_system_pickle())
            Path(td, "ignore.txt").write_text("not an artifact")
            out = scanner.scan_directory(td)
            self.assertEqual(out["files_scanned"], 2)
            self.assertEqual(out["verdict"], "dangerous")
            self.assertEqual(out["counts"]["clean"], 1)
            self.assertEqual(out["counts"]["dangerous"], 1)


# ---------------------------------------------------------------------------
# Plugin wiring tests
# ---------------------------------------------------------------------------

class FakeCtx:
    def __init__(self):
        self.tools = {}
        self.hooks = {}

    def register_tool(self, name, toolset, schema, handler):
        self.tools[name] = (toolset, schema, handler)

    def register_hook(self, name, callback):
        self.hooks[name] = callback

    def get_config(self, key, default=None):
        return default


class TestPluginWiring(unittest.TestCase):
    def test_registration_matches_manifest(self):
        ctx = FakeCtx()
        pkg.register(ctx)
        self.assertEqual(set(ctx.tools), {"artifact_scan", "artifact_audit_plugin"})
        self.assertEqual(set(ctx.hooks), {"pre_tool_call"})

    def test_tool_handler_round_trip(self):
        with tempfile.TemporaryDirectory() as td:
            target = Path(td, "w.pkl")
            target.write_bytes(evil_system_pickle())
            out = json.loads(pkg.tools.artifact_scan({"path": str(target)}))
            self.assertEqual(out["verdict"], "dangerous")

    def test_audit_handler(self):
        with tempfile.TemporaryDirectory() as td:
            Path(td, "w.safetensors").write_bytes(safetensors_bytes())
            out = json.loads(pkg.tools.artifact_audit_plugin({"path": td}))
            self.assertEqual(out["verdict"], "clean")
            self.assertEqual(out["files_scanned"], 1)


class TestGuardHook(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.evil = Path(self.td.name, "evil.pkl")
        self.evil.write_bytes(evil_system_pickle())
        self.good = Path(self.td.name, "good.pkl")
        self.good.write_bytes(benign_pickle())

    def tearDown(self):
        self.td.cleanup()

    def test_blocks_torch_load_of_evil_artifact(self):
        hook = pkg._guard_pre_tool_call
        result = hook("execute_code",
                      {"code": f"import torch\nm = torch.load({str(self.evil)!r})"})
        self.assertIsInstance(result, dict)
        self.assertEqual(result["action"], "block")

    def test_blocks_terminal_pickle_load_of_evil_artifact(self):
        hook = pkg._guard_pre_tool_call
        result = hook("terminal",
                      {"command": f"python3 -c \"import pickle; pickle.load(open({str(self.evil)!r},'rb'))\""})
        self.assertIsInstance(result, dict)
        self.assertEqual(result["action"], "block")

    def test_allows_benign_load(self):
        hook = pkg._guard_pre_tool_call
        result = hook("execute_code",
                      {"code": f"import torch\nm = torch.load({str(self.good)!r})"})
        self.assertIsNone(result)

    def test_ignores_plain_file_reads(self):
        hook = pkg._guard_pre_tool_call
        result = hook("read_file", {"path": str(self.evil)})
        self.assertIsNone(result)

    def test_ignores_unrelated_tool_calls(self):
        hook = pkg._guard_pre_tool_call
        result = hook("web_search", {"query": "pickle security"})
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
