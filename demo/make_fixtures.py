#!/usr/bin/env python3
"""Create the demo workspace: a realistic mix of artifacts an agent might
download. The two hostile files are generated here, in the open, and are
only ever scanned — nothing in the demo loads them.
"""
import io
import json
import os
import pickle
import struct
import sys
import zipfile
from collections import OrderedDict
from pathlib import Path

WS = Path(__file__).resolve().parent.parent / "demo-workspace"


class _ReduceTo:
    def __init__(self, func, arg):
        self.func, self.arg = func, arg

    def __reduce__(self):
        return (self.func, (self.arg,))


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


def safetensors_bytes() -> bytes:
    header = json.dumps({"w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    return struct.pack("<Q", len(header)) + header + b"\x00\x00\x00\x00"


def main() -> None:
    WS.mkdir(parents=True, exist_ok=True)
    benign = pickle.dumps(OrderedDict([("layer.weight", [0.1, 0.2])]), protocol=4)
    files = {
        # clean, realistic artifacts
        "model-good.pt": torch_zip(benign),
        "weights.pkl": benign,
        "model.safetensors": safetensors_bytes(),
        # hostile: a "checkpoint" whose data.pkl runs a shell command on load
        "model-trojan.pt": torch_zip(pickle.dumps(_ReduceTo(os.system, "echo pwned"), protocol=2)),
        # hostile: object-dtype .npy hiding an eval() payload in its data section
        "features.npy": npy_bytes("|O", pickle.dumps(_ReduceTo(eval, "__import__('os').system('id')"), protocol=4)),
    }
    for name, data in files.items():
        (WS / name).write_bytes(data)
        print(f"  wrote {name} ({len(data)} bytes)")
    print(f"Demo workspace ready: {WS}")


if __name__ == "__main__":
    sys.exit(main())
