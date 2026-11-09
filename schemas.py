"""Tool schemas — what the LLM sees."""

ARTIFACT_SCAN = {
    "name": "artifact_scan",
    "description": (
        "Statically scan a serialized model artifact for hidden code execution "
        "BEFORE anything loads it. Covers pickle files (.pkl, .pickle, .joblib), "
        "PyTorch checkpoints (.pt, .pth, .ckpt — zip members included), numpy "
        "files (.npy, .npz — object arrays included) and .safetensors. The scan "
        "disassembles pickle opcodes and reports every global the file would "
        "resolve and invoke (os.system, subprocess.Popen, builtins.eval, ...) "
        "without ever unpickling it. Use this whenever a model, checkpoint or "
        "data file comes from an untrusted source (a download, another agent, "
        "a plugin bundle) and before any torch.load / pickle.load / joblib.load."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to a single artifact file to scan.",
            },
        },
        "required": ["path"],
    },
}

ARTIFACT_AUDIT_PLUGIN = {
    "name": "artifact_audit_plugin",
    "description": (
        "Audit a directory (for example a plugin folder about to be installed, "
        "a downloaded model bundle, or a workspace) for serialized artifacts: "
        "recursively finds every .pkl/.pt/.pth/.joblib/.npy/.npz/.safetensors "
        "file, scans each one with artifact_scan's engine, and returns an "
        "aggregate verdict with per-file reports. Use before installing a "
        "plugin that bundles data files, or when handing a directory of "
        "artifacts between agents."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the directory to audit recursively.",
            },
        },
        "required": ["path"],
    },
}
