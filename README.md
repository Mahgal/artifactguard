# ArtifactGuard

**Scan the artifact before the agent loads it.**

AI agents download model files, checkpoints and data bundles — and then
`torch.load()` / `pickle.load()` them. A pickle is not data; it is a
program for a stack machine, and loading one executes whatever globals
it names: `os.system`, `subprocess.Popen`, `builtins.eval`. Existing
plugin scanners review *source code*. ArtifactGuard reviews the other
half of the supply chain: the **serialized artifacts themselves**.

ArtifactGuard disassembles pickle streams at the opcode level and runs a
small abstract interpreter over them — tracking string constants and
global references — so it can tell you *exactly* which callables an
artifact would resolve and invoke when loaded. **It never unpickles
anything.** A hostile file is evidence, not an execution.

## Do X → see Y

**Do:** install the plugin and ask your agent:

> "Scan `./downloads/model.pt` before you load it."

**See:** the `artifact_scan` tool return a verdict with receipts:

```json
{
  "path": "downloads/model.pt",
  "format": "zip-container",
  "verdict": "dangerous",
  "members": [{
    "path": "downloads/model.pt!archive/data.pkl",
    "format": "pickle",
    "verdict": "dangerous",
    "globals": [
      {"module": "os", "name": "system", "severity": "critical",
       "reason": "shell command execution"}
    ],
    "findings": [
      {"severity": "critical", "kind": "invocation",
       "detail": "os.system is actually invoked via REDUCE"}
    ]
  }]
}
```

**Do:** let the agent try to load a flagged artifact anyway, e.g.
`torch.load("trojan.pkl")` in a code tool.

**See:** the `pre_tool_call` guard hook block the call before it runs:

```text
ArtifactGuard blocked this call: trojan.pkl scanned as DANGEROUS —
os.system is actually invoked via REDUCE. The artifact was NOT loaded.
```

**Do:** point `artifact_audit_plugin` at a plugin directory before
installing it.

**See:** every bundled `.pkl` / `.pt` / `.joblib` / `.npy` / `.npz` /
`.safetensors` file in the tree scanned, with one aggregate verdict —
so a plugin that ships a trojaned "weights" file is caught at the door.

## What it covers

| Format | How it is checked |
|---|---|
| `.pkl`, `.pickle`, `.joblib`, `.ckpt`, legacy `.bin` | full opcode-level abstract interpretation (GLOBAL / STACK_GLOBAL resolution, REDUCE / NEWOBJ / OBJ invocation tracking, extension-registry refs, malformed-stream detection) |
| `.pt`, `.pth` (zip checkpoints) | every embedded pickle member scanned; member inventory reported |
| `.npy` | header parsed; object-dtype arrays (which embed pickles in the data section) have their payloads scanned |
| `.npz` | each member `.npy` checked as above |
| `.safetensors` | header/size consistency; the format cannot carry code, but unexpected trailing bytes are flagged |

Verdicts: **clean** (only expected ML content — numpy/torch rebuild
helpers, `OrderedDict` state dicts), **caution** (unreviewed globals,
dynamically-resolved callables, malformed or truncated streams),
**dangerous** (code-execution / process / dynamic-eval primitives
referenced or invoked).

## The guard hook

ArtifactGuard registers a `pre_tool_call` hook. It stays out of the way
— ordinary file reads and unrelated tools are ignored — and only
inspects calls that would actually deserialize something:

* code/command tools (`terminal`, `execute_code`, …) whose text calls a
  known loader (`torch.load`, `pickle.load`, `joblib.load`,
  `dill.loads`, `numpy.load`, `marshal.loads`, `shelve.open`, …) on an
  artifact path;
* tools whose name indicates loading and that receive an artifact path.

With the default `block_on: dangerous` setting, a dangerous verdict
blocks the call with the findings as the reason. Set `block_on: caution`
in the plugin settings to also block unreviewed or malformed artifacts.

## Disclosure

* **Fully local and offline.** No network calls, no telemetry, no
  analytics, no credentials, no background processes, no self-updating.
* **Reads:** artifact files named in tool arguments, and artifact paths
  named inside load calls intercepted by the hook. Nothing else.
* **Executes:** nothing it scans. Analysis is pure disassembly.
* Dependencies: Python standard library only.

## Design notes

* The rule tables in `scanner.py` are a curated core set (execution
  primitives, high-risk capabilities, known-benign ML globals). Extended
  rule packs are maintained separately and are not required for the core
  scan to be useful.
* The test suite **generates its malicious fixtures in memory at test
  time** and only ever scans them. This repository deliberately ships
  no serialized artifact files — a scanner for hostile payloads should
  not carry any.

## Development

```bash
python -m unittest discover -s tests -v   # 24 tests, stdlib only
./demo/demo.sh                            # narrated end-to-end run
```

## License

MIT — see `LICENSE`.
