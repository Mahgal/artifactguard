#!/usr/bin/env python3
"""Demo driver: exercises ArtifactGuard's tools and guard hook directly,
the same code paths Hermes calls. Prints a narrated, terminal-friendly
transcript. Usage: python3 demo/run_demo.py
"""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WS = ROOT / "demo-workspace"

spec = importlib.util.spec_from_file_location(
    "artifactguard", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
pkg = importlib.util.module_from_spec(spec)
sys.modules["artifactguard"] = pkg
spec.loader.exec_module(pkg)


def hr(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def main() -> None:
    hr("STEP 1 — Audit a folder of downloaded artifacts before touching them")
    print("$ artifact_audit_plugin(path='demo-workspace/')")
    audit = json.loads(pkg.tools.artifact_audit_plugin({"path": str(WS)}))
    print(f"files scanned: {audit['files_scanned']}   aggregate verdict: {audit['verdict'].upper()}")
    for rep in audit["results"]:
        name = Path(rep["path"]).name
        bad = [f"{g['module']}.{g['name']}" for g in rep["globals"]
               if g["severity"] in ("critical", "high")]
        for member in rep.get("members", []):
            bad += [f"{g['module']}.{g['name']}" for g in member.get("globals", [])
                    if g["severity"] in ("critical", "high")]
        suffix = f"   <-- invokes {', '.join(bad)}" if bad else ""
        print(f"  {name:<20} {rep['verdict']:<10}{suffix}")

    hr("STEP 2 — Full report on the hostile checkpoint")
    print("$ artifact_scan(path='demo-workspace/model-trojan.pt')")
    report = json.loads(pkg.tools.artifact_scan({"path": str(WS / "model-trojan.pt")}))
    print(json.dumps(report, indent=2)[:1400])

    hr("STEP 3 — The agent tries to load it anyway; the guard hook fires")
    code = f"import torch\nmodel = torch.load({str(WS / 'model-trojan.pt')!r})"
    print("$ execute_code:")
    print("    " + code.replace("\n", "\n    "))
    result = pkg._guard_pre_tool_call("execute_code", {"code": code})
    print("-> pre_tool_call verdict:")
    print(json.dumps(result, indent=2))

    hr("STEP 4 — And the clean checkpoint loads unimpeded")
    code = f"import torch\nmodel = torch.load({str(WS / 'model-good.pt')!r})"
    result = pkg._guard_pre_tool_call("execute_code", {"code": code})
    print(f"guard returns: {result!r}  (None = no objection, the load proceeds)")
    print()
    print("ArtifactGuard: the artifact was evidence, never an execution.")


if __name__ == "__main__":
    main()
