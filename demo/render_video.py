#!/usr/bin/env python3
"""Render demo/demo-output.txt as a 1080p terminal-style MP4.

Progressive reveal with a scrolling window; keyword-based coloring.
Output: demo/artifactguard-demo.mp4 (1920x1080, under 90 seconds).
"""
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "demo" / "demo-output.txt"
OUT = ROOT / "demo" / "artifactguard-demo.mp4"
FRAMES = Path("/tmp/ag-frames")

W, H = 1920, 1080
BG = (13, 17, 23)
FG = (201, 209, 217)
GREEN = (63, 185, 80)
RED = (248, 81, 73)
GOLD = (210, 168, 35)
DIM = (139, 148, 158)
FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 25)
FONT_B = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 25)
LINE_H = 34
TOP = 46
LEFT = 48
VISIBLE = (H - TOP - 30) // LINE_H
HOLD_SECONDS = 1.35  # per reveal step (2 lines)


def color_for(line: str):
    if line.startswith("=") or line.startswith("STEP") or line.startswith("###"):
        return GOLD, FONT_B
    if "DANGEROUS" in line or "dangerous" in line and "<--" in line:
        return RED, FONT_B
    if line.strip().startswith("$"):
        return GREEN, FONT_B
    if "blocked" in line or '"block"' in line:
        return RED, FONT
    if "clean" in line:
        return DIM, FONT
    return FG, FONT


def render(lines: list[str], path: Path) -> None:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((LEFT, 12), "ArtifactGuard — demo", font=FONT_B, fill=DIM)
    window = lines[-VISIBLE:]
    y = TOP
    for line in window:
        color, font = color_for(line)
        d.text((LEFT, y), line[:118], font=font, fill=color)
        y += LINE_H
    img.save(path)


def main() -> None:
    raw = SRC.read_text(encoding="utf-8").splitlines()
    # Drop the fixture-prep preamble lines except the header, keep pacing tight.
    lines = [ln.rstrip() for ln in raw]
    FRAMES.mkdir(parents=True, exist_ok=True)
    for old in FRAMES.glob("*.png"):
        old.unlink()
    states = []
    shown: list[str] = []
    for i in range(0, len(lines), 2):
        shown = lines[: i + 2]
        states.append(shown)
    if shown != lines:
        states.append(lines)
    concat = []
    for idx, state in enumerate(states):
        frame = FRAMES / f"f{idx:04d}.png"
        render(state, frame)
        concat.append(f"file '{frame}'\nduration {HOLD_SECONDS}")
    concat.append(f"file '{FRAMES / f'f{len(states) - 1:04d}.png'}'")
    (FRAMES / "concat.txt").write_text("\n".join(concat))
    total = len(states) * HOLD_SECONDS
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(FRAMES / "concat.txt"),
         "-vf", "fps=24,format=yuv420p", "-c:v", "libx264", "-preset", "medium",
         "-movflags", "+faststart", str(OUT)],
        check=True, capture_output=True)
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes, ~{total:.0f}s of states)")


if __name__ == "__main__":
    sys.exit(main())
