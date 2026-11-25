"""Quality checks you can run on your own video (you cannot watch it, so look at it this way).

  ./bin/vk qa contact <mp4> [--every 4] [--start 0] [--end END]   contact sheets (4x4 frames, timestamped) -> qa/
  ./bin/vk qa frames <mp4> 12.5 30 61                               full-res stills at given seconds -> qa/
  ./bin/vk qa key <video_dir> [-q l] [--chapter name] [--kinds stage,bubble]   frames at logged key moments -> qa/
  ./bin/vk qa asr <video_dir> [--scene s03_method]                  transcribe narration clips, flag mispronunciations
  ./bin/vk qa grid <video_dir> "Figure 4" [--sub x0 y0 x1 y1]          figure with a 10% grid, for p.sub(...) fractions
  ./bin/vk qa lint <video_dir>                                      print layout warnings from the last render

Read the PNGs with the Read tool. Check: nothing overlaps or is clipped, text is legible at a glance,
each frame has one clear focal point, consecutive sheets do not look alike.
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import subprocess
import sys
from pathlib import Path

FONT = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"


def probe(path: Path) -> float:
    return float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                          str(path)]).decode().strip())


def qa_dir(mp4: Path) -> Path:
    d = mp4.parent.parent / "qa" if mp4.parent.name == "out" else mp4.parent / "qa"
    d.mkdir(parents=True, exist_ok=True)
    return d


def contact(a) -> None:
    mp4 = Path(a.mp4).resolve()
    dur = probe(mp4)
    end = min(a.end or dur, dur)
    out = qa_dir(mp4)
    per = a.cols * a.rows
    times = []
    t = a.start
    while t < end:
        times.append(t)
        t += a.every
    sheets = [times[i:i + per] for i in range(0, len(times), per)]
    stem = mp4.stem
    for k, ts in enumerate(sheets):
        frames = []
        for t in ts:
            f = out / f".f_{stem}_{t:08.2f}.png"
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{t:.2f}", "-i", str(mp4), "-frames:v", "1",
                            "-vf", "scale=480:-2", str(f)], check=True)
            subprocess.run(["magick", str(f), "-gravity", "NorthWest", "-fill", "#0008", "-draw", "rectangle 0,0 92,26",
                            "-fill", "yellow", "-font", FONT, "-pointsize", "18", "-annotate", "+6+3",
                            f"{int(t // 60)}:{t % 60:05.2f}", str(f)], check=True)
            frames.append(str(f))
        dst = out / f"contact_{stem}_{k + 1:02}.png"
        subprocess.run(["magick", "montage", *frames, "-tile", f"{a.cols}x{a.rows}", "-geometry", "+3+3",
                        "-background", "#333", str(dst)], check=True)
        for f in frames:
            Path(f).unlink()
        print(f"{dst}  ({ts[0]:.0f}s–{ts[-1]:.0f}s)")


def key(a) -> None:
    """Frames at the moments the tools logged (stages, equations, formula terms, bubbles, read-alongs, figures,
    chapter starts), labelled, 3x3 per sheet. Cheaper and more targeted than a full contact sheet."""
    video = Path(a.video).resolve()
    name = video.name + ("" if a.quality == "h" else "_preview")
    mp4 = video / "out" / f"{name}.mp4"
    evs = json.loads((video / "out" / f"{name}_events.json").read_text())
    kinds = set(a.kinds.split(",")) if a.kinds else {"stage", "equation", "formula", "bubble", "read", "figure", "chapter"}
    evs = [e for e in evs if e["kind"] in kinds and (not a.chapter or a.chapter.lower() in e["chapter"].lower())]
    out = qa_dir(mp4)
    for old in out.glob(f"key_{name}_*.png"):
        old.unlink()
    per = 9
    for k in range(0, len(evs), per):
        frames = []
        for e in evs[k:k + per]:
            t = e["t"]
            f = out / f".k_{t:08.2f}.png"
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{t:.2f}", "-i", str(mp4), "-frames:v", "1",
                            "-vf", "scale=640:-2", str(f)], check=True)
            label = f"{int(t // 60)}:{t % 60:04.1f}  {e['kind']}: {e['label']}"[:70]
            subprocess.run(["magick", str(f), "-gravity", "NorthWest", "-fill", "#000a", "-draw", "rectangle 0,0 640,24",
                            "-fill", "yellow", "-font", FONT, "-pointsize", "15", "-annotate", "+6+4", label, str(f)], check=True)
            frames.append(str(f))
        dst = out / f"key_{name}_{k // per + 1:02}.png"
        subprocess.run(["magick", "montage", *frames, "-tile", "3x3", "-geometry", "+3+3", "-background", "#333", str(dst)],
                       check=True)
        print(dst)
    print(f"{len(evs)} key moments")


def grid(a) -> None:
    """A region (figure/table/page) with a labelled 10% grid, for choosing p.sub(region, x0, y0, x1, y1) fractions."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from vk.scroll import PaperScroll
    from PIL import Image, ImageDraw, ImageFont

    video = Path(a.video).resolve()
    p = PaperScroll(str(video / "paper.pdf"))
    what = a.what.strip()
    if what.lower().startswith("page"):
        reg = p.page(int(what.split()[-1]))
    else:
        num = what.split()[-1]
        reg = p.figure(int(num) if num.isdigit() and what.lower().startswith(("fig", "figure")) else what,
                       caption=False)
    if a.sub:
        reg = p.sub(reg, *a.sub)
    pix = p.doc[reg.page - 1].get_pixmap(dpi=a.dpi, clip=reg.pdf_rect, alpha=False)
    im = Image.frombytes("RGB", (pix.w, pix.h), pix.samples)
    W, H = im.size
    d = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype(FONT, max(11, W // 70))
    except OSError:
        font = ImageFont.load_default()
    for i in range(1, 10):
        x, y = W * i / 10, H * i / 10
        c = (255, 0, 160) if i == 5 else (0, 150, 255)
        d.line([(x, 0), (x, H)], fill=c, width=1)
        d.line([(0, y), (W, y)], fill=c, width=1)
        d.text((x + 2, 2), f".{i}", fill=c, font=font)
        d.text((2, y + 1), f".{i}", fill=c, font=font)
    out = video / "qa" / f"grid_{re.sub(r'[^A-Za-z0-9]+', '_', what)}.png"
    out.parent.mkdir(exist_ok=True)
    im.save(out)
    print(out, f"(page {reg.page})")


def frames(a) -> None:
    mp4 = Path(a.mp4).resolve()
    out = qa_dir(mp4)
    for t in a.times:
        dst = out / f"frame_{mp4.stem}_{float(t):07.2f}.png"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", str(t), "-i", str(mp4), "-frames:v", "1", str(dst)],
                       check=True)
        print(dst)


def norm(s: str) -> list[str]:
    """Lowercase words with numbers spelled out and spelled-out acronyms joined ("R L V R" == "RLVR")."""
    from num2words import num2words

    s = s.lower().replace("-", " ").replace("'", "")
    s = re.sub(r"\d+(\.\d+)?", lambda m: " " + num2words(float(m.group()) if "." in m.group() else int(m.group())) + " ", s)
    toks = re.findall(r"[a-z]+", s.replace("-", " "))
    out: list[str] = []
    for t in toks:
        if len(t) == 1 and out and out[-1].startswith("#"):
            out[-1] += t
        elif len(t) == 1 and t not in ("a", "i"):
            out.append("#" + t)
        else:
            out.append(t)
    return [t.lstrip("#") for t in out]


def asr(a) -> None:
    from .slots import slot

    video = Path(a.video).resolve()
    mf = video / "build" / "manifest.json"
    if not mf.exists():
        sys.exit("run `vk build <dir> tts` first")
    items = [m for m in json.loads(mf.read_text()) if m["path"] not in (None, "None")]
    if a.scene:
        items = [m for m in items if m["scene"] in a.scene.split(",")]
    with slot("asr"):
        from parakeet_mlx import from_pretrained

        model = from_pretrained("mlx-community/parakeet-tdt-0.6b-v3")
        rows = []
        for m in items:
            heard = model.transcribe(m["path"]).text.strip()
            r = difflib.SequenceMatcher(None, norm(m["spoken"]), norm(heard)).ratio()
            rows.append({**m, "heard": heard, "ratio": round(r, 3)})
    out = video / "qa" / "asr.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))
    bad = sorted((r for r in rows if r["ratio"] < a.threshold), key=lambda r: r["ratio"])
    print(f"{len(rows)} clips, mean match {sum(r['ratio'] for r in rows) / max(1, len(rows)):.3f}, "
          f"{len(bad)} below {a.threshold}  (full results: {out})")
    for r in bad:
        print(f"\n[{r['scene']}] {r['ratio']:.2f}\n  meant: {r['spoken']}\n  heard: {r['heard']}")
    print("\nNumbers, symbols and names often differ harmlessly; fix real mispronunciations via pronounce.toml or "
          "{display|spoken} / [word](/phonemes/) and re-run tts.")


def lint(a) -> None:
    video = Path(a.video).resolve()
    found = False
    for f in sorted((video / "media").glob("*/vk/*.lint.txt")):
        txt = f.read_text().strip()
        if txt:
            found = True
            print(f"[{f.parent.parent.name}]\n{txt}\n")
    if not found:
        print("no layout warnings")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="vk qa", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("contact"); s.add_argument("mp4"); s.add_argument("--every", type=float, default=4.0)
    s.add_argument("--start", type=float, default=0.0); s.add_argument("--end", type=float)
    s.add_argument("--cols", type=int, default=4); s.add_argument("--rows", type=int, default=4); s.set_defaults(fn=contact)
    s = sub.add_parser("frames"); s.add_argument("mp4"); s.add_argument("times", nargs="+"); s.set_defaults(fn=frames)
    s = sub.add_parser("asr"); s.add_argument("video"); s.add_argument("--scene")
    s.add_argument("--threshold", type=float, default=0.85); s.set_defaults(fn=asr)
    s = sub.add_parser("lint"); s.add_argument("video"); s.set_defaults(fn=lint)
    s = sub.add_parser("key", help="labelled frames at logged key moments (stages, equations, bubbles, ...)")
    s.add_argument("video"); s.add_argument("-q", "--quality", default="l"); s.add_argument("--kinds")
    s.add_argument("--chapter"); s.set_defaults(fn=key)
    s = sub.add_parser("grid", help="a figure/table/page with a 10%% grid, to pick p.sub() fractions")
    s.add_argument("video"); s.add_argument("what", help='"Figure 4", "Table 1" or "page 5"')
    s.add_argument("--sub", type=float, nargs=4); s.add_argument("--dpi", type=int, default=200); s.set_defaults(fn=grid)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main(sys.argv[1:])
