"""PDF toolbox: read a paper, crop its figures/tables/excerpts with coordinate metadata.

Every image this tool writes gets a sidecar `<name>.json`:
    {"pdf", "page" (1-based), "clip": [x0,y0,x1,y1] in PDF points, "dpi", "size": [w_px,h_px],
     "kind", "label", "caption", "words": [[u0,v0,u1,v1,"text"], ...], "highlights": [[u0,v0,u1,v1], ...]}
(u, v) are normalized image coordinates: u = 0 left .. 1 right, v = 0 top .. 1 bottom.
`vk.engine.PaperImage` reads this sidecar so scenes can point at words, regions and highlights.

Run `./vk paper -h` for the commands.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

import pymupdf

CAPTION = re.compile(r"^\s*(Figure|Fig\.|Table|Algorithm|Listing)\s*([A-Z]?\d+)\s*[:.|]", re.I)
PAD = 3.0  # points of padding around crops


# ---------------------------------------------------------------- helpers
def open_pdf(path: str) -> pymupdf.Document:
    return pymupdf.open(path)


def parse_pages(spec: str | None, n: int) -> list[int]:
    """'1-3,7' -> [0,1,2,6] (0-based). None -> all."""
    if not spec:
        return list(range(n))
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a) - 1, min(int(b), n)))
        else:
            out.append(int(part) - 1)
    return [p for p in out if 0 <= p < n]


def block_text(block: dict) -> str:
    return " ".join("".join(s["text"] for s in line["spans"]) for line in block["lines"]).strip()


def block_size(block: dict) -> float:
    sizes = [s["size"] for line in block["lines"] for s in line["spans"] if s["text"].strip()]
    return statistics.median(sizes) if sizes else 0.0


def is_prose(block: dict, body: float) -> bool:
    words = block_text(block).split()
    alpha = sum(w.strip(".,;:()").isalpha() for w in words) / max(1, len(words))
    return abs(block_size(block) - body) < 0.8 and alpha > 0.75 and len(words) >= 6


def body_font_size(doc: pymupdf.Document) -> float:
    weights: dict[float, int] = {}
    for page in list(doc)[: min(len(doc), 12)]:
        for b in page.get_text("dict")["blocks"]:
            if b["type"] != 0:
                continue
            for line in b["lines"]:
                for s in line["spans"]:
                    k = round(s["size"], 1)
                    weights[k] = weights.get(k, 0) + len(s["text"])
    return max(weights, key=weights.get) if weights else 10.0


def words_in(page: pymupdf.Page, clip: pymupdf.Rect) -> list[list]:
    out = []
    for x0, y0, x1, y1, w, *_ in page.get_text("words", clip=clip):
        out.append([round((x0 - clip.x0) / clip.width, 4), round((y0 - clip.y0) / clip.height, 4),
                    round((x1 - clip.x0) / clip.width, 4), round((y1 - clip.y0) / clip.height, 4), w])
    return out


def norm_rect(r: pymupdf.Rect, clip: pymupdf.Rect) -> list[float]:
    return [round((r.x0 - clip.x0) / clip.width, 4), round((r.y0 - clip.y0) / clip.height, 4),
            round((r.x1 - clip.x0) / clip.width, 4), round((r.y1 - clip.y0) / clip.height, 4)]


def save_crop(doc, pno: int, clip: pymupdf.Rect, out: Path, *, dpi: int, kind: str, label: str = "",
              caption: str = "", highlights: list | None = None, svg: bool = False) -> Path:
    page = doc[pno]
    clip = clip & page.rect
    out.parent.mkdir(parents=True, exist_ok=True)
    png = out.with_suffix(".png")
    pix = page.get_pixmap(dpi=dpi, clip=clip, alpha=False)
    pix.save(str(png))
    meta = {"pdf": str(Path(doc.name).resolve()), "page": pno + 1, "clip": [round(v, 2) for v in clip],
            "page_size": [page.rect.width, page.rect.height], "dpi": dpi, "size": [pix.width, pix.height],
            "kind": kind, "label": label, "caption": caption, "words": words_in(page, clip),
            "highlights": [norm_rect(h, clip) for h in (highlights or [])]}
    if svg:
        meta["svg"] = str(save_svg(doc, pno, clip, out.with_suffix(".svg")))
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    return png


def save_svg(doc, pno: int, clip: pymupdf.Rect, out: Path) -> Path:
    """Vector copy of a region (text converted to paths) — load with SVGMobject to animate parts."""
    tmp = pymupdf.open()
    page = tmp.new_page(width=clip.width, height=clip.height)
    page.show_pdf_page(page.rect, doc, pno, clip=clip)
    out.write_text(page.get_svg_image(text_as_path=True))
    return out


# ---------------------------------------------------------------- figure detection
def graphics_rects(page: pymupdf.Page) -> list[pymupdf.Rect]:
    area = page.rect.width * page.rect.height
    rects = []
    scissors: dict[int, pymupdf.Rect] = {}
    for d in page.get_drawings(extended=True):
        level = d.get("level", 0)
        for k in [k for k in scissors if k >= level]:
            del scissors[k]
        if d.get("type") in ("clip", "group"):
            if d.get("scissor") is not None:
                scissors[level] = pymupdf.Rect(d["scissor"])
            continue
        r = pymupdf.Rect(d["rect"])
        for sc in scissors.values():  # what is actually visible through the active clip paths
            r = r & sc if not r.is_empty else pymupdf.Rect(max(r.x0, sc.x0), max(r.y0, sc.y0), min(r.x1, sc.x1), min(r.y1, sc.y1))
        if r.x0 > r.x1 or r.y0 > r.y1 or not page.rect.intersects(r) and not page.rect.contains(r.tl):
            continue
        if d.get("fill") == (1.0, 1.0, 1.0) and d.get("color") is None:
            continue  # white background patches
        if r.is_empty and (r.width < 0.5 and r.height < 0.5):
            continue
        if r.width * r.height > 0.9 * area:
            continue
        rects.append(r)
    for info in page.get_image_info():
        r = pymupdf.Rect(info["bbox"])
        if r.width > 4 and r.height > 4:
            rects.append(r)
    return rects


def detect(doc: pymupdf.Document, pages: list[int]) -> list[dict]:
    """Heuristic figure/table regions: graphics + small text between a caption and the nearest body paragraph."""
    body = body_font_size(doc)
    found: list[dict] = []
    for pno in pages:
        page = doc[pno]
        W, H = page.rect.width, page.rect.height
        blocks = [b for b in page.get_text("dict")["blocks"] if b["type"] == 0 and block_text(b)]
        gfx = graphics_rects(page)
        caps, paras = [], []
        for b in blocks:
            t = block_text(b)
            r = pymupdf.Rect(b["bbox"])
            m = CAPTION.match(t)
            if m:
                caps.append((m, r, t))
                continue
            boxed = any(g.contains(r) or (g & r).get_area() > 0.6 * r.get_area() for g in gfx if g.width > 30 and g.height > 15)
            chars = len(t)
            if (not boxed and len(b["lines"]) >= 2 and abs(block_size(b) - body) < 0.8 and r.width > 0.3 * W
                    and chars / max(1, len(b["lines"])) > 40):
                paras.append(r)
        # page header/footer bands
        top_limit, bottom_limit = 0.06 * H, 0.94 * H
        claimed: list[pymupdf.Rect] = []
        # tables (caption above) first, so a figure stacked under a table cannot swallow it
        caps.sort(key=lambda c: 0 if c[0].group(1).lower() == "table" else 1)
        for m, cap, text in caps:
            kind = m.group(1).lower().replace("fig.", "figure")
            label = f"{kind[:3]}{m.group(2)}" if kind not in ("figure", "table") else f"{'fig' if kind == 'figure' else 'tab'}{m.group(2)}"
            cx = (cap.x0 + cap.x1) / 2
            if cap.width > 0.55 * W or abs(cx - W / 2) < 0.12 * W:
                col = (0.0, W)
            elif cx < W / 2:
                col = (0.0, W / 2 + 0.02 * W)
            else:
                col = (W / 2 - 0.02 * W, W)
            others = [c for _, c, _ in caps if c != cap]

            def collect(above: bool) -> pymupdf.Rect:
                if above:
                    lim = max([p.y1 for p in paras if p.y1 <= cap.y0 + 1 and p.x1 > col[0] and p.x0 < col[1]] +
                              [c.y1 for c in others if c.y1 <= cap.y0 and c.x1 > col[0] and c.x0 < col[1]] +
                              [c.y1 for c in claimed if c.y1 <= cap.y0 and c.x1 > col[0] and c.x0 < col[1]] + [top_limit])
                    window = pymupdf.Rect(col[0], lim, col[1], cap.y0)
                else:
                    lim = min([p.y0 for p in paras if p.y0 >= cap.y1 - 1 and p.x1 > col[0] and p.x0 < col[1]] +
                              [c.y0 for c in others if c.y0 >= cap.y1 and c.x1 > col[0] and c.x0 < col[1]] +
                              [c.y0 for c in claimed if c.y0 >= cap.y1 and c.x1 > col[0] and c.x0 < col[1]] + [bottom_limit])
                    window = pymupdf.Rect(col[0], cap.y1, col[1], lim)
                if window.is_empty or window.height < 8:
                    return pymupdf.Rect(), None
                inside = [g for g in gfx if g.width < 0.97 * W and
                          ((g & window).get_area() > 0.5 * max(g.get_area(), 1e-3) or (g.height < 1 and window.contains(g)))]
                inside += [pymupdf.Rect(b["bbox"]) for b in blocks if pymupdf.Rect(b["bbox"]) != cap
                           and window.contains(pymupdf.Rect(b["bbox"]))
                           and (not is_prose(b, body) or pymupdf.Rect(b["bbox"]).width < 0.4 * W)]   # table cells
                # keep the contiguous run of elements next to the caption (stop at a big gap)
                inside.sort(key=lambda r: r.y0 if not above else -r.y1)
                region, edge = pymupdf.Rect(), (cap.y0 if above else cap.y1)
                first_gap = None
                for r in inside:
                    gap = (edge - r.y1) if above else (r.y0 - edge)
                    if gap > (12 if not region.is_empty else 30):
                        break
                    if first_gap is None:
                        first_gap = max(gap, 0.0)
                    region |= r
                    edge = min(edge, r.y0) if above else max(edge, r.y1)
                if above and not region.is_empty:
                    region.y1 = min(region.y1, cap.y0 - 1)
                ok = not region.is_empty and region.height >= 8 and region.width >= 8
                return (region, first_gap) if ok else (pymupdf.Rect(), None)

            # figures usually sit above their caption, tables below; take the usual side unless the other side's
            # content hugs the caption clearly closer (caption-below tables, caption-above figures)
            first = kind != "table"
            r1, g1 = collect(first)
            r2, g2 = collect(not first)
            if r1.is_empty or (g2 is not None and g2 + 4 < g1):
                region = r2
            else:
                region = r1
            if region.is_empty:
                continue
            region = pymupdf.Rect(region.x0 - PAD, region.y0 - PAD, region.x1 + PAD, region.y1 + PAD) & page.rect
            claimed.append(region)
            found.append({"label": label, "kind": kind, "page": pno, "rect": region, "caption_rect": cap, "caption": text})
    found.sort(key=lambda f: (f["page"], f["rect"].y0))
    return found


# ---------------------------------------------------------------- commands
def cmd_info(a):
    doc = open_pdf(a.pdf)
    first = doc[0].get_text("dict")["blocks"]
    spans = [(s["size"], s["text"]) for b in first if b["type"] == 0 for l in b["lines"] for s in l["spans"] if s["text"].strip()]
    big = max(s for s, _ in spans) if spans else 0
    title = " ".join(t for s, t in spans if s >= big - 0.5).strip()
    print(f"file: {a.pdf}\npages: {len(doc)}\ntitle (largest font, page 1): {title}\nbody font: {body_font_size(doc)} pt")
    toc = doc.get_toc()
    if toc:
        print("\noutline (PDF bookmarks):")
        pages_of = {}
        for lvl, name, pg in toc:
            print(f"{'  ' * (lvl - 1)}- {name}  (p{pg})")
            pages_of[name] = pg
    print("\ncaptions:")
    for pno, page in enumerate(doc):
        for b in page.get_text("dict")["blocks"]:
            if b["type"] == 0:
                t = block_text(b)
                if CAPTION.match(t):
                    print(f"  p{pno + 1:<3} {t[:150]}")


def cmd_text(a):
    doc = open_pdf(a.pdf)
    out = []
    for pno in parse_pages(a.pages, len(doc)):
        out.append(f"\n=== page {pno + 1} ===\n")
        out.append(doc[pno].get_text("text", sort=True))
    text = "".join(out)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text)
        print(f"wrote {a.out} ({len(text.split())} words)")
    else:
        sys.stdout.write(text)


def cmd_figures(a):
    doc = open_pdf(a.pdf)
    outdir = Path(a.out)
    items = detect(doc, parse_pages(a.pages, len(doc)))
    seen: dict[str, int] = {}
    by_page: dict[int, list] = {}
    for it in items:
        name = it["label"]
        seen[name] = seen.get(name, 0) + 1
        if seen[name] > 1:
            name = f"{name}_{seen[name]}"
        png = save_crop(doc, it["page"], it["rect"], outdir / name, dpi=a.dpi, kind=it["kind"], label=it["label"],
                        caption=it["caption"], svg=a.svg)
        by_page.setdefault(it["page"], []).append((name, it))
        print(f"{name:10} p{it['page'] + 1:<3} {png}  bbox={[round(v) for v in it['rect']]}  {it['caption'][:70]}")
    # verification overlays: what was detected on each page
    det = outdir / "_detect"
    det.mkdir(parents=True, exist_ok=True)
    for pno, lst in by_page.items():
        page = doc[pno]
        shape = page.new_shape()
        for name, it in lst:
            shape.draw_rect(it["rect"])
            shape.finish(color=(1, 0, 0), width=1.5)
            shape.insert_text(it["rect"].tl + (2, 9), name, fontsize=9, color=(1, 0, 0))
        tmp = pymupdf.open()
        tmp.insert_pdf(doc, from_page=pno, to_page=pno)
        p2 = tmp[0]
        for name, it in lst:
            p2.draw_rect(it["rect"], color=(1, 0, 0), width=1.5)
            p2.insert_text(it["rect"].tl + (2, 9), name, fontsize=9, color=(1, 0, 0))
        p2.get_pixmap(dpi=80).save(str(det / f"p{pno + 1:02}.png"))
    print(f"\n{len(items)} regions. CHECK the overlays in {det}/ (Read them); fix bad ones with `vk paper crop`.")


def cmd_crop(a):
    doc = open_pdf(a.pdf)
    pno = a.page - 1
    page = doc[pno]
    x0, y0, x1, y1 = [float(v) for v in a.bbox.split(",")]
    if a.units == "frac":
        x0, x1 = x0 * page.rect.width, x1 * page.rect.width
        y0, y1 = y0 * page.rect.height, y1 * page.rect.height
    png = save_crop(doc, pno, pymupdf.Rect(x0, y0, x1, y1), Path(a.out), dpi=a.dpi, kind=a.kind, label=a.label or Path(a.out).stem,
                    svg=a.svg)
    print(png)


def cmd_page(a):
    doc = open_pdf(a.pdf)
    pno = a.page - 1
    out = Path(a.out or f"page{a.page:02}")
    print(save_crop(doc, pno, doc[pno].rect, out, dpi=a.dpi, kind="page", label=f"page {a.page}"))


def search(doc, query: str, pages: list[int]) -> list[tuple[int, list[pymupdf.Rect]]]:
    """Case-insensitive search tolerant of line breaks/hyphenation; returns (page, [line rects]) per hit.
    Word-sequence matching first (it covers every word of a phrase that wraps over several lines); PyMuPDF's
    search_for only as a fallback (it sometimes returns boxes for part of a wrapped phrase)."""
    hits = []
    q = re.sub(r"\s+", " ", query.strip())
    qt = [re.sub(r"[^\w@%.\-]", "", t).lower() for t in q.split()]
    qt = [t for t in qt if t]
    target = "".join(qt).replace("-", "")
    for pno in pages:
        page = doc[pno]
        found = False
        if target:
            words = page.get_text("words", sort=True)
            toks = [re.sub(r"[^\w@%.\-]", "", w[4]).lower().replace("-", "") for w in words]
            i = 0
            while i < len(toks):
                if not toks[i] or not target.startswith(toks[i]):
                    i += 1
                    continue
                acc, j = "", i
                while j < len(toks) and len(acc) < len(target):
                    acc += toks[j]
                    j += 1
                    if not target.startswith(acc):
                        break
                if acc == target:
                    lines: list[pymupdf.Rect] = []
                    for k in range(i, j):
                        r = pymupdf.Rect(words[k][:4])
                        if lines and abs(r.y0 - lines[-1].y0) < 3 and abs(r.y1 - lines[-1].y1) < 4:
                            lines[-1] |= r
                        else:
                            lines.append(r)
                    hits.append((pno, lines))
                    found = True
                    i = j
                else:
                    i += 1
        if found:
            continue
        quads = page.search_for(q, quads=False)
        if quads:
            group: list[pymupdf.Rect] = []
            for r in quads:
                if group and r.y0 > group[-1].y1 + 14:
                    hits.append((pno, group))
                    group = []
                group.append(r)
            if group:
                hits.append((pno, group))
    return hits


def cmd_find(a):
    doc = open_pdf(a.pdf)
    hits = search(doc, a.query, parse_pages(a.pages, len(doc)))
    for pno, rects in hits:
        r = rects[0]
        for x in rects[1:]:
            r = r | x
        ctx = doc[pno].get_text("text", clip=pymupdf.Rect(0, r.y0 - 12, doc[pno].rect.width, r.y1 + 12)).replace("\n", " ")
        print(f"p{pno + 1}  bbox={[round(v, 1) for v in r]}  …{ctx[:160]}…")
    if not hits:
        print("no hits (try a shorter phrase; ligatures like 'fi' or math may differ)")


def cmd_excerpt(a):
    doc = open_pdf(a.pdf)
    hits = search(doc, a.query, parse_pages(a.pages, len(doc)))
    if not hits:
        sys.exit(f"no hit for {a.query!r}")
    pno, rects = hits[a.hit - 1]
    page = doc[pno]
    hit = rects[0]
    for r in rects[1:]:
        hit |= r
    if a.context == "hit":
        clip = pymupdf.Rect(hit)
    else:
        blocks = [pymupdf.Rect(b[:4]) for b in page.get_text("blocks") if pymupdf.Rect(b[:4]).intersects(hit)]
        clip = pymupdf.Rect(hit)
        for b in blocks:
            clip |= b
        if a.context == "lines":
            clip = pymupdf.Rect(clip.x0, hit.y0 - a.lines * 12, clip.x1, hit.y1 + a.lines * 12)
    clip = pymupdf.Rect(clip.x0 - a.pad, clip.y0 - a.pad, clip.x1 + a.pad, clip.y1 + a.pad)
    hl = []
    for r in rects:
        hl.append(pymupdf.Rect(r.x0 - 0.5, r.y0 - 0.5, r.x1 + 0.5, r.y1 + 0.5))
    png = save_crop(doc, pno, clip, Path(a.out), dpi=a.dpi, kind="excerpt", label=a.query[:60], highlights=hl)
    print(f"{png}  (p{pno + 1}, {len(hl)} highlight line(s))")


def cmd_grid(a):
    """Overlay a labeled normalized grid on a crop so you can read off (u, v) coordinates by eye."""
    from PIL import Image, ImageDraw, ImageFont
    src = Path(a.png)
    im = Image.open(src).convert("RGB")
    W, H = im.size
    draw = ImageDraw.Draw(im, "RGBA")
    fs = max(12, min(W, H) // 45)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", fs)
    except OSError:
        font = ImageFont.load_default()
    n = a.n
    for i in range(n + 1):
        x, y = round((W - 1) * i / n), round((H - 1) * i / n)
        major = i % 5 == 0
        col = (230, 30, 30, 170) if major else (40, 100, 255, 110)
        draw.line([(x, 0), (x, H)], fill=col, width=3 if major else 1)
        draw.line([(0, y), (W, y)], fill=col, width=3 if major else 1)
    for i in range(0, n, 2):
        lab = f"{i / n:.2f}".lstrip("0") if i else "0"
        x, y = round(W * i / n), round(H * i / n)
        for pos in [(x + 3, 2), (x + 3, H - fs - 4)]:
            draw.text(pos, lab, fill=(220, 0, 0, 255), font=font, stroke_width=2, stroke_fill=(255, 255, 255))
        for pos in [(3, y + 2), (W - fs * 3, y + 2)]:
            draw.text(pos, lab, fill=(220, 0, 0, 255), font=font, stroke_width=2, stroke_fill=(255, 255, 255))
    if max(W, H) > 2400:
        im.thumbnail((2400, 2400))
    dst = src.with_name(src.stem + "_grid.png")
    im.save(dst)
    print(dst)


def cmd_words(a):
    meta = json.loads(Path(a.json).read_text())
    pat = re.compile(a.grep, re.I) if a.grep else None
    for u0, v0, u1, v1, w in meta["words"]:
        if pat is None or pat.search(w):
            print(f"[{u0:.3f}, {v0:.3f}, {u1:.3f}, {v1:.3f}]  {w}")


def cmd_refs(a):
    doc = open_pdf(a.pdf)
    text = "".join(p.get_text("text") for p in doc)
    m = re.search(r"\n\s*(References|Bibliography)\s*\n", text)
    if not m:
        sys.exit("no References heading found")
    tail = text[m.end():]
    end = re.search(r"\n\s*(Appendix|A\s+[A-Z][a-z]+.*\n)", tail)
    sys.stdout.write(tail[: end.start()] if end and end.start() > 2000 else tail[:60000])


def main(argv=None):
    ap = argparse.ArgumentParser(prog="vk paper", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("info", help="page count, title, outline, every caption with its page")
    s.add_argument("pdf"); s.set_defaults(fn=cmd_info)
    s = sub.add_parser("text", help="plain text with '=== page N ===' markers")
    s.add_argument("pdf"); s.add_argument("--pages"); s.add_argument("--out"); s.set_defaults(fn=cmd_text)
    s = sub.add_parser("figures", help="auto-crop every captioned figure/table (+ _detect/ overlays to verify)")
    s.add_argument("pdf"); s.add_argument("--out", required=True); s.add_argument("--pages"); s.add_argument("--dpi", type=int, default=300)
    s.add_argument("--svg", action="store_true", help="also write vector .svg copies"); s.set_defaults(fn=cmd_figures)
    s = sub.add_parser("crop", help="crop an exact region (PDF points, or --units frac of the page)")
    s.add_argument("pdf"); s.add_argument("--page", type=int, required=True); s.add_argument("--bbox", required=True, help="x0,y0,x1,y1")
    s.add_argument("--units", choices=["pt", "frac"], default="pt"); s.add_argument("--out", required=True)
    s.add_argument("--dpi", type=int, default=300); s.add_argument("--kind", default="figure"); s.add_argument("--label", default="")
    s.add_argument("--svg", action="store_true"); s.set_defaults(fn=cmd_crop)
    s = sub.add_parser("page", help="render a whole page (+ word boxes), e.g. for a zoom-from-page shot")
    s.add_argument("pdf"); s.add_argument("--page", type=int, required=True); s.add_argument("--out"); s.add_argument("--dpi", type=int, default=200)
    s.set_defaults(fn=cmd_page)
    s = sub.add_parser("find", help="locate a phrase: page, bbox in points, context")
    s.add_argument("pdf"); s.add_argument("query"); s.add_argument("--pages"); s.set_defaults(fn=cmd_find)
    s = sub.add_parser("excerpt", help="crop the paragraph containing a phrase, with the phrase stored as highlight boxes")
    s.add_argument("pdf"); s.add_argument("query"); s.add_argument("--out", required=True); s.add_argument("--pages")
    s.add_argument("--hit", type=int, default=1, help="which hit (1-based)")
    s.add_argument("--context", choices=["paragraph", "lines", "hit"], default="paragraph")
    s.add_argument("--lines", type=int, default=2, help="with --context lines: lines of context above/below")
    s.add_argument("--pad", type=float, default=6); s.add_argument("--dpi", type=int, default=300); s.set_defaults(fn=cmd_excerpt)
    s = sub.add_parser("grid", help="overlay a labeled 0..1 grid on a PNG to read off (u, v) coordinates")
    s.add_argument("png"); s.add_argument("--n", type=int, default=20); s.set_defaults(fn=cmd_grid)
    s = sub.add_parser("words", help="list word boxes (normalized) inside a crop, optionally filtered by regex")
    s.add_argument("json"); s.add_argument("--grep"); s.set_defaults(fn=cmd_words)
    s = sub.add_parser("refs", help="dump the bibliography text")
    s.add_argument("pdf"); s.set_defaults(fn=cmd_refs)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
