"""Paper walkthrough: the whole PDF as a long strip of sheets that the camera travels over.

    from vk.engine import *
    from vk.scroll import *

    class Walk(PaperScene):                       # PaperScene = VScene + the verbs below
        def construct(self):
            p = self.open_paper("paper.pdf")      # pages stacked top→bottom; camera starts on page 1's top
            self.travel(p.block("Frontier reasoning"), fill=0.9)      # smooth zoom-out/pan/zoom-in arc
            self.highlight(p.find("by pure sampling"))                # marker strokes, line by line
            self.circle(p.find("MATH500"))                            # hand-drawn ellipse
            self.dim(keep=p.figure(2))                                # darken everything else
            snap = self.dive(p.figure(2, caption=False))              # zoom in until the figure fills the frame,
            ...                                                       # then normal manim in default coords
            self.surface(snap)                                        # back onto the paper, zoom out

The paper is drawn live, every frame, by rendering only the visible part of each page with MuPDF at exactly
the output resolution, so text is crisp at any zoom (a single number can fill the screen) and frames
cost the same whatever the zoom. Overlays (highlights, circles, notes, anything you add) live in *paper
world coordinates*: they move with the paper when the camera travels.

Regions.  Every finder returns an invisible `Rectangle` (or a `VGroup` of per-line rectangles for `find`)
placed on the paper, usable as a travel / dim / circle / highlight target:
    p.page(3)                         whole page (1-based)
    p.rect(3, 72, 90, 300, 260)       PDF points on page 3 (x0, y0, x1, y1; `vk paper find` prints these)
    p.rect(3, 0, 0, 1, .5, units="frac")   top half of page 3
    p.find("phrase", page=None, nth=1)     VGroup of line boxes (case-insensitive, survives line breaks)
    p.block("phrase")                 the text paragraph containing the phrase
    p.heading("Conclusion")           a numbered section heading;  p.section("Conclusion") heading → next heading
    p.figure(2) / p.figure("Table 1") figure or table body (+ caption with caption=True)
    p.union(r1, r2, ...)              bounding box of several regions

Screen-space vs world.  While the camera is on the paper, a mobject built in normal frame coordinates
can be placed into the current view with `self.hud(mob)`; `self.stick(mob)` keeps it fixed on screen
while the camera moves (source tags, a chapter label). `freeze()` / `dive()` switch the whole scene to
default coordinates (the paper becomes a full-frame snapshot image you can shrink, slide, fade, or put
in a box next to your animation); `unfreeze()` / `surface()` switch back to the live paper.
"""
from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path

import numpy as np
import pymupdf
from manim import *  # noqa: F403

from . import engine as E
from .engine import BG, DIM, FH, FW, GOLD, INK, MARKER, RED_, T, VScene, callout
from .paper import CAPTION, block_size, block_text, body_font_size, detect, search

SHADOW = "#000000"


def _rgb(hex_: str) -> np.ndarray:
    return (np.array(color_to_rgb(hex_)) * 255).astype(np.uint8)


def _region(x0: float, y0: float, x1: float, y1: float) -> Rectangle:
    """Invisible rectangle from world corners (x0,y0)=top-left, (x1,y1)=bottom-right."""
    r = Rectangle(width=max(1e-4, abs(x1 - x0)), height=max(1e-4, abs(y1 - y0)), stroke_width=0, fill_opacity=0)
    return r.move_to([(x0 + x1) / 2, (y0 + y1) / 2, 0])


# ---------------------------------------------------------------- the paper strip
class PaperScroll(Group):
    """The PDF laid out as a strip of sheets in world coordinates.
    width: page width in scene units (FW = 14.2 is one frame width); gap: space between sheets;
    direction: DOWN (a vertical scroll) or RIGHT (a row of sheets); pages: '1-11' to drop the appendix."""

    def __init__(self, pdf: str | Path, width: float = 8.0, gap: float = 0.45, direction=DOWN,
                 pages: str | None = None, shadow: bool = False):
        super().__init__()
        self.doc = pymupdf.open(str(pdf))
        n = len(self.doc)
        if pages:
            from .paper import parse_pages
            self.pnos = parse_pages(pages, n)
        else:
            self.pnos = list(range(n))
        self.page_w, self.gap, self.shadow = width, gap, shadow   # (not .width: that is the Group bbox)
        self.k = width / self.doc[self.pnos[0]].rect.width   # world units per PDF point
        self.origin: dict[int, np.ndarray] = {}                # pno -> world top-left of the sheet
        x, y = -width / 2, 0.0
        for pno in self.pnos:
            r = self.doc[pno].rect
            self.origin[pno] = np.array([x, y, 0.0])
            if np.allclose(direction, RIGHT):
                x += r.width * self.k + gap
            else:
                y -= r.height * self.k + gap
        self.body = body_font_size(self.doc)
        self._figs = None
        self._cache_key = None
        self._bg = _rgb(BG)
        self._shadow = _rgb(SHADOW)
        self.view = ImageMobject(np.zeros((2, 2, 4), dtype=np.uint8))
        self.view.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
        self.add(self.view)
        self.scene = None

    # coordinates --------------------------------------------------------------
    def _pno(self, page: int) -> int:
        pno = page - 1
        if pno not in self.origin:
            raise KeyError(f"page {page} is not in this scroll")
        return pno

    def world(self, page: int, x: float, y: float) -> np.ndarray:
        """World point of PDF point (x, y) on 1-based `page`."""
        return self.origin[self._pno(page)] + np.array([x * self.k, -y * self.k, 0.0])

    def rect(self, page: int, x0: float, y0: float, x1: float, y1: float, units: str = "pt") -> Rectangle:
        if units == "frac":
            r = self.doc[self._pno(page)].rect
            x0, x1, y0, y1 = x0 * r.width, x1 * r.width, y0 * r.height, y1 * r.height
        a, b = self.world(page, x0, y0), self.world(page, x1, y1)
        reg = _region(a[0], a[1], b[0], b[1])
        reg.page, reg.pdf_rect = page, pymupdf.Rect(x0, y0, x1, y1)
        return reg

    def sub(self, region: Mobject, x0: float, y0: float, x1: float, y1: float) -> Rectangle:
        """Part of a region by fractions of it (0,0 = top-left, 1,1 = bottom-right), e.g. a legend entry or a
        bar inside a raster figure. Read the fractions off `vk grid <dir> "Figure 4"`."""
        r = region.pdf_rect
        return self.rect(region.page, r.x0 + x0 * r.width, r.y0 + y0 * r.height,
                         r.x0 + x1 * r.width, r.y0 + y1 * r.height)

    def _r(self, pno: int, r: pymupdf.Rect, pad: float = 0.0) -> Rectangle:
        return self.rect(pno + 1, r.x0 - pad, r.y0 - pad, r.x1 + pad, r.y1 + pad)

    def page(self, page: int) -> Rectangle:
        r = self.doc[self._pno(page)].rect
        return self.rect(page, 0, 0, r.width, r.height)

    def pages(self, a: int, b: int) -> Rectangle:
        return self.union(self.page(a), self.page(b))

    @staticmethod
    def union(*regions: Mobject) -> Rectangle:
        g = Group(*regions)
        reg = _region(g.get_left()[0], g.get_top()[1], g.get_right()[0], g.get_bottom()[1])
        reg.page = getattr(regions[0], "page", None)
        return reg

    # finders ----------------------------------------------------------------
    def find(self, phrase: str, page: int | None = None, nth: int = 1, pad: float = 1.0) -> VGroup:
        """Line boxes of the nth occurrence of `phrase` (searching `page` only if given)."""
        pages = [self._pno(page)] if page else self.pnos
        hits = search(self.doc, phrase, pages)
        if len(hits) < nth:
            raise KeyError(f"{phrase!r}: {len(hits)} hit(s), wanted #{nth} (check `vk paper find`)")
        pno, rects = hits[nth - 1]
        g = VGroup(*[self._r(pno, r, pad) for r in rects])
        g.page, g.pdf_rects = pno + 1, rects
        return g

    def cell(self, table: Mobject, text: str | None = None, row: str | None = None, col: str | None = None,
             nth: int = 1, pad: float = 1.0) -> VGroup:
        """A cell of a table (or any text inside a region): cell(t, "57.3") finds that text inside the region;
        cell(t, row="Ours", col="MATH500") takes the row label's line and the column header's horizontal span."""
        R = table.pdf_rect
        page = self.doc[table.page - 1]
        words = [(pymupdf.Rect(w[:4]), w[4]) for w in page.get_text("words") if pymupdf.Rect(w[:4]).intersects(R)]

        def locate(s: str, k: int = 1) -> pymupdf.Rect:
            toks = s.split()
            hits = []
            for i in range(len(words) - len(toks) + 1):
                if all(words[i + j][1].strip(".,;:()*") == t.strip(".,;:()*") for j, t in enumerate(toks)):
                    r = pymupdf.Rect(words[i][0])
                    for j in range(1, len(toks)):
                        r |= words[i + j][0]
                    hits.append(r)
            if len(hits) < k:
                raise KeyError(f"{s!r} not found inside the table on p.{table.page} ({len(hits)} hit(s))")
            return hits[k - 1]

        if text is not None:
            r = locate(text, nth)
        elif row is not None and col is not None:
            rr, cc = locate(row), locate(col)
            band = [w for w, _ in words if w.y0 < (rr.y0 + rr.y1) / 2 < w.y1
                    and min(w.x1, cc.x1 + 4) - max(w.x0, cc.x0 - 4) > 0]
            if not band:
                raise KeyError(f"no cell at row {row!r} / column {col!r} on p.{table.page}")
            r = pymupdf.Rect(band[0])
            for w in band[1:]:
                r |= w
        else:
            raise ValueError("cell(table, text) or cell(table, row=..., col=...)")
        g = VGroup(self._r(table.page - 1, r, pad))
        g.page, g.pdf_rects = table.page, [r]
        return g

    def eq(self, n: int | str, tag: bool = False, pad: float = 2.0) -> Rectangle:
        """Numbered display equation `(n)`: finds its tag at the right margin and grows up/down over the adjacent
        maths lines (stopping at prose), across the text column. tag=True includes the "(n)" label."""
        label = f"({n})"
        for pno in self.pnos:
            page = self.doc[pno]
            W = page.rect.width
            tags = [r for r in page.search_for(label) if r.x0 > 0.7 * W]
            if not tags:                  # search_for misses some fonts: look for a line that is just the tag
                tags = [pymupdf.Rect(l["bbox"]) for b in page.get_text("dict")["blocks"] if b["type"] == 0
                        for l in b["lines"] if "".join(sp["text"] for sp in l["spans"]).strip() == label
                        and l["bbox"][0] > 0.7 * W]
            if not tags:
                continue
            t = tags[0]
            lines = []
            for b in page.get_text("dict")["blocks"]:
                if b["type"] != 0:
                    continue
                for l in b["lines"]:
                    txt = "".join(sp["text"] for sp in l["spans"]).strip()
                    if txt:
                        lines.append((pymupdf.Rect(l["bbox"]), txt, l["spans"]))
            col = [r for r, _, _ in lines if r.width > 0.5 * W]
            x_left = min([r.x0 for r in col] + [0.17 * W])

            mathy = ("CM", "Math", "MSBM", "MSAM", "Symbol", "Euler", "rsfs", "STIX", "esint", "bbm", "dsrom")

            def prose(r, txt, spans):
                text_spans = " ".join(sp["text"] for sp in spans if not any(m in sp["font"] for m in mathy)
                                      or "CMR" in sp["font"] and len(sp["text"].split()) > 2)
                words = [w for w in text_spans.split() if w.strip(".,;:()").isalpha() and len(w) > 1]
                return len(words) >= 3

            band = pymupdf.Rect(t)
            mid = (t.y0 + t.y1) / 2
            grew = True
            while grew:
                grew = False
                for r, txt, spans in lines:
                    if r.y1 < band.y0 - 6 or r.y0 > band.y1 + 6 or band.contains(r):
                        continue
                    if prose(r, txt, spans) and not (r.y0 <= mid <= r.y1):
                        continue
                    if r.x1 < x_left or r.width > 0.98 * W:
                        continue
                    band |= r
                    grew = True
            for d in page.get_drawings():            # fraction bars, sqrt signs
                r = pymupdf.Rect(d["rect"])
                if band.y0 - 2 <= (r.y0 + r.y1) / 2 <= band.y1 + 2 and r.x0 >= x_left - 2 and r.width < 0.9 * W:
                    band |= r
            x1 = band.x1 if tag else t.x0 - 4
            body = pymupdf.Rect(x_left, band.y0, x1, band.y1)
            # tighten horizontally to the ink actually in the band
            ink = [r for r, _, _ in lines if r.intersects(body) and r.x0 < x1]
            if ink:
                body.x0 = max(x_left, min(r.x0 for r in ink))
                body.x1 = min(x1, max(min(r.x1, x1) for r in ink))
            return self._r(pno, body, pad)
        raise KeyError(f"equation {label} not found (tags are searched at the right margin)")

    def word_boxes(self, lines: VGroup) -> list[Rectangle]:
        """World boxes of each word inside a `find` hit, in reading order, widened to close the gaps between
        words on a line (so a word-by-word highlight reads as one continuous stroke)."""
        pno = lines.page - 1
        rects = [pymupdf.Rect(r.x0 - 1, r.y0 - 1, r.x1 + 1, r.y1 + 1) for r in lines.pdf_rects]
        ws = []
        for x0, y0, x1, y1, w, b, l, n in self.doc[pno].get_text("words", sort=True):
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            for li, r in enumerate(rects):
                if r.x0 <= cx <= r.x1 and r.y0 <= cy <= r.y1:
                    ws.append((li, x0, y0, x1, y1))
                    break
        ws.sort(key=lambda t: (t[0], t[1]))
        out = []
        for i, (li, x0, y0, x1, y1) in enumerate(ws):
            r = lines.pdf_rects[li]
            if i + 1 < len(ws) and ws[i + 1][0] == li:
                x1 = ws[i + 1][1]
            out.append(self.rect(pno + 1, x0, r.y0, x1, r.y1))
        return out or [self.rect(pno + 1, *r) for r in lines.pdf_rects]

    def block(self, phrase: str, page: int | None = None, nth: int = 1, pad: float = 3.0) -> Rectangle:
        """The text block (paragraph) containing the phrase."""
        hit = self.find(phrase, page, nth)
        pno = hit.page - 1
        c = hit.pdf_rects[0]
        for b in self.doc[pno].get_text("blocks"):
            br = pymupdf.Rect(b[:4])
            if br.intersects(c) and (br & c).get_area() > 0.5 * c.get_area():
                return self._r(pno, br | c, pad)
        return self._r(pno, c, pad)

    def _headings(self) -> list[tuple[int, pymupdf.Rect, str]]:
        if getattr(self, "_heads", None) is not None:
            return self._heads
        pat = re.compile(r"^\s*([A-Z]|\d+)(\.\d+)*\.?\s+[A-Z][A-Za-z]")
        out = []
        for pno in self.pnos:
            for b in self.doc[pno].get_text("dict")["blocks"]:
                if b["type"] != 0 or len(b["lines"]) > 2:
                    continue
                t = block_text(b)
                spans = [s for l in b["lines"] for s in l["spans"] if s["text"].strip()]
                bold = spans and all("Bold" in s["font"] or s["flags"] & 16 for s in spans)
                big = block_size(b) > self.body + 0.5
                r = pymupdf.Rect(b["bbox"])
                lone = t.isupper() and (" " in t or t in ("ABSTRACT", "REFERENCES", "ACKNOWLEDGEMENTS", "ACKNOWLEDGMENTS", "APPENDIX"))
                # numbered small-caps subsections ("3.1 ESTIMATING THE CMP") are set in regular body-size type
                smallcaps = bool(re.match(r"^\s*(\d+|[A-Z])(\.\d+)+\.?\s+\S", t)) and re.sub(r"[^A-Za-z]", "", t).isupper()
                if len(t) < 90 and (pat.match(t) or lone) and (bold or big or smallcaps) and not self._in_figure(pno, r):
                    out.append((pno, pymupdf.Rect(b["bbox"]), t))
        self._heads = out
        return out

    def _in_figure(self, pno: int, r: pymupdf.Rect) -> bool:
        if self._figs is None:
            self._figs = detect(self.doc, self.pnos)
        return any(f["page"] == pno and f["rect"].intersects(r) for f in self._figs)

    def heading(self, title: str, nth: int = 1, pad: float = 3.0) -> Rectangle:
        want = re.sub(r"\s+", " ", title).lower()
        hits = [h for h in self._headings() if want in re.sub(r"\s+", " ", h[2]).lower()]
        if len(hits) < nth:
            raise KeyError(f"no heading matching {title!r}; headings: {[h[2] for h in self._headings()][:30]}")
        pno, r, _ = hits[nth - 1]
        return self._r(pno, r, pad)

    def subsection(self, title: str, nth: int = 1, pad: float = 3.0) -> Rectangle:
        """Same as section(); subsections ("3.1 ...") are headings too."""
        return self.section(title, nth, pad)

    def section(self, title: str, nth: int = 1, pad: float = 3.0) -> Rectangle:
        """From a heading down to the next heading on the same page (or the bottom of the text area)."""
        want = re.sub(r"\s+", " ", title).lower()
        heads = self._headings()
        idx = [i for i, h in enumerate(heads) if want in re.sub(r"\s+", " ", h[2]).lower()]
        if len(idx) < nth:
            raise KeyError(f"no heading matching {title!r}")
        i = idx[nth - 1]
        pno, r, _ = heads[i]
        page = self.doc[pno]
        nxt = [h for h in heads[i + 1:] if h[0] == pno]
        text = [pymupdf.Rect(b[:4]) for b in page.get_text("blocks") if b[1] >= r.y0 - 1]
        bottom = nxt[0][1].y0 - 2 if nxt else max([t.y1 for t in text if t.y1 < 0.93 * page.rect.height] + [r.y1])
        body = [t for t in text if t.y1 <= bottom + 1]
        x0 = min([t.x0 for t in body] + [r.x0])
        x1 = max([t.x1 for t in body] + [r.x1])
        return self._r(pno, pymupdf.Rect(x0, r.y0, x1, bottom), pad)

    def figure(self, which: int | str, caption: bool = True, pad: float = 2.0) -> Rectangle:
        """figure(2), figure("Figure 2"), figure("Table 1"), figure("tab1"), figure("alg1")."""
        if self._figs is None:
            self._figs = detect(self.doc, self.pnos)
        s = str(which).strip().lower()
        m = re.match(r"(fig(?:ure)?|tab(?:le)?|alg(?:orithm)?)?\.?\s*([a-z]?\d+)$", s)
        if not m:
            raise KeyError(which)
        kind = (m.group(1) or "fig")[:3]
        label = f"{kind}{m.group(2)}"
        for f in self._figs:
            if f["label"] == label:
                r = f["rect"] | f["caption_rect"] if caption else f["rect"]
                return self._r(f["page"], r, pad)
        # fallback (wrapped figures, odd tables): grow from the caption over adjacent graphics / non-prose text
        name = {"fig": "Figure", "tab": "Table", "alg": "Algorithm"}[kind]
        for pno in self.pnos:
            for b in self.doc[pno].get_text("blocks"):
                mm = CAPTION.match(b[4].replace("\n", " "))
                if mm and mm.group(1).lower().startswith(name.lower()[:3]) and mm.group(2) == m.group(2):
                    cap = pymupdf.Rect(b[:4])
                    first = kind != "tab"   # figures: body above the caption first; tables: below first
                    body = self._grow(pno, cap, above=first) or self._grow(pno, cap, above=not first)
                    if body is None:
                        raise KeyError(f"{name} {m.group(2)} body not detected (caption at p{pno + 1} "
                                       f"{[round(v) for v in cap]}); use p.rect(page, x0, y0, x1, y1)")
                    return self._r(pno, body | cap if caption else body, pad)
        raise KeyError(f"{label} not found")

    def _grow(self, pno: int, cap: pymupdf.Rect, above: bool) -> pymupdf.Rect | None:
        from .paper import graphics_rects, is_prose
        page = self.doc[pno]
        items = [g for g in graphics_rects(page) if g.width > 2 or g.height > 2]
        items += [pymupdf.Rect(b["bbox"]) for b in page.get_text("dict")["blocks"]
                  if b["type"] == 1 or (b["type"] == 0 and not is_prose(b, self.body) and not CAPTION.match(block_text(b)))]
        items = [r for r in items if (r.y1 <= cap.y0 + 2 if above else r.y0 >= cap.y1 - 2)
                 and min(r.x1, cap.x1) - max(r.x0, cap.x0) > -4 and r.width < page.rect.width * 0.95]
        edge = cap.y0 if above else cap.y1
        region = None
        for _ in range(40):
            grew = False
            for r in items:
                near = (abs((r.y1 if above else r.y0) - edge) < 24) if region is None else \
                    pymupdf.Rect(region.x0 - 8, region.y0 - 10, region.x1 + 8, region.y1 + 10).intersects(r)
                if near and (region is None or not region.contains(r)):
                    region = pymupdf.Rect(r) if region is None else region | r
                    grew = True
            if not grew:
                break
        return region if region is not None and region.height > 12 else None

    # rendering ----------------------------------------------------------------
    def render(self, center: np.ndarray, fw: float, fh: float, W: int, H: int) -> np.ndarray:
        """RGB array of what a camera at `center` with frame size fw×fh sees, at W×H pixels."""
        arr = np.empty((H, W, 3), dtype=np.uint8)
        arr[:] = self._bg
        ppu = W / fw
        left, top = center[0] - fw / 2, center[1] + fh / 2
        z = ppu * self.k  # pixels per PDF point
        for pno in self.pnos:
            page = self.doc[pno]
            o = self.origin[pno]
            px0, py0 = (o[0] - left) * ppu, (top - o[1]) * ppu
            pw, ph = page.rect.width * z, page.rect.height * z
            if px0 > W or py0 > H or px0 + pw < 0 or py0 + ph < 0:
                continue
            if self.shadow:
                s = max(1, int(round(0.006 * pw)))
                sx0, sy0 = max(0, int(px0) + s), max(0, int(py0) + s)
                sx1, sy1 = min(W, int(px0 + pw) + s), min(H, int(py0 + ph) + s)
                if sx1 > sx0 and sy1 > sy0:
                    arr[sy0:sy1, sx0:sx1] = self._shadow
            ix0, iy0 = max(0, math.floor(px0)), max(0, math.floor(py0))
            ix1, iy1 = min(W, math.ceil(px0 + pw)), min(H, math.ceil(py0 + ph))
            if ix1 - ix0 < 1 or iy1 - iy0 < 1:
                continue
            fx, fy = px0 - math.floor(px0), py0 - math.floor(py0)
            clip = pymupdf.Rect((ix0 - px0) / z, (iy0 - py0) / z, (ix1 - px0) / z, (iy1 - py0) / z) & page.rect
            if clip.is_empty:
                continue
            pix = page.get_pixmap(matrix=pymupdf.Matrix(z, 0, 0, z, fx, fy), clip=clip, alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)[..., :3]
            dx, dy = math.floor(px0) + pix.x, math.floor(py0) + pix.y
            ax0, ay0 = max(0, dx), max(0, dy)
            ax1, ay1 = min(W, dx + pix.w, math.floor(px0) + math.ceil(pw + fx)), min(H, dy + pix.h, math.floor(py0) + math.ceil(ph + fy))
            if ax1 > ax0 and ay1 > ay0:
                arr[ay0:ay1, ax0:ax1] = img[ay0 - dy:ay1 - dy, ax0 - dx:ax1 - dx]
        return arr

    def _update(self, view: Mobject) -> None:
        f = self.scene.camera.frame
        W, H = config.pixel_width, config.pixel_height
        c = f.get_center()
        key = (round(c[0], 6), round(c[1], 6), round(f.width, 7), W, H)
        if key == self._cache_key:
            return
        self._cache_key = key
        rgb = self.render(c, f.width, f.height, W, H)
        rgba = np.concatenate([rgb, np.full(rgb.shape[:2] + (1,), 255, np.uint8)], axis=2)
        view.pixel_array = rgba
        view.stretch_to_fit_width(f.width).stretch_to_fit_height(f.height).move_to(c)

    def page_at(self, point) -> int | None:
        """1-based page whose sheet (or the gap after it) contains the world point."""
        best, bd = None, 1e9
        for pno in self.pnos:
            o, r = self.origin[pno], self.doc[pno].rect
            x0, y0, x1, y1 = o[0], o[1] - r.height * self.k, o[0] + r.width * self.k, o[1]
            dx = max(x0 - point[0], 0, point[0] - x1)
            dy = max(y0 - point[1], 0, point[1] - y1)
            if dx + dy < bd:
                best, bd = pno + 1, dx + dy
        return best

    def go_live(self, scene: "PaperScene") -> None:
        self.scene = scene
        self._cache_key = None
        self.view.clear_updaters()
        self.view.add_updater(self._update)
        self._update(self.view)

    def snapshot(self, center, fw: float, sharp: float = 1.0) -> ImageMobject:
        """The current view as a full-frame ImageMobject in default coordinates (FW×FH at ORIGIN)."""
        W, H = int(config.pixel_width * sharp), int(config.pixel_height * sharp)
        rgb = self.render(np.array(center), fw, fw * FH / FW, W, H)
        img = ImageMobject(rgb)
        img.set_resampling_algorithm(RESAMPLING_ALGORITHMS["bicubic" if sharp != 1 else "nearest"])
        img.stretch_to_fit_width(FW).stretch_to_fit_height(FH).move_to(ORIGIN)
        return img

    def text_column(self, page: int) -> tuple[float, float]:
        """(left, right) world x of the page's main text column (from its wide text blocks)."""
        cache = self.__dict__.setdefault("_cols", {})
        if page not in cache:
            pg = self.doc[page - 1]
            W = pg.rect.width
            bl = [b for b in pg.get_text("blocks") if b[6] == 0 and (b[2] - b[0]) > 0.3 * W]
            x0 = min([b[0] for b in bl], default=0.15 * W)
            x1 = max([b[2] for b in bl], default=0.85 * W)
            pr = self.page(page)
            s = pr.width / W
            cache[page] = (pr.get_left()[0] + x0 * s, pr.get_left()[0] + x1 * s)
        return cache[page]

    def crop(self, region: Mobject, dpi: int = 300) -> ImageMobject:
        """A crisp image of a page region (for `dive` hand-offs), placed exactly over the region."""
        pno = region.page - 1
        r = region.pdf_rect
        pix = self.doc[pno].get_pixmap(dpi=dpi, clip=r, alpha=False)
        img = ImageMobject(np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)[..., :3].copy())
        img.set_resampling_algorithm(RESAMPLING_ALGORITHMS["bicubic"])
        img.stretch_to_fit_width(region.width).stretch_to_fit_height(region.height).move_to(region)
        return img


# ---------------------------------------------------------------- camera path
def zoom_path(c0, w0: float, c1, w1: float, rho: float | None = math.sqrt(2)):
    """van Wijk & Nuij 'smooth and efficient' zoom-pan: returns (path(t) -> (center, width), length S).
    rho ~ 1.4 zooms out enough to see both ends mid-flight; rho=None pans straight with log-width."""
    c0, c1 = np.array(c0, dtype=float), np.array(c1, dtype=float)
    d = c1 - c0
    d1 = float(np.linalg.norm(d))
    if rho is None or d1 < 1e-6:
        S = abs(math.log(w1 / w0)) + d1 / max(w0, w1)

        def straight(t):
            return c0 + d * t, w0 * (w1 / w0) ** t
        return straight, S
    r2, r4 = rho * rho, rho ** 4
    b0 = (w1 * w1 - w0 * w0 + r4 * d1 * d1) / (2 * w0 * r2 * d1)
    b1 = (w1 * w1 - w0 * w0 - r4 * d1 * d1) / (2 * w1 * r2 * d1)
    r0 = math.log(math.sqrt(b0 * b0 + 1) - b0)
    r1 = math.log(math.sqrt(b1 * b1 + 1) - b1)
    S = (r1 - r0) / rho

    def arc(t):
        s = t * S
        u = w0 / (r2 * d1) * (math.cosh(r0) * math.tanh(rho * s + r0) - math.sinh(r0))
        return c0 + d * u, w0 * math.cosh(r0) / math.cosh(rho * s + r0)
    return arc, S


def scribble_ellipse(region: Mobject, color: str = RED_, stroke: float = 3, seed: int = 0, pad: float = 0.18,
                     turns: float = 1.1, tilt: float = -0.06) -> VMobject:
    """A hand-drawn-looking ellipse around a region (slightly wobbly, overshoots its start)."""
    rng = random.Random(seed)
    w, h = region.width, region.height
    ry = h / 2 + max(h * pad, min(w, h) * 0.22)
    rx = (w / 2) / math.sqrt(max(0.05, 1 - (h / 2 / ry) ** 2)) * 1.02   # the target's corners stay inside
    tilt *= min(1.0, 3 * h / max(w, 1e-6))   # wide flat targets (a phrase in a line): keep the ellipse level
    t0 = math.radians(rng.uniform(100, 140))
    ph = [rng.uniform(0, 2 * math.pi) for _ in range(3)]
    pts = []
    N = 90
    for i in range(N + 1):
        a = t0 + 2 * math.pi * turns * i / N
        q = i / N
        wob = 1 + 0.035 * math.sin(2 * a + ph[0]) + 0.02 * math.sin(3 * a + ph[1]) + 0.06 * q * q
        x, y = rx * wob * math.cos(a), ry * wob * math.sin(a)
        x, y = x * math.cos(tilt) - y * math.sin(tilt), x * math.sin(tilt) + y * math.cos(tilt)
        pts.append(region.get_center() + np.array([x, y, 0]))
    m = VMobject(stroke_color=color, stroke_width=stroke).set_points_smoothly(pts)
    return m


# ---------------------------------------------------------------- the scene
class PaperScene(VScene):
    """VScene with a paper strip. See the module docstring for the workflow."""

    def setup(self) -> None:
        super().setup()
        self.paper: PaperScroll | None = None
        self._stuck: list[Mobject] = []
        self._frozen = None
        self._top: list[Mobject] = []   # HUD items kept above everything (page counter)
        self._events: list[dict] = []

    def qa_event(self, kind: str, label: str = "", delay: float = 0.0) -> None:
        """Record a moment worth looking at (`vk qa key` grabs a frame there). Tools call this themselves."""
        self._events.append({"t": round(self.renderer.time + delay, 3), "kind": kind, "label": label[:60]})

    def tear_down(self) -> None:
        super().tear_down()
        if config.dry_run:
            return
        out = Path(config.media_dir) / "vk"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{type(self).__name__}.events.json").write_text(json.dumps(self._events, indent=0))

    def add(self, *mobjects):
        super().add(*mobjects)
        top = [m for m in getattr(self, "_top", []) if m in self.mobjects and m not in mobjects]
        if top:
            super().add(*top)
        return self

    # camera state -------------------------------------------------------------
    @property
    def frame(self) -> Mobject:
        return self.camera.frame

    def _set_frame(self, c, w: float) -> None:
        f = self.camera.frame
        f.set_width(w)
        f.move_to(c)

    def scale_k(self) -> float:
        """World units per default-frame unit at the current zoom (multiply screen sizes by this)."""
        return self.camera.frame.width / FW

    def open_paper(self, pdf: str | Path, start: Mobject | None = None, **kw) -> PaperScroll:
        """Add the paper strip; the camera starts at reading zoom on `start` (default: top of page 1)."""
        p = PaperScroll(pdf, **kw)
        self.paper = p
        self.add(p.view)
        self.bring_to_back(p.view)
        tgt = start if start is not None else p.rect(1, 0, 0, 1, 0.4, units="frac")
        w = p.page_w / self.read_frac
        self._set_frame(tgt.get_center() if start is not None else
                        np.array([0, tgt.get_top()[1] - w * FH / FW / 2 + 0.2, 0]), w)
        p.go_live(self)
        return p

    # Camera policy (motion comfort): the frame is never narrower than one page (no zooming into the paper),
    # moves within +-2 pages are straight pans that keep the zoom whenever the target fits, and only longer
    # jumps use the zoom-out/zoom-in arc. Zoom changes only when a target does not fit.
    read_frac = 0.70      # reading zoom: the page fills 70% of the frame width (leaves room for margin notes)
    near_pages = 2

    def _need(self, target: Mobject, fill: float, fit: str) -> float:
        fill = min(max(fill, 0.8), 0.95)      # `fill` only matters for large targets; small ones never zoom in
        fw_from_h = target.height / fill * FW / FH
        fw_from_w = target.width / fill
        w = {"contain": max(fw_from_w, fw_from_h), "width": fw_from_w, "height": fw_from_h}[fit]
        if self.paper is not None:
            w = max(w, self.paper.page_w * 1.02)
        return w

    def _fit(self, target: Mobject, fill: float, fit: str, offset=(0, 0), close: bool = False):
        w = self._need(target, fill, fit)
        c = target.get_center() - np.array([offset[0], offset[1], 0]) * w / FW
        return c, w

    def _plan(self, target: Mobject, fill: float, fit: str, offset=(0, 0)):
        """Where the camera should go for `target` under the comfort policy: (centre, width, arc?)."""
        f = self.camera.frame
        c0, w0 = np.array(f.get_center()), f.width
        need = self._need(target, fill, fit)
        p = self.paper
        far = False
        if p is not None:
            pg0, pg1 = p.page_at(c0) or 1, p.page_at(target.get_center()) or 1
            far = abs(pg1 - pg0) > self.near_pages
        if far:
            w1 = max(need, p.page_w / self.read_frac if p else need)
        else:
            w1 = w0 if need <= w0 * 1.001 else need
        c1 = target.get_center() - np.array([offset[0], offset[1], 0]) * w1 / FW
        if p is not None and not offset[0]:
            c1[0] = 0.0   # the page is always horizontally centred (margin notes shift it only while they are up)
        return c1, w1, far

    def jump(self, target: Mobject, fill: float = 0.85, fit: str = "contain", offset=(0, 0), **_) -> None:
        """Put the camera on `target` instantly, at reading zoom (or wider if the target needs it)."""
        w = max(self._need(target, fill, fit), self.paper.page_w / self.read_frac if self.paper else 0)
        c = target.get_center() - np.array([offset[0], offset[1], 0]) * w / FW
        if self.paper is not None and not offset[0]:
            c[0] = 0.0
        self._set_frame(c, w)
        if self.paper is not None:
            self.paper._update(self.paper.view)

    def travel_anim(self, target: Mobject, fill: float = 0.85, fit: str = "contain", offset=(0, 0),
                    arc: bool | None = None, run_time: float | None = None, rate_func=smooth, **_):
        """The camera move as an Animation (to play alongside others). Follows the comfort policy above."""
        f = self.camera.frame
        c0, w0 = np.array(f.get_center()), f.width
        c1, w1, far = self._plan(target, fill, fit, offset)
        if arc is None:
            arc = far
        path, S = zoom_path(c0, w0, c1, w1, math.sqrt(2) if arc else None)
        if run_time is None:
            dist = float(np.linalg.norm(c1 - c0)) / max(w0, w1)
            run_time = float(np.clip(0.9 + 0.5 * S, 1.0, 4.0)) if arc else float(np.clip(0.6 + 0.9 * dist, 0.6, 2.2))
        self._cam_after = (c1, w1)

        def upd(m, a):
            c, w = path(rate_func(a))
            m.set_width(w)
            m.move_to(c)
        return UpdateFromAlphaFunc(f, upd, run_time=run_time, rate_func=linear)

    def travel(self, target: Mobject, *anims, fill: float = 0.85, fit: str = "contain", offset=(0, 0),
               arc: bool | None = None, run_time: float | None = None, rate_func=smooth, **_) -> None:
        """Move the camera to `target` (comfort policy: pans nearby, arcs only for jumps of more than 2 pages,
        never zooms in past full page width). Does nothing if the target is already comfortably in view."""
        f = self.camera.frame
        c1, w1, far = self._plan(target, fill, fit, offset)
        if (not anims and not far and abs(w1 - f.width) < 1e-6 and abs(f.get_center()[0] - c1[0]) < 1e-3
                and self._in_view(target, margin=0.12)):
            return
        self.play(self.travel_anim(target, fill, fit, offset, arc, run_time, rate_func), *anims)
        self._set_frame(*self._cam_after)

    def _in_view(self, target: Mobject, margin: float = 0.1) -> bool:
        f = self.camera.frame
        c, w, h = f.get_center(), f.width, f.height
        return (target.get_left()[0] >= c[0] - w / 2 * (1 - margin) and target.get_right()[0] <= c[0] + w / 2 * (1 - margin)
                and target.get_bottom()[1] >= c[1] - h / 2 * (1 - margin) and target.get_top()[1] <= c[1] + h / 2 * (1 - margin))

    def scroll(self, target: Mobject, run_time: float, fill: float = 0.85, fit: str = "width", offset=(0, 0),
               rate_func=smooth) -> None:
        """Read-along pan at roughly constant zoom (e.g. down a column while narration runs)."""
        self.travel(target, fill=fill, fit=fit, offset=offset, arc=False, run_time=run_time, rate_func=rate_func)

    # overlays -------------------------------------------------------------------
    def highlight(self, lines: Mobject, color: str = MARKER, opacity: float = 0.42, run_time: float = 1.0,
                  play: bool = True) -> VGroup:
        """Marker strokes over each line box of `p.find(...)` (or over any region)."""
        boxes = lines if isinstance(lines, VGroup) and len(lines) and not isinstance(lines, Rectangle) else VGroup(lines)
        marks = VGroup(*[Rectangle(width=b.width, height=b.height * 1.05, stroke_width=0, fill_color=color,
                                   fill_opacity=opacity).move_to(b) for b in boxes])
        if play:
            self.play(LaggedStart(*[GrowFromEdge(m, LEFT) for m in marks], lag_ratio=0.9), run_time=run_time)
        return marks

    def underline(self, lines: Mobject, color: str = RED_, stroke: float = 3, run_time: float = 0.8,
                  play: bool = True) -> VGroup:
        """A straight rule under each line box, drawn left to right."""
        boxes = lines if isinstance(lines, VGroup) and not isinstance(lines, Rectangle) else VGroup(lines)
        out = VGroup()
        for b in boxes:
            y = b.get_bottom()[1] + b.height * 0.04
            out.add(Line([b.get_left()[0], y, 0], [b.get_right()[0], y, 0], stroke_color=color, stroke_width=stroke))
        if play:
            self.play(LaggedStart(*[Create(u) for u in out], lag_ratio=0.9), run_time=run_time)
        return out

    def circle(self, target: Mobject, color: str = RED_, stroke: float = 3, run_time: float = 0.8, seed: int = 0,
               play: bool = True, **kw) -> VMobject:
        """Hand-drawn ellipse around a small target (a number, a word, a legend entry). A big target (a panel, a
        block) gets a box instead: ellipses around large regions look sloppy and cover the page."""
        k = self.scale_k() if self._frozen is None else 1.0
        if target.height / k > 1.1 or (target.width / k > 4.5 and target.height / k > 0.6):
            return self.frame_box(target, color=color, stroke=stroke, run_time=run_time, play=play)
        e = scribble_ellipse(target, color=color, stroke=stroke, seed=seed, **kw)
        if play:
            self.play(Create(e), run_time=run_time, rate_func=rate_functions.ease_in_out_sine)
        return e

    def frame_box(self, target: Mobject, color: str = GOLD, stroke: float = 4, pad: float = 0.06,
                  run_time: float = 0.7, play: bool = True) -> Mobject:
        """A clean rounded box (for a whole figure / equation)."""
        b = SurroundingRectangle(target, color=color, stroke_width=stroke, buff=pad * self.scale_k(),
                                 corner_radius=0.08 * self.scale_k())
        if play:
            self.play(Create(b), run_time=run_time)
        return b

    def dim(self, keep: Mobject | list[Mobject], opacity: float = 0.72, color: str = BG, pad: float = 0.08,
            run_time: float = 0.7, play: bool = True) -> Mobject:
        """Darken the whole paper except the `keep` region(s). FadeOut the returned mobject to undo."""
        keeps = keep if isinstance(keep, (list, tuple)) else [keep]
        f = self.camera.frame
        big = Rectangle(width=f.width * 40, height=f.height * 40).move_to(f.get_center())
        pad = pad * self.scale_k()
        holes = [RoundedRectangle(width=k.width + 2 * pad, height=k.height + 2 * pad, corner_radius=pad * 0.8).move_to(k)
                 for k in keeps]
        d = Cutout(big, *holes, fill_color=color, fill_opacity=opacity, stroke_width=0)
        if play:
            self.play(FadeIn(d), run_time=run_time)
            self.qa_event("figure", "dim", 0.3)
        return d

    def note(self, target, text: str, direction=RIGHT, color: str = GOLD, size: float = 30, dist: float = 1.2,
             width: int = 26, run_time: float = 0.8, play: bool = True) -> VGroup:
        """A callout (label + arrow) sized for the current zoom, pinned to the paper."""
        f = self.camera.frame
        k, c = self.scale_k(), f.get_center()
        p = target.get_center() if isinstance(target, Mobject) else np.array(target, dtype=float)
        if isinstance(target, Mobject):
            d = np.array(direction, dtype=float)
            d = d / (np.linalg.norm(d) or 1)
            p = p + np.array([d[0] * target.width / 2, d[1] * target.height / 2, 0])
        co = callout((p - c) / k, text, direction=direction, color=color, size=size, dist=dist, width=width)
        co.scale(k, about_point=ORIGIN).shift(c)
        if play:
            self.play(GrowArrow(co[1]), FadeIn(co[0], shift=-np.array(direction) * 0.15 * k), run_time=run_time)
        return co

    # one narrated line on the paper ----------------------------------------------
    def here(self, phrase: str, nth: int = 1) -> VGroup:
        """`p.find` on the page under the camera first, then the pages around it, then the whole paper."""
        p = self.paper
        cur = p.page_at(self.camera.frame.get_center()) or 1
        order = [cur] + [q for d in (1, -1, 2, -2) for q in [cur + d] if q - 1 in p.origin]
        for pg in order:
            try:
                return p.find(phrase, page=pg, nth=nth)
            except KeyError:
                pass
        return p.find(phrase, nth=nth)

    def beat(self, text: str, *marks, go: Mobject | None = None, fill: float = 0.85, fit: str = "contain",
             offset=(0, 0), go_time: float | None = None, pad: float = 0.3) -> list:
        """Speak one line; optionally travel to `go` first (while the line starts); fire marks on cue words.

        Each mark is one of
            "phrase"                        highlight that phrase of the paper while it is spoken (quote-reading)
            ("cue", verb)                   verb applied to the paper text equal to the cue
            ("cue", verb, target)           target: a region/mobject, or a string found with here()
            ("cue", "note", target, "label text")
            ("cue", Animation or callable)  anything else; a callable gets (scene, narration) and plays itself
        verbs: hl (highlight, lasts as long as the cue is spoken), ul (underline), circle, box, dim (keep=target),
               go (travel to target), flash (Indicate-style pulse of a box). Returns the created mobjects.
        """
        made = []
        with self.voice(text, pad=pad) as n:
            if go is not None:
                self.travel(go, fill=fill, fit=fit, offset=offset, run_time=go_time)
            for m in marks:
                if isinstance(m, str):
                    m = (m, "hl")
                cue, verb, *rest = m
                n.wait_until(cue)
                if not isinstance(verb, str):
                    if callable(verb) and not isinstance(verb, Animation):
                        verb(self, n)
                    else:
                        self.play(verb)
                    continue
                tgt = self._resolve(rest[0] if rest else cue)
                spoken_for = max(0.4, n.at(cue, end=True) - n.now())
                if verb == "hl":
                    made.append(self.highlight(tgt, run_time=spoken_for))
                elif verb == "ul":
                    made.append(self.underline(tgt, run_time=min(1.0, spoken_for)))
                elif verb == "circle":
                    made.append(self.circle(tgt, run_time=0.6, seed=len(made)))
                elif verb == "box":
                    made.append(self.frame_box(tgt, run_time=0.6))
                elif verb == "dim":
                    made.append(self.dim(keep=tgt, run_time=0.6))
                elif verb == "note":
                    made.append(self.note(tgt, rest[1], run_time=0.7))
                elif verb == "go":
                    self.travel(tgt, fill=fill, fit=fit, offset=offset)
                elif verb == "flash":
                    b = self.frame_box(tgt, play=False)
                    self.play(ShowPassingFlash(b.copy().set_stroke(width=6), time_width=0.6), run_time=0.8)
                else:
                    raise ValueError(f"unknown mark verb {verb!r}")
            if marks:
                self.qa_event("marks", text, max(0.0, n.duration - n.now() - 0.2))
        return made

    def _keep_on_clear(self) -> list:
        return []

    def _resolve(self, tgt):
        """A beat's mark target: strings are searched with here()."""
        return self.here(tgt) if isinstance(tgt, str) else tgt

    def clear_marks(self, run_time: float = 0.4) -> None:
        """Fade every overlay on the paper (keeps the page counter and stuck HUD items)."""
        olds = [m for m in self.mobjects if m is not self.camera.frame and m is not self.paper.view
                and m not in self._stuck and m not in self._top and m not in self._keep_on_clear()]
        if olds:
            self.play(*[FadeOut(m) for m in olds], run_time=run_time)
            self.remove(*olds)

    # screen-space helpers -----------------------------------------------------
    def hud(self, mob: Mobject) -> Mobject:
        """Map a mobject built in default frame coordinates into the current camera view (in place)."""
        f = self.camera.frame
        return mob.scale(self.scale_k(), about_point=ORIGIN).shift(f.get_center())

    def stick(self, mob: Mobject) -> Mobject:
        """Keep a default-coordinates mobject fixed on screen while the camera moves (don't animate it then)."""
        f = self.camera.frame
        state = {"c": np.array(f.get_center()), "k": self.scale_k()}
        self.hud(mob)

        def upd(m):
            c1, k1 = np.array(self.camera.frame.get_center()), self.scale_k()
            r = k1 / state["k"]
            if abs(r - 1) > 1e-9 or np.linalg.norm(c1 - state["c"]) > 1e-9:
                m.shift(-state["c"]).scale(r, about_point=ORIGIN).shift(c1)
                state["c"], state["k"] = c1, k1
        mob.add_updater(upd)
        self._stuck.append(mob)
        return mob

    def source(self, text: str, run_time: float = 0.4) -> Mobject:
        tag = T(text, size=18, color=DIM, font=E.SANS).to_corner(DR, buff=0.22)
        if self._frozen is None and self.paper is not None:
            self.stick(tag)
        self.play(FadeIn(tag), run_time=run_time)
        return tag

    def page_counter(self, size: float = 20, label: str = "p. {page} / {total}", corner=DL) -> Mobject:
        """A small 'p. 3 / 19' tag fixed on screen that follows whichever page is at the frame centre."""
        p = self.paper
        total = len(p.doc)
        cache: dict[int, Mobject] = {}
        state = {"page": None}

        def make(pg):
            if pg not in cache:
                txt = T(label.format(page=pg, total=total), size=size, color=INK, font=E.SANS)
                plate = BackgroundRectangle(txt, color=BG, fill_opacity=0.8, buff=0.1, corner_radius=0.08)
                cache[pg] = VGroup(plate, txt).to_corner(corner, buff=0.22)
            return cache[pg]

        tag = make(1).copy()

        def upd(m):
            f = self.camera.frame
            c, k = f.get_center(), f.width / FW
            if self._frozen is not None:
                c0, w0 = self._frozen[0], self._frozen[1]
                pg = p.page_at(c0)
                c, k = ORIGIN, 1.0
            else:
                pg = p.page_at(c)
            ref = make(pg if pg else state["page"] or 1)
            state["page"] = pg or state["page"]
            m.become(ref.copy().scale(k, about_point=ORIGIN).shift(c))
        tag.add_updater(upd)
        upd(tag)
        self.add(tag)
        self._top.append(tag)
        return tag

    # live paper <-> default coordinates ----------------------------------------
    def freeze(self, sharp: float = 1.0) -> ImageMobject:
        """Switch to default coordinates: the paper becomes a full-frame snapshot (returned, at the back),
        every other mobject is mapped to where it appears on screen, and the camera resets to the default frame.
        Nothing visibly changes. Build normal animations afterwards; `unfreeze()` maps everything back."""
        assert self.paper is not None and self._frozen is None, "freeze() needs a live paper"
        f = self.camera.frame
        c, w = np.array(f.get_center()), f.width
        k = w / FW
        snap = self.paper.snapshot(c, w, sharp)
        self.paper.view.clear_updaters()
        self.remove(self.paper.view)
        for m in self._stuck:
            m.clear_updaters()
        for m in list(self.mobjects):
            if m is f or m is self.paper.view:
                continue
            m.shift(-c).scale(1 / k, about_point=ORIGIN)
        self._set_frame(ORIGIN, FW)
        self.add(snap)
        self.bring_to_back(snap)
        self._frozen = (c, w, snap, list(self._stuck))
        self._stuck = []
        return snap

    def unfreeze(self) -> None:
        """Back to the live paper at the camera position of the last freeze(). Every mobject still in the scene
        (including ones you made while frozen) is mapped onto the paper: they now move with it.
        The snapshot is removed; put it back where freeze() left it first if you moved it."""
        assert self._frozen is not None
        c, w, snap, stuck = self._frozen
        k = w / FW
        self.remove(snap)
        for m in list(self.mobjects):
            if m is self.camera.frame:
                continue
            m.scale(k, about_point=ORIGIN).shift(c)
        self._set_frame(c, w)
        self.add(self.paper.view)
        self.bring_to_back(self.paper.view)
        self.paper.go_live(self)
        self._frozen = None
        for m in stuck:
            if m in self.mobjects:
                m.shift(-c).scale(1 / k, about_point=ORIGIN)
                self.stick(m)

    def dive(self, target: Mobject, fill: float = 0.96, run_time: float | None = None, sharp: float = 1.0,
             clear: bool = True, **kw) -> ImageMobject:
        """Fly into `target` until it fills the frame, then freeze(). Returns the snapshot (full-frame image in
        default coordinates). clear=True fades out overlays first so the snapshot is the clean page.
        Typical next step: crossfade the snapshot into your animation; then `surface(snap)`."""
        if clear:
            olds = [m for m in self.mobjects if m is not self.camera.frame and m is not self.paper.view and m not in self._stuck and m not in self._top]
            if olds:
                self.play(*[FadeOut(m) for m in olds], run_time=0.4)
                self.remove(*olds)
        self.travel(target, fill=fill, run_time=run_time, **kw)
        return self.freeze(sharp)

    def surface(self, snap: ImageMobject | None = None, to: Mobject | None = None, fill: float = 0.85,
                run_time: float | None = 0.6, back_run_time: float | None = None, **kw) -> None:
        """Return from a dive: the snapshot animates back to full frame and full opacity over everything
        (so `snap.animate.scale(1.5).set_opacity(0)` on the way in is reversed on the way out), the
        animation fades, the live paper resumes, then (optionally) the camera flies to `to`."""
        assert self._frozen is not None
        snap = snap or self._frozen[2]
        others = [m for m in self.mobjects if m is not snap and m is not self.camera.frame and m not in self._top]
        if snap not in self.mobjects:
            self.add(snap)          # brought back from wherever you left it (e.g. scaled up and transparent)
        else:
            self.bring_to_front(snap)
        home = snap.copy().stretch_to_fit_width(FW).stretch_to_fit_height(FH).move_to(ORIGIN).set_opacity(1)
        self.play(Transform(snap, home), *[FadeOut(m) for m in others], run_time=run_time)
        self.remove(*others)
        self.unfreeze()
        if to is not None:
            self.travel(to, fill=fill, run_time=back_run_time, **kw)


__all__ = ["PaperScroll", "PaperScene", "zoom_path", "scribble_ellipse"]
