"""Scene engine for paper explainers.  `from vk.engine import *` in every scene file.

What it gives you (see GUIDELINES.md §Engine for examples):
  VScene                 base Scene (MovingCameraScene): narration, word-synced timing, captions, layout lint
    self.voice(text)     context manager -> Narration n; animations inside run while the line is spoken;
                         n.wait_until("phrase"), n.until("phrase") (a run_time), n.play_at("phrase", *anims)
    self.say(text, *anims)          one-liner for a narrated beat
    self.source(text)              small provenance tag (bottom-right), e.g. "Fig. 2 · p. 4"
    self.zoom_camera(mob) / self.reset_camera()
    self.clear_all()
  PaperImage(json_or_png)   a crop from `vk paper` (figure/table/excerpt/page) as a Group with
                         .uv(u,v) point, .region(u0,v0,u1,v1), .find("words") box, .box(...), .marks(), .spot(...)
  spotlight(region_mob)  full-frame dimmer with a hole over the region
  zoom_to(group, region_mob, fill)  animation that scales/shifts `group` so the region fills the frame
  callout(target_point, text, direction)   label + arrow pointing at something
  Typography: T (prose), M (mono), para (wrapped prose with `code`/**bold**), Tx (MathTex), F (STIX markup)
  Shapes: chip, labeled_box, box, arrow; palette constants (BG, INK, DIM, ACCENT, GOLD, GREEN_, RED_, ...)
"""
from __future__ import annotations

import html
import json
import os
import re
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from manim import *  # noqa: F403

from . import tts

# ---------------------------------------------------------------- palette & type
BG = "#000000"   # pure black (walkthrough format)
PANEL = "#161B22"
PANEL_EDGE = "#30363D"
INK = "#E6EDF3"
DIM = "#7D8590"
FAINT = "#3A424D"
ACCENT = "#58C4DD"   # blue
GOLD = "#F4D35E"
GREEN_ = "#83C167"
RED_ = "#FC6255"
PURPLE_ = "#B189F5"
ORANGE_ = "#FF9F45"
TEAL_ = "#5CD0B3"
PINK_ = "#F07EB0"
PAPER = "#FBFAF7"    # sheet colour behind paper crops
MARKER = "#FFE14D"   # highlighter

PROSE = "CMU Serif"
SANS = "CMU Sans Serif"
MONO = "JetBrains Mono NL"
FW, FH = 14.222, 8.0   # frame width/height in scene units

config.background_color = BG


def T(text: str, size: float = 30, color: str = INK, weight: str = NORMAL, font: str = PROSE, **kw) -> Text:
    return Text(text, font=font, font_size=size, color=color, weight=weight, **kw)


def M(text: str, size: float = 24, color: str = INK, **kw) -> Text:
    return Text(text, font=MONO, font_size=size, color=color, **kw)


def Tx(*parts: str, size: float = 44, color: str = INK, colors: dict[str, str] | None = None, **kw) -> MathTex:
    r"""MathTex built from pieces so each piece can be coloured/animated on its own:
        eq = Tx(r"\text{pass@}", "k", "=", r"\mathbb{E}\Big[", "1-", r"\tfrac{\binom{n-c}{k}}{\binom{n}{k}}", r"\Big]",
                colors={"k": GOLD})
        eq[1] is the "k" piece.  Pieces must each be valid where they are split (split at top level, never inside
        braces). `colors` maps a piece's exact string to a colour (all matching pieces)."""
    mob = MathTex(*parts, font_size=size, color=color, **kw)
    for i, part in enumerate(parts):
        if colors and part in colors:
            mob[i].set_color(colors[part])
    return mob


def F(markup: str, size: float = 40, color: str = INK) -> MarkupText:
    """Formula-ish text via Pango markup in STIX Two (<i>, <sub>, <sup>) when LaTeX is overkill."""
    return MarkupText(markup, font="STIX Two Text", font_size=size, color=color)


def _prose_markup(text: str, code_color: str = GOLD, bold_color: str = INK) -> str:
    out = []
    for piece in re.split(r"(`[^`]+`|\*\*[^*]+\*\*|\*[^*]+\*)", text):
        if piece.startswith("`") and piece.endswith("`"):
            out.append(f'<span font_family="{MONO}" foreground="{code_color}" size="88%">{html.escape(piece[1:-1], quote=False)}</span>')
        elif piece.startswith("**"):
            out.append(f'<span foreground="{bold_color}" weight="bold">{html.escape(piece[2:-2], quote=False)}</span>')
        elif piece.startswith("*") and len(piece) > 2:
            out.append(f"<i>{html.escape(piece[1:-1], quote=False)}</i>")
        else:
            out.append(html.escape(piece, quote=False))
    return "".join(out)


def _wrap(text: str, width: int) -> list[str]:
    tokens = re.findall(r"`[^`]+`\S*|\*\*[^*]+\*\*\S*|\S+", text)
    lines, cur, n = [], [], 0
    for tok in tokens:
        vis = len(tok.replace("`", "").replace("**", ""))
        if cur and n + 1 + vis > width:
            lines.append(" ".join(cur))
            cur, n = [], 0
        cur.append(tok)
        n += vis + (1 if n else 0)
    if cur:
        lines.append(" ".join(cur))
    return lines


def para(text: str, width: int = 40, size: float = 26, color: str = INK, max_w: float | None = None,
         line_spacing: float = 0.9, code_color: str = GOLD, font: str = PROSE) -> MarkupText:
    """Wrapped prose; supports `code`, **bold**, *italic*; `width` is characters per line."""
    text = tts.display(text)
    rows = []
    for block in text.split("\n"):
        rows.extend(_wrap(block, width) or [" "])
    mob = MarkupText("\n".join(_prose_markup(r, code_color) for r in rows), font=font, font_size=size,
                     color=color, line_spacing=line_spacing)
    if max_w and mob.width > max_w:
        mob.scale_to_fit_width(max_w)
    return mob


def box(mob: Mobject, color: str = ACCENT, buff: float = 0.2, fill: float = 0.08, radius: float = 0.12, stroke: float = 2):
    return SurroundingRectangle(mob, color=color, buff=buff, corner_radius=radius, stroke_width=stroke,
                                fill_color=color, fill_opacity=fill)


def chip(label: str, color: str = ACCENT, size: float = 24, mono: bool = False, width: float | None = None) -> VGroup:
    text = M(label, size=size * 0.9) if mono else T(label, size=size)
    rect = RoundedRectangle(width=max(width or 0, text.width + 0.4), height=text.height + 0.32, corner_radius=0.1,
                            stroke_color=color, stroke_width=2, fill_color=color, fill_opacity=0.15)
    return VGroup(rect, text.move_to(rect))


def labeled_box(label: str, sub: str = "", color: str = ACCENT, w: float = 2.6, h: float = 1.1, mono: bool = False,
                size: float = 26) -> VGroup:
    rect = RoundedRectangle(width=w, height=h, corner_radius=0.14, stroke_color=color, stroke_width=2.5,
                            fill_color=color, fill_opacity=0.12)
    parts = VGroup(M(label, size=size * 0.85) if mono else T(label, size=size, weight=BOLD))
    if sub:
        parts.add(T(sub, size=max(18, size * 0.78), color=DIM))
    parts.arrange(DOWN, buff=0.08)
    if parts.width > w - 0.2:
        parts.scale_to_fit_width(w - 0.2)
    return VGroup(rect, parts.move_to(rect))


def arrow(a, b, color: str = DIM, buff: float = 0.15, width: float = 3, **kw) -> Arrow:
    pa = a.get_center() if isinstance(a, Mobject) else np.array(a, dtype=float)
    pb = b.get_center() if isinstance(b, Mobject) else np.array(b, dtype=float)
    return Arrow(pa, pb, buff=buff, color=color, stroke_width=width, max_tip_length_to_length_ratio=0.12, **kw)


def callout(target, text: str, direction=UR, color: str = GOLD, size: float = 30, dist: float = 1.4,
            width: int = 28) -> VGroup:
    """Label placed `dist` away from `target` (point or mobject) in `direction`, with an arrow to it."""
    p = target.get_center() if isinstance(target, Mobject) else np.array(target, dtype=float)
    words = para(text, width=width, size=size, color=color)
    plate = BackgroundRectangle(words, color=BG, fill_opacity=0.88, buff=0.14, corner_radius=0.08)
    label = VGroup(plate, words)
    d = np.array(direction, dtype=float)
    d = d / (np.linalg.norm(d) or 1)
    label.move_to(p + d * dist + d * label.width / 2 * abs(d[0]) + d * label.height / 2 * abs(d[1]))
    keep_in_frame(label)
    start = label.get_critical_point(-np.sign(d)) if np.any(d) else label.get_bottom()
    arr = Arrow(start, p, buff=0.08, color=color, stroke_width=3, max_tip_length_to_length_ratio=0.15)
    return VGroup(label, arr)


def keep_in_frame(mob: Mobject, margin: float = 0.25) -> Mobject:
    x0, x1 = -FW / 2 + margin, FW / 2 - margin
    y0, y1 = -FH / 2 + margin, FH / 2 - margin
    dx = max(0, x0 - mob.get_left()[0]) - max(0, mob.get_right()[0] - x1)
    dy = max(0, y0 - mob.get_bottom()[1]) - max(0, mob.get_top()[1] - y1)
    mob.shift(RIGHT * dx + UP * dy)
    return mob


def spotlight(region: Mobject, opacity: float = 0.72, buff: float = 0.06, color: str = BG) -> Mobject:
    """Dark overlay covering the whole frame except a rounded hole over `region`. FadeIn/FadeOut it."""
    hole = RoundedRectangle(width=region.width + 2 * buff, height=region.height + 2 * buff, corner_radius=0.08)
    hole.move_to(region)
    full = Rectangle(width=FW * 3, height=FH * 3)
    return Cutout(full, hole, fill_color=color, fill_opacity=opacity, stroke_width=0)


def zoom_to(group: Mobject, region: Mobject, fill: float = 0.85, center=ORIGIN):
    """Animation: scale+shift `group` so that `region` (a mobject inside or aligned with it) fills the frame."""
    s = min(FW * fill / max(region.width, 1e-3), FH * fill / max(region.height, 1e-3))
    c = region.get_center()
    return group.animate.scale(s, about_point=c).shift(np.array(center) - c)


# ---------------------------------------------------------------- paper crops
class PaperImage(Group):
    """A crop written by `vk paper` (figure / table / excerpt / page), with coordinate helpers.

    PaperImage("assets/fig2.json", height=6)         # sheet=True puts it on a paper-coloured card
    PaperImage("assets/ex1.json", width=11, dark=True)  # recolour black-on-white text to INK-on-BG
    (u, v) are normalized image coords: u 0..1 left→right, v 0..1 top→bottom (read them off `vk paper grid`).
    """

    def __init__(self, path: str | Path, height: float | None = None, width: float | None = None, sheet: bool = True,
                 dark: bool = False, pad: float = 0.12, radius: float = 0.08, _pixels=None, _meta=None):
        super().__init__()
        path = Path(path)
        js = path.with_suffix(".json")
        self.meta = json.loads(js.read_text()) if js.exists() else {"words": [], "highlights": []}
        png = path.with_suffix(".png")
        from PIL import Image

        self.pixels = np.asarray(Image.open(png).convert("RGB")) if _pixels is None else _pixels
        if _meta is not None:
            self.meta = _meta
        self._args = dict(sheet=sheet, dark=dark, pad=pad, radius=radius)
        if dark:
            arr = np.asarray(Image.fromarray(self.pixels).convert("L"), dtype=np.float32) / 255.0
            ink = 1.0 - arr
            bg, fg = np.array(color_to_rgb(BG)), np.array(color_to_rgb(INK))
            rgb = (bg[None, None, :] * (1 - ink[..., None]) + fg[None, None, :] * ink[..., None]) * 255
            self.image = ImageMobject(rgb.astype(np.uint8))
            sheet = False
        else:
            self.image = ImageMobject(self.pixels)
        self.image.set_resampling_algorithm(RESAMPLING_ALGORITHMS["bicubic"])
        if height:
            self.image.scale_to_fit_height(height)
        if width and (not height or self.image.width > width):
            self.image.scale_to_fit_width(width)
        if not height and not width:
            self.image.scale_to_fit_height(min(6.0, FH - 1.5))
            if self.image.width > FW - 1.2:
                self.image.scale_to_fit_width(FW - 1.2)
        self.sheet = None
        if sheet:
            self.sheet = RoundedRectangle(width=self.image.width + 2 * pad, height=self.image.height + 2 * pad,
                                          corner_radius=radius, fill_color=PAPER, fill_opacity=1, stroke_width=0)
            self.sheet.move_to(self.image)
            self.add(self.sheet)
        self.add(self.image)

    def panel(self, u0: float, v0: float, u1: float, v1: float, **kw) -> "PaperImage":
        """A new PaperImage of just this region (clean edges, words remapped), placed exactly over it.
        Typical: zoom_to(fig, fig.region(...)), then FadeIn(p := fig.panel(...)), FadeOut(fig), then move/scale p."""
        h, w = self.pixels.shape[:2]
        x0, x1 = int(max(0, u0) * w), int(min(1, u1) * w)
        y0, y1 = int(max(0, v0) * h), int(min(1, v1) * h)
        sub = self.pixels[y0:y1, x0:x1]
        rw, rh = (x1 - x0) / w, (y1 - y0) / h
        remap = lambda a, b, lo, span: [round((a - lo) / span, 4), round((b - lo) / span, 4)]
        words = []
        for wu0, wv0, wu1, wv1, txt in self.meta.get("words", []):
            if wu0 >= u0 and wu1 <= u1 and wv0 >= v0 and wv1 <= v1:
                a, b = remap(wu0, wu1, x0 / w, rw)
                c, d = remap(wv0, wv1, y0 / h, rh)
                words.append([a, c, b, d, txt])
        meta = {**self.meta, "words": words, "highlights": []}
        target = self.region(u0, v0, u1, v1)
        args = {**self._args, **kw}
        p = PaperImage("<panel>", height=target.height, _pixels=sub, _meta=meta, **args)
        p.move_to(target)
        return p

    # geometry ------------------------------------------------------------
    def uv(self, u: float, v: float) -> np.ndarray:
        ul = self.image.get_corner(UL)
        return ul + RIGHT * u * self.image.width + DOWN * v * self.image.height

    def region(self, u0: float, v0: float, u1: float, v1: float) -> Rectangle:
        """Invisible rectangle over a normalized region; use as a target for zoom_to / spotlight / box."""
        r = Rectangle(width=max(1e-3, (u1 - u0) * self.image.width), height=max(1e-3, (v1 - v0) * self.image.height),
                      stroke_width=0, fill_opacity=0)
        return r.move_to((self.uv(u0, v0) + self.uv(u1, v1)) / 2)

    def find(self, phrase: str, nth: int = 1, pad: float = 0.004) -> Rectangle:
        """Region of a word sequence printed in the crop (vector PDFs only), e.g. fig.find("GRPO-step450")."""
        target = [w.lower() for w in phrase.split()]
        words = self.meta.get("words", [])
        toks = [w[4].lower().strip(".,;:()") for w in words]
        hits = 0
        for i in range(len(toks)):
            if all(t.strip(".,;:()") == toks[i + j] if i + j < len(toks) else False for j, t in enumerate(target)) or \
                    (len(target) == 1 and target[0] in toks[i]):
                hits += 1
                if hits == nth:
                    ws = words[i:i + len(target)]
                    return self.region(min(w[0] for w in ws) - pad, min(w[1] for w in ws) - pad,
                                       max(w[2] for w in ws) + pad, max(w[3] for w in ws) + pad)
        raise KeyError(f"{phrase!r} not among the crop's words (try `vk paper words <json> --grep ...`)")

    def box(self, u0, v0, u1, v1, color: str = GOLD, stroke: float = 4, buff: float = 0.04) -> SurroundingRectangle:
        return SurroundingRectangle(self.region(u0, v0, u1, v1), color=color, buff=buff, stroke_width=stroke,
                                    corner_radius=0.05)

    def marks(self, color: str = MARKER, opacity: float = 0.45, indices: list[int] | None = None) -> VGroup:
        """Highlighter strokes over the excerpt's stored highlight lines (from `vk paper excerpt`)."""
        hs = self.meta.get("highlights", [])
        group = VGroup()
        for i, (u0, v0, u1, v1) in enumerate(hs):
            if indices is None or i in indices:
                r = self.region(u0, v0, u1, v1)
                group.add(Rectangle(width=r.width, height=r.height, stroke_width=0, fill_color=color,
                                    fill_opacity=opacity).move_to(r))
        return group

    def mark_anim(self, marks: VGroup, run_time: float = 1.2) -> Animation:
        return LaggedStart(*[GrowFromEdge(m, LEFT) for m in marks], lag_ratio=0.85, run_time=run_time)


# ---------------------------------------------------------------- narration
class Narration:
    """Handle yielded by VScene.voice(); times are seconds since the line started."""

    def __init__(self, scene: "VScene", clip: tts.Clip):
        self.scene, self.clip, self.start = scene, clip, scene.renderer.time
        self.duration = clip.duration

    def now(self) -> float:
        return self.scene.renderer.time - self.start

    def at(self, phrase: str, nth: int = 1, end: bool = False) -> float:
        return self.clip.at(phrase, nth, end)

    def until(self, phrase: str, nth: int = 1, end: bool = False, minimum: float = 0.3) -> float:
        """Use as run_time so an animation ends exactly when `phrase` is spoken."""
        return max(minimum, self.at(phrase, nth, end) - self.now())

    def wait_until(self, phrase: str, nth: int = 1, end: bool = False) -> None:
        d = self.at(phrase, nth, end) - self.now()
        if d > 1 / 60:
            self.scene.wait(d)

    def play_at(self, phrase: str, *anims, nth: int = 1, until: str | None = None, **kw) -> None:
        """Start `anims` when `phrase` begins; with until="later words" they end exactly when those are spoken
        (the run time is computed after the wait, so no dead air)."""
        self.wait_until(phrase, nth)
        if until is not None:
            kw["run_time"] = self.until(until)
        self.scene.play(*anims, **kw)

    def remaining(self) -> float:
        return max(0.0, self.duration - self.now())


TEXTLIKE = (Text, MarkupText, MathTex, Tex, SingleStringMathTex, Paragraph, DecimalNumber, Integer)


class VScene(MovingCameraScene):
    """Base class for every scene. Narration drives timing: an animation inside `with self.voice(...)`
    runs while that line is spoken, and the block holds until the line (plus `pad`) has finished."""

    lint = True

    def setup(self) -> None:
        super().setup()
        self.camera.background_color = BG
        self._captions: list[dict] = []
        self._lint_seen: set = set()
        self._lint_log: list[str] = []
        self._frame0 = (self.camera.frame.width, tuple(self.camera.frame.get_center()))

    # narration ------------------------------------------------------------
    @contextmanager
    def voice(self, text: str, pad: float = 0.3):
        clip = tts.synth(text)
        n = Narration(self, clip)
        if clip.path is not None:
            self.add_sound(str(clip.path))
        self._captions.append({"start": n.start, "end": n.start + clip.duration, "text": tts.display(text).replace("**", "")})
        yield n
        rest = clip.duration + pad - n.now()
        if rest > 1 / 60:
            self.wait(rest)
        elif rest < -2.0:
            self._warn(f"animations overran the narration by {-rest:.1f}s (dead air): {text[:60]!r}")

    def say(self, text: str, *anims, run_time: float | None = None, pad: float = 0.3, **kw) -> None:
        with self.voice(text, pad=pad) as n:
            if anims:
                self.play(*anims, run_time=run_time or min(1.2, max(0.4, n.duration * 0.5)), **kw)

    # helpers ----------------------------------------------------------------
    def source(self, text: str, run_time: float = 0.4) -> Mobject:
        """Provenance tag in the bottom-right corner; returns it so you can FadeOut later."""
        tag = T(text, size=18, color=DIM, font=SANS).to_corner(DR, buff=0.22)
        self.play(FadeIn(tag), run_time=run_time)
        return tag

    def clear_all(self, run_time: float = 0.5, keep: list | None = None) -> None:
        keep = keep or []
        mobs = [m for m in self.mobjects if m not in keep and m is not self.camera.frame]
        if mobs:
            self.play(*[FadeOut(m) for m in mobs], run_time=run_time)
        for m in mobs:
            self.remove(m)

    def zoom_camera(self, target: Mobject, fill: float = 0.85, run_time: float = 1.2, **kw) -> None:
        f = self.camera.frame
        w = max(target.width / fill, target.height / fill * FW / FH)
        self.play(f.animate.move_to(target).set(width=w), run_time=run_time, **kw)

    def reset_camera(self, run_time: float = 1.0) -> None:
        self.play(self.camera.frame.animate.move_to(ORIGIN).set(width=FW), run_time=run_time)

    # lint -------------------------------------------------------------------
    def _warn(self, msg: str) -> None:
        key = msg[:90]
        if key not in self._lint_seen:
            self._lint_seen.add(key)
            self._lint_log.append(f"t={self.renderer.time:7.2f}s  {msg}")

    def play(self, *args, **kwargs):
        super().play(*args, **kwargs)
        if self.lint:
            try:
                self._lint()
            except Exception as e:  # never break a render because of the linter
                self._warn(f"lint error {e!r}")

    def _visible(self, m: Mobject) -> bool:
        try:
            if isinstance(m, ImageMobject):
                return True
            return max(m.get_fill_opacity(), m.get_stroke_opacity()) > 0.05 and m.width > 1e-3
        except Exception:
            return True

    def _lint(self) -> None:
        f = self.camera.frame
        if abs(f.width - FW) > 0.01 or np.linalg.norm(f.get_center()) > 0.01:
            return  # camera moved: frame checks would be misleading
        texts: list[Mobject] = []

        def walk(m: Mobject):
            if isinstance(m, TEXTLIKE):
                if self._visible(m):
                    texts.append(m)
                return
            for s in m.submobjects:
                walk(s)

        for m in self.mobjects:
            if isinstance(m, Cutout) or not self._visible(m) and not m.submobjects:
                continue
            walk(m)
        for t in texts:
            l, r, b, top = t.get_left()[0], t.get_right()[0], t.get_bottom()[1], t.get_top()[1]
            if l < -FW / 2 + 0.05 or r > FW / 2 - 0.05 or b < -FH / 2 + 0.05 or top > FH / 2 - 0.05:
                self._warn(f"text off-frame: {_label(t)!r} x[{l:.2f},{r:.2f}] y[{b:.2f},{top:.2f}]")
        boxes = [(t, t.get_left()[0], t.get_right()[0], t.get_bottom()[1], t.get_top()[1]) for t in texts]
        for i in range(len(boxes)):
            a, al, ar, ab, at_ = boxes[i]
            for j in range(i + 1, len(boxes)):
                b, bl, br, bb, bt = boxes[j]
                w, h = min(ar, br) - max(al, bl), min(at_, bt) - max(ab, bb)
                if w > 0 and h > 0:
                    small = min((ar - al) * (at_ - ab), (br - bl) * (bt - bb))
                    if small > 0 and w * h > 0.2 * small:
                        self._warn(f"text overlap: {_label(a)!r} <> {_label(b)!r}")

    def tear_down(self) -> None:
        super().tear_down()
        out = Path(config.media_dir) / "vk"
        out.mkdir(parents=True, exist_ok=True)
        name = type(self).__name__
        (out / f"{name}.captions.json").write_text(json.dumps(self._captions, indent=1))
        (out / f"{name}.lint.txt").write_text("\n".join(self._lint_log) + ("\n" if self._lint_log else ""))


def _label(m: Mobject) -> str:
    for attr in ("text", "tex_string", "original_text"):
        v = getattr(m, attr, None)
        if isinstance(v, str) and v.strip():
            return v.strip()[:40]
    return type(m).__name__


# ---------------------------------------------------------------- plotting
def axes(x_range, y_range, x_len: float = 8, y_len: float = 4.5, x_label: str = "", y_label: str = "",
         x_ticks=None, y_ticks=None, color: str = DIM, font_size: float = 28, **kw) -> Axes:
    """Styled Axes. For log2 axes (pass@k) plot log2(k) and pass x_ticks={log2k: "k"} labels."""
    ax = Axes(x_range=x_range, y_range=y_range, x_length=x_len, y_length=y_len, tips=False,
              axis_config={"color": color, "stroke_width": 2, "include_ticks": True, "tick_size": 0.06,
                           "font_size": font_size}, **kw)
    if x_ticks is not None:
        ax.x_axis.add_labels({k: T(str(v), size=font_size, color=DIM) for k, v in dict(x_ticks).items()})
    if y_ticks is not None:
        ax.y_axis.add_labels({k: T(str(v), size=font_size, color=DIM) for k, v in dict(y_ticks).items()})
    if x_label:
        ax.add(T(x_label, size=font_size + 2, color=DIM).next_to(ax.x_axis, DOWN, buff=0.45))
    if y_label:
        ax.add(T(y_label, size=font_size + 2, color=DIM).rotate(PI / 2).next_to(ax.y_axis, LEFT, buff=0.45))
    return ax


def series(ax: Axes, xs, ys, color: str = ACCENT, width: float = 4, dots: bool = True, dot_r: float = 0.045) -> VGroup:
    """Polyline (+ dots) through data points; animate with Create(series[0]) then FadeIn(series[1])."""
    pts = [ax.c2p(x, y) for x, y in zip(xs, ys)]
    line = VMobject(stroke_color=color, stroke_width=width).set_points_as_corners(pts)
    group = VGroup(line)
    group.add(VGroup(*[Dot(p, radius=dot_r, color=color) for p in pts]) if dots else VGroup())
    return group


def bars(ax: Axes, xs, ys, color: str = ACCENT, width: float = 0.6, opacity: float = 0.85) -> VGroup:
    out = VGroup()
    for x, y in zip(xs, ys):
        p0, p1 = ax.c2p(x - width / 2, 0), ax.c2p(x + width / 2, y)
        out.add(Rectangle(width=abs(p1[0] - p0[0]), height=max(1e-3, abs(p1[1] - p0[1])), stroke_width=0,
                          fill_color=color, fill_opacity=opacity).move_to((p0 + p1) / 2))
    return out


__all__ = [n for n in dir() if not n.startswith("_")]
