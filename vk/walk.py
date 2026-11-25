"""Illustration side of the walkthrough, plus the `Video` program class.  `from vk.walk import *`.

    class Paper(Video):                      # the whole video is one class in video.py
        title = "Reasoning with Sampling"

        @chapter("Title and abstract")
        def open(self, p):                   # p = the paper (PaperScroll); self = the scene
            self.beat("...", "phrase to highlight", go=p.block("Frontier reasoning"))
            ...

Stages (the paper freezes, blurs and darkens behind; nothing is boxed):
    area = self.stage("over")                         # draw over the blurred page
    area = self.stage("aside", target=p.figure(3))    # the figure is pulled out to the left; `area` = right side
    area = self.stage("dive", target=p.figure(2, caption=False))  # the figure grows to fill the frame;
    self.reveal(*mobs)                                #   then dissolves into your animation
    area = self.stage("full")                         # paper hidden entirely (impact minute, etc.)
    fit(mob, area); ...; self.unstage()               # everything fades, pulled figure flies home, page sharpens
Equations: eq = self.lift_equation(region, *tex_pieces); self.explain(eq, line, (cue, idx, label, colour), ...)
Bubbles:   self.bubble(anchor, title, body, kind="recall|concept|term|note|source", math=...)
Reading:   self.read("exact sentence from the paper")  # spoken, highlighted word by word in sync
Components for illustrations: vk/illus.py (re-exported here).
"""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path

import numpy as np
from manim import *  # noqa: F403

from .engine import *  # noqa: F403
from .engine import BG, DIM, FH, FW, GOLD, INK, MARKER, SANS, T, Tx, para, keep_in_frame
from .scroll import *  # noqa: F403
from .scroll import PaperScene, PaperScroll, zoom_path
from .illus import *  # noqa: F403

KIND = {  # bubble kind -> (label, colour)
    "recall": ("RECALL", ACCENT),
    "concept": ("CONCEPT", PURPLE_),
    "term": ("TERM", TEAL_),
    "note": ("NOTE", GOLD),
    "source": ("SOURCE", DIM),
}
MARGIN = 0.45


def fit(mob: Mobject, area: Mobject, buff: float = 0.25, up: bool = False) -> Mobject:
    """Scale `mob` down (or up if up=True) to fit inside `area` minus `buff`, and centre it there."""
    w, h = max(area.width - 2 * buff, 0.1), max(area.height - 2 * buff, 0.1)
    s = min(w / max(mob.width, 1e-3), h / max(mob.height, 1e-3))
    if s < 1 or up:
        mob.scale(s)
    return mob.move_to(area)


def _area(x0: float, x1: float, y0: float = -FH / 2 + MARGIN, y1: float = FH / 2 - MARGIN) -> Rectangle:
    r = Rectangle(width=x1 - x0, height=y1 - y0, stroke_width=0, fill_opacity=0)
    return r.move_to([(x0 + x1) / 2, (y0 + y1) / 2, 0])


def _blurred(img: ImageMobject, radius: float, dark: float) -> ImageMobject:
    """Blurred, darkened copy of a full-frame snapshot (same placement)."""
    from PIL import Image, ImageFilter

    arr = img.get_pixel_array()[..., :3]
    pil = Image.fromarray(arr.astype(np.uint8))
    r = radius * pil.width / 1920
    if r > 0.3:
        pil = pil.filter(ImageFilter.GaussianBlur(r))
    out = np.asarray(pil, dtype=np.float32) * (1 - dark) + np.array(color_to_rgb(BG)) * 255 * dark
    m = ImageMobject(out.astype(np.uint8))
    m.set_resampling_algorithm(RESAMPLING_ALGORITHMS["bilinear"])
    m.stretch_to_fit_width(img.width).stretch_to_fit_height(img.height).move_to(img)
    return m


# ---------------------------------------------------------------- chapters
def chapter(title: str):
    """Decorator marking a method of a `Video` subclass as one chapter (rendered as its own clip, in order)."""
    def deco(fn):
        fn._chapter = title
        return fn
    return deco


CURRENT = {"chapter": None}   # build hooks read this (which chapter is speaking)


class WalkScene(PaperScene):
    """PaperScene + stages, formula breakdowns, bubbles, read-along."""

    blur = 14          # px at 1080p
    dark = 0.78        # how much the frozen page is darkened behind an illustration

    def setup(self) -> None:
        super().setup()
        self._stage = None
        self._bubble, self._bubble_t, self._bubble_live = None, -1e9, False

    # ---------------------------------------------------------------- stages
    def _screen_crop(self, region: Mobject, dpi: int = 300) -> ImageMobject:
        """High-res crop of a paper region, placed where it currently appears on screen (default coords)."""
        crop = self.paper.crop(region, dpi=dpi)
        f = self.camera.frame
        k, c = f.width / FW, np.array(f.get_center())
        crop.scale(1 / k).move_to((region.get_center() - c) / k)
        return crop

    def stage(self, mode: str = "over", target: Mobject | None = None, frac: float = 0.45,
              run_time: float = 0.9, dark: float | None = None) -> Rectangle:
        """Open an illustration stage over the frozen, blurred, darkened page and return the free area.
        over: draw over the page (with `target`: that region stays sharp and lifted, in place).
        aside: `target` is pulled out to the left `frac` of the frame; area = the rest. A wide, short target (a
               table, a row of panels) goes to the top `frac` of the frame instead; area = below it.
        dive: `target` grows to fill the frame; call self.reveal(...) to dissolve it into your animation.
        full: the page darkens to black."""
        st = self._stage
        if st is not None and st["mode"] == "look" and target is not None and mode in ("dive", "aside") \
                and hasattr(target, "pdf_rect") and target.page == st["target"].page \
                and abs(target.pdf_rect & st["target"].pdf_rect) >= 0.6 * abs(target.pdf_rect):
            return self._restage(mode, target, frac, run_time)      # continue from the close-up, no blur round-trip
        if st is not None and st["mode"] == "look":
            self.unstage()
        assert self._stage is None, "call unstage() before opening another stage"
        assert self.paper is not None and self._frozen is None
        self._dismiss_bubble()
        dpi = 300
        if target is not None:        # enough pixels for the crop's final on-screen size
            px = config.pixel_width * (FW - 0.6) / FW
            dpi = int(min(700, max(200, 1.15 * px / (target.pdf_rect.width / 72))))
        crop = self._screen_crop(target, dpi) if target is not None else None
        dest = None
        home = crop.copy() if crop is not None else None
        self.clear_marks()
        snap = self.freeze()
        d = 1.0 if mode == "full" else (self.dark if dark is None else dark)
        bg = _blurred(snap, self.blur, d)
        anims = [FadeIn(bg)]
        area = _area(-FW / 2 + MARGIN, FW / 2 - MARGIN)
        if crop is not None:
            self.add(bg, crop)
            dest, area = self._stage_dest(mode, crop, frac)
            if dest is not None:
                anims.append(Transform(crop, dest))
        self.play(*anims, run_time=run_time)
        self._stage = {"snap": snap, "bg": bg, "crop": crop, "home": home, "mode": mode, "target": target,
                       "dest": dest if crop is not None and mode != "over" else home}
        return area

    def _stage_dest(self, mode, crop, frac):
        """Where a stage puts its pulled crop, and the free area left for the illustration."""
        area = _area(-FW / 2 + MARGIN, FW / 2 - MARGIN)
        dest = None
        if mode == "aside" and crop.width / crop.height > 2.2:
            # a wide, short target (a table, a row of panels): pull it to the top, illustrate below
            top = _area(-FW / 2 + MARGIN, FW / 2 - MARGIN, FH / 2 - FH * frac, FH / 2 - 0.3)
            dest = fit(crop.copy(), top, buff=0.05, up=True)
            area = _area(-FW / 2 + MARGIN, FW / 2 - MARGIN, -FH / 2 + 0.35, dest.get_bottom()[1] - 0.35)
        elif mode == "aside":
            left = _area(-FW / 2 + MARGIN, -FW / 2 + FW * frac)
            dest = fit(crop.copy(), left, buff=0.1, up=True)
            area = _area(-FW / 2 + FW * frac + 0.4, FW / 2 - MARGIN)
        elif mode in ("dive", "look"):
            m = 0.3 if mode == "dive" else 0.55
            dest = fit(crop.copy(), _area(-FW / 2 + m, FW / 2 - m, -FH / 2 + m, FH / 2 - m), buff=0, up=True)
        return dest, area

    def _restage(self, mode, target, frac, run_time):
        """A look() close-up is open on (or around) `target`: turn it into a dive/aside directly."""
        st = self._stage
        c, w = self._frozen[0], self._frozen[1]
        k = w / FW
        keep = {st["snap"], st["bg"], st["crop"], self.camera.frame, *self._top}
        marks = [m for m in self.mobjects if m not in keep]
        px = config.pixel_width * (FW - 0.6) / FW
        dpi = int(min(700, max(200, 1.15 * px / (target.pdf_rect.width / 72))))
        world = self.paper.crop(target, dpi=dpi)                 # placed over the region, in paper coordinates
        start = self._lifted(world)                              # exactly where it shows inside the close-up
        home = world.copy().shift(-c).scale(1 / k, about_point=ORIGIN)
        dest, area = self._stage_dest(mode, start, frac)
        old = st["crop"]
        self.add(start)
        anims = [Transform(start, dest), FadeOut(old)] + [FadeOut(m) for m in marks]
        self.play(*anims, run_time=run_time)
        self.remove(old, *marks)
        st.update({"crop": start, "home": home, "mode": mode, "target": target, "dest": dest})
        return area

    def look(self, region: Mobject, run_time: float = 0.8) -> None:
        """Bring a figure, table, panel or passage up close: the page blurs and darkens, the region grows to fill
        the screen (the camera itself never zooms past page width). Following beats mark *on the enlarged
        region* (phrases are searched on its page; regions map onto it). Closes on the next travel/go/stage,
        or with self.unstage()."""
        r = region.pdf_rect
        pr = self.paper.doc[region.page - 1].rect
        m = 4.0                          # a little paper around the region, so nothing at its edge is clipped
        # (sideways only for figures/tables found by p.figure, whose top/bottom already stop at the caption)
        mv = 1.0 if getattr(region, "is_figure", False) else m
        region = self.paper.rect(region.page, max(0, r.x0 - m), max(0, r.y0 - mv), min(pr.width, r.x1 + m),
                                 min(pr.height, r.y1 + mv))
        self.stage("look", target=region, run_time=run_time)
        self.qa_event("figure", "look", 0.2)

    def _lifted(self, m: Mobject) -> Mobject:
        """Map a paper-space mobject onto the lifted crop of the open stage."""
        st = self._stage
        c, w = self._frozen[0], self._frozen[1]
        k = w / FW
        home, dest = st["home"], st["dest"]
        s = dest.width / home.width
        out = m.copy()
        out.shift(-c).scale(1 / k, about_point=ORIGIN)
        out.scale(s, about_point=home.get_center()).shift(dest.get_center() - home.get_center())
        return out

    def _resolve(self, tgt):
        st = self._stage
        if st is None or st["mode"] not in ("look", "aside") or st["target"] is None or st["crop"] is None:
            return super()._resolve(tgt)
        reg = st["target"]
        if isinstance(tgt, str):
            try:
                found = self.paper.find(tgt, page=reg.page)
            except KeyError:
                found = None
        else:
            found = tgt

        def inside(m):
            rs = getattr(m, "pdf_rects", None) or ([m.pdf_rect] if hasattr(m, "pdf_rect") else None)
            if rs is None or getattr(m, "page", reg.page) != reg.page:
                return rs is None          # a plain mobject: assume it is meant for the lifted view
            R = reg.pdf_rect
            return all(abs(r & R) >= 0.6 * abs(r) for r in rs)

        if found is not None and inside(found):
            return self._lifted(found)
        if st["mode"] == "look":               # the mark is elsewhere on the paper: put the page back first
            self.unstage()
            return super()._resolve(tgt)
        return self._lifted(found if found is not None else self.paper.find(tgt))

    def clear_marks(self, run_time: float = 0.4) -> None:
        b = getattr(self, "_bubble", None)
        if b is not None:
            self._hold_bubble_for(b)
            self._bubble = None
        super().clear_marks(run_time)

    def _keep_on_clear(self) -> list:
        st = self._stage
        return [st["snap"], st["bg"], st["crop"]] if st else []

    def travel(self, target, *anims, **kw):
        if getattr(self, "_bubble", None) is not None and getattr(self, "_bubble_live", False):
            self._dismiss_bubble()                 # a margin note is pinned to the page: never let it slide off
        if self._stage is not None and self._stage["mode"] == "look":
            self.unstage()
        return super().travel(target, *anims, **kw)

    def dim(self, keep, *a, **kw):
        """On the live page: if the kept region would be small on screen (under ~60% of the width and 75% of the
        height), look() at it instead, so it is readable. Otherwise darken everything else."""
        keeps = keep if isinstance(keep, (list, tuple)) else [keep]
        if (self._stage is None and len(keeps) == 1 and hasattr(keeps[0], "pdf_rect") and kw.get("play", True)):
            k = self.camera.frame.width / FW
            if keeps[0].width / k < 0.62 * FW and keeps[0].height / k < 0.75 * FH:
                self.look(keeps[0])
                return VGroup()
        return super().dim(keep, *a, **kw)

    @property
    def pulled(self) -> ImageMobject | None:
        """The figure crop pulled out by the current stage (None if none)."""
        return self._stage["crop"] if self._stage else None

    def reveal(self, *mobs, run_time: float = 1.0) -> None:
        """Dissolve the pulled figure into `mobs` (it swells slightly and fades as they appear)."""
        crop = self.pulled
        anims = [FadeIn(m, scale=0.92) for m in mobs]
        if crop is not None and crop in self.mobjects:
            anims.append(crop.animate.scale(1.12).set_opacity(0))
        self.play(*anims, run_time=run_time)
        if crop is not None:
            self.remove(crop)

    def unstage(self, to: Mobject | None = None, run_time: float = 0.8, **kw) -> None:
        """Close the stage: drawings fade, a pulled figure flies back to its place, the page sharpens again."""
        st = self._stage
        assert st is not None, "no open stage"
        self.qa_event("stage", st["mode"])
        self._stage = None
        keep = {st["snap"], st["bg"], st["crop"], self.camera.frame, *self._top}
        others = [m for m in self.mobjects if m not in keep]
        anims = [FadeOut(m) for m in others]
        crop, home = st["crop"], st["home"]
        if crop is not None:
            if crop not in self.mobjects:
                crop.set_opacity(0)
                self.add(crop)
            anims.append(Transform(crop, home))
        anims.append(FadeOut(st["bg"], rate_func=lambda t: smooth(max(0.0, (t - 0.35) / 0.65))))
        self.play(*anims, run_time=run_time)          # one move: drawings fade, figure flies home, page sharpens
        self.remove(*others, st["bg"], *([crop] if crop is not None else []))
        self.unfreeze()
        if to is not None:
            self.travel(to, **kw)

    # ---------------------------------------------------------------- formulas
    def lift_equation(self, region: Mobject, *parts: str, colors: dict | None = None, size: float = 54,
                      to=UP * 1.2, run_time: float = 1.0) -> MathTex:
        """The printed equation stays sharp while the page blurs behind it, dissolves into live LaTeX in place,
        and moves to `to`. Opens an 'over' stage: close with self.unstage()."""
        self.travel(region)
        self.stage("over", target=region, run_time=0.6)
        crop = self.pulled
        eq = Tx(*parts, size=size, colors=colors)
        eq.match_height(crop).scale(0.9)
        if eq.width > crop.width * 1.15:
            eq.match_width(crop)
        eq.move_to(crop)
        dest = Tx(*parts, size=size, colors=colors)
        if dest.width > FW - 1.2:
            dest.scale_to_fit_width(FW - 1.2)
        dest.move_to(to)
        self.play(FadeOut(crop), FadeIn(eq), run_time=0.45)
        self.remove(crop)
        self._stage["crop"] = None
        self.play(Transform(eq, dest), run_time=run_time)
        self.remove(eq)
        self.add(dest)
        self.qa_event("equation", " ".join(parts)[:40], 0.2)
        return dest

    def explain(self, eq: MathTex, text: str, *terms, dim: float = 0.3, keep: bool = False, pad: float = 0.3) -> VGroup:
        """Narrate a line while walking through pieces of `eq`. Each term is
        ("cue words", index or (i, j, ...), "label", colour[, UP|DOWN]).
        On its cue the piece takes the colour, the other pieces dim, a brace + label appear (the previous
        brace leaves). keep=True leaves the last brace up. Returns the braces+labels still on screen."""
        shown = VGroup()
        prev = None
        side_toggle = [DOWN, UP]
        with self.voice(text, pad=pad) as n:
            for i, term in enumerate(terms):
                cue, idx, label, color, *rest = term
                side = rest[0] if rest else side_toggle[i % 2]
                idxs = idx if isinstance(idx, (tuple, list)) else (idx,)
                piece = VGroup(*[eq[j] for j in idxs])
                n.wait_until(cue)
                br = Brace(piece, side, buff=0.12, color=color)
                lab = T(label, size=28, color=color).next_to(br, side, buff=0.12)
                keep_in_frame(lab)
                others = [eq[j] for j in range(len(eq)) if j not in idxs]
                anims = [piece.animate.set_color(color).set_opacity(1), GrowFromCenter(br), FadeIn(lab, shift=side * 0.15)]
                anims += [o.animate.set_opacity(dim) for o in others]
                if prev is not None:
                    anims.append(FadeOut(prev))
                self.play(*anims, run_time=0.6)
                self.qa_event("formula", label, 0.3)
                prev = VGroup(br, lab)
        if prev is not None and not keep:
            self.play(FadeOut(prev), *[m.animate.set_opacity(1) for m in eq], run_time=0.5)
        elif prev is not None:
            shown.add(prev)
            self.play(*[m.animate.set_opacity(1) for m in eq], run_time=0.4)
        return shown

    def morph(self, eq: MathTex, *parts: str, colors: dict | None = None, size: float = 54, run_time: float = 1.2,
              **kw) -> MathTex:
        """Rewrite `eq` into a new form; matching pieces fly to their new places (TransformMatchingTex)."""
        new = Tx(*parts, size=size, colors=colors).move_to(eq)
        if new.width > FW - 1.2:
            new.scale_to_fit_width(FW - 1.2)
        self.play(TransformMatchingTex(eq, new, **kw), run_time=run_time)
        return new

    # ---------------------------------------------------------------- bubbles
    def _bubble_mob(self, kind: str, title: str, body: str, math: str | None, width: int, color: str | None):
        label, col = KIND.get(kind, (kind.upper(), GOLD))
        col = color or col
        rows = [T(label, size=18, color=col, weight=BOLD, font=SANS)]
        if title:
            rows.append(T(title, size=28, color=INK))
        if body:
            rows.append(para(body, width=width, size=22, color=INK))
        if math:
            m = Tx(math, size=34)
            rows.append(m)
        content = VGroup(*rows).arrange(DOWN, aligned_edge=LEFT, buff=0.14)
        if content.width > 3.8:
            content.scale_to_fit_width(3.8)
        rule = Line(content.get_corner(UL) + LEFT * 0.2 + UP * 0.05, content.get_corner(DL) + LEFT * 0.2 + DOWN * 0.05,
                    stroke_color=col, stroke_width=4)
        plate = Rectangle(width=content.width + 0.75, height=content.height + 0.4, stroke_width=0,
                          fill_color=BG, fill_opacity=0.94)
        plate.move_to(VGroup(rule, content)).shift(RIGHT * 0.08)
        return VGroup(plate, rule, content), col

    def bubble(self, anchor: Mobject | None, title: str = "", body: str = "", kind: str = "recall",
               math: str | None = None, side=RIGHT, width: int = 24, color: str | None = None,
               run_time: float = 0.8, play: bool = True, say: str | None = None) -> VGroup:
        """A side-note for a prerequisite, a reminder or a piece of jargon: a coloured rule, a small-caps kind label,
        a title, a short body and/or one formula.
        Live on the paper it sits in the page margin: the camera slides (zooming out a little if needed) so the
        page leaves room beside `anchor`, the anchor is underlined in the bubble's colour and a short leader joins
        the page edge to the note. On a stage it is placed next to the anchor on a dark plate.
        Only one bubble at a time: a new one, a camera move, a stage or clear_marks dismisses it, but never before it
        has been readable for MIN_BUBBLE seconds. say="..." speaks a line while it is up (use it when nothing else is
        being said, e.g. right before a stage)."""
        self._dismiss_bubble()
        live = self._frozen is None and self.paper is not None
        mob, col = self._bubble_mob(kind, title, body, math, width, color)
        if live and anchor is not None:
            out = self._margin_bubble(anchor, mob, col, side, run_time if play else 0)
        else:
            out = self._plate_bubble(anchor, mob, col, side)
        if play:
            plate, rule, content = out[-1]
            anims = [FadeIn(plate), Create(rule), LaggedStart(*[FadeIn(r, shift=RIGHT * 0.1 * self.scale_k()) for r in content],
                                                             lag_ratio=0.15)]
            anims = [Create(m) for m in out[:-1]] + anims
            if say:
                with self.voice(say) as n:
                    self.play(*anims, run_time=run_time)
                    self._bubble_t = self.renderer.time
            else:
                self.play(*anims, run_time=run_time)
            self.qa_event("bubble", f"{kind}: {title}", 0.5)
        self._bubble = out
        self._bubble_t = getattr(self, "_bubble_t", 0.0) if say else self.renderer.time
        self._bubble_live = live and anchor is not None
        return out

    MIN_BUBBLE = 3.0      # seconds a bubble stays readable before the scene moves on

    def _dismiss_bubble(self, run_time: float = 0.3) -> None:
        b = getattr(self, "_bubble", None)
        self._bubble = None
        if b is None:
            return
        self._hold_bubble_for(b)
        fam = set(b.get_family())
        parts = [m for m in self.mobjects if m in fam]
        if parts:
            self.play(*[FadeOut(m) for m in parts], run_time=run_time)
            self.remove(*parts)

    def _hold_bubble_for(self, b) -> None:
        if any(m in self.mobjects for m in b.get_family()):
            rest = self.MIN_BUBBLE - (self.renderer.time - getattr(self, "_bubble_t", -1e9))
            if rest > 0.05:
                self.wait(rest)

    def _margin_bubble(self, anchor, mob, col, side, run_time):
        p = self.paper
        f = self.camera.frame
        page = p.page(p.page_at(anchor.get_center()) or 1)
        right = not np.allclose(side, LEFT)
        wb = mob.width
        mob[0].set_fill(opacity=0)                       # in the margin: no plate
        edge_w = page.get_right()[0] if right else page.get_left()[0]
        # keep the whole text column on screen (pages have ~15% side margins), not just the anchor
        pno = p.page_at(anchor.get_center()) or 1
        tl, tr = p.text_column(pno)
        a_far = tl - 0.02 * page.width if right else tr + 0.02 * page.width
        R = (FW / 2 - 0.35 - wb - 0.5) if right else (-FW / 2 + 0.35 + wb + 0.5)   # page edge, screen x
        k = f.width / FW
        lim = -FW / 2 + 0.35 if right else FW / 2 - 0.35
        need = abs(edge_w - a_far) / max(abs(R - lim), 1e-3)
        k = max(k, need)
        cx = edge_w - R * k
        cy = f.get_center()[1]
        top, bot = cy + FH / 2 * k, cy - FH / 2 * k
        ay = anchor.get_center()[1]
        if not (bot + 0.6 * k < ay < top - 0.6 * k):
            cy = ay
        if run_time:
            path, _ = zoom_path(f.get_center(), f.width, np.array([cx, cy, 0]), k * FW, None)

            def upd(m, a):
                c, w = path(smooth(a))
                m.set_width(w)
                m.move_to(c)
            self.play(UpdateFromAlphaFunc(f, upd), run_time=0.8)
            self._set_frame(np.array([cx, cy, 0]), k * FW)
        else:
            self._set_frame(np.array([cx, cy, 0]), k * FW)
        # place in screen coords, then map to the paper
        c = np.array(self.camera.frame.get_center())
        k = self.camera.frame.width / FW
        a_s = (anchor.get_center() - c) / k
        edge_s = (edge_w - c[0]) / k
        if right:
            mob.next_to([edge_s, a_s[1], 0], RIGHT, buff=0.5)
        else:
            mob.next_to([edge_s, a_s[1], 0], LEFT, buff=0.5)
        room = (FW / 2 - 0.3 - (edge_s + 0.5)) if right else ((edge_s - 0.5) - (-FW / 2 + 0.3))
        if mob.width > room > 1.5:                       # never run off the screen edge: shrink to the margin
            mob.scale(room / mob.width, about_edge=LEFT if right else RIGHT)
        if mob.height > FH - 0.8:
            mob.scale((FH - 0.8) / mob.height)
        keep_in_frame(mob, margin=0.3)
        rule = mob[1]
        y = np.clip(a_s[1], rule.get_bottom()[1], rule.get_top()[1])
        end_x = rule.get_x() if right else mob.get_right()[0] + 0.05
        y = np.clip(a_s[1], mob.get_bottom()[1], mob.get_top()[1])
        leader = Line([edge_s, a_s[1], 0], [end_x, y, 0], stroke_color=col, stroke_width=2)
        under = Line([(anchor.get_left()[0] - c[0]) / k, (anchor.get_bottom()[1] - c[1]) / k, 0],
                     [(anchor.get_right()[0] - c[0]) / k, (anchor.get_bottom()[1] - c[1]) / k, 0], stroke_color=col, stroke_width=3)
        out = VGroup(under, leader, mob)
        out.scale(k, about_point=ORIGIN).shift(c)
        return out

    def _plate_bubble(self, anchor, mob, col, side):
        if anchor is None:
            return VGroup(mob.to_corner(UR, buff=0.4))
        live = self._frozen is None and self.paper is not None
        f = self.camera.frame
        k, c = (f.width / FW, np.array(f.get_center())) if live else (1.0, ORIGIN)
        a = (anchor.get_center() - c) / k
        aw, ah = anchor.width / k, anchor.height / k
        box = (a[0] - aw / 2, a[0] + aw / 2, a[1] - ah / 2, a[1] + ah / 2)
        best, best_ov = None, 1e9
        for d in [side, -np.array(side), DOWN, UP]:
            trial = mob.copy()
            if np.allclose(d, RIGHT):
                trial.next_to([box[1], a[1], 0], RIGHT, buff=0.45)
            elif np.allclose(d, LEFT):
                trial.next_to([box[0], a[1], 0], LEFT, buff=0.45)
            elif np.allclose(d, DOWN):
                trial.next_to([a[0], box[2], 0], DOWN, buff=0.45)
            else:
                trial.next_to([a[0], box[3], 0], UP, buff=0.45)
            keep_in_frame(trial, margin=0.3)
            l, r, b, t = trial.get_left()[0], trial.get_right()[0], trial.get_bottom()[1], trial.get_top()[1]
            ov = max(0, min(r, box[1]) - max(l, box[0])) * max(0, min(t, box[3]) - max(b, box[2]))
            if ov < best_ov:
                best, best_ov = trial, ov
            if ov < 1e-3:
                break
        rule = best[1]
        end = rule.get_center()
        start = np.array([np.clip(end[0], box[0], box[1]), np.clip(end[1], box[2], box[3]), 0])
        out = VGroup(Line(start, end, stroke_color=col, stroke_width=2), best)
        if live:
            out.scale(k, about_point=ORIGIN).shift(c)
        return out

    # ---------------------------------------------------------------- external media
    def external(self, path: str | Path, area: Mobject | None = None, credit: str | None = None,
                 height: float | None = None, play: bool = True) -> Group:
        """An external image (downloaded with `vk fetch`) with its credit line, fitted to `area`."""
        path = Path(path)
        meta_p = path.with_suffix(".json")
        meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
        credit = credit or meta.get("credit") or meta.get("url", "")
        img = ImageMobject(str(path))
        img.set_resampling_algorithm(RESAMPLING_ALGORITHMS["bicubic"])
        img.scale_to_fit_height(height or 5.0)
        line = T(credit, size=18, color=DIM, font=SANS)
        if line.width > img.width:
            line.scale_to_fit_width(img.width)
        line.next_to(img, DOWN, buff=0.15, aligned_edge=LEFT)
        g = Group(img, line)
        fit(g, area or _area(-FW / 2 + MARGIN, FW / 2 - MARGIN), buff=0.1)
        if play:
            self.play(FadeIn(img, shift=UP * 0.15), FadeIn(line), run_time=0.7)
        return g



    # ---------------------------------------------------------------- read-along
    def read(self, phrase: str, text: str | None = None, color: str = MARKER, opacity: float = 0.42,
             go: Mobject | None = None, page: int | None = None, pad: float = 0.3) -> VGroup:
        """Speak `text` (default: the phrase itself) while the phrase is highlighted word by word on the paper,
        in sync with the voice. Use it for the sentences the video quotes verbatim."""
        if go is not None:
            self.travel(go)
        st = self._stage
        lifted = False
        if st is not None and st["mode"] in ("look", "aside") and st["target"] is not None and st["crop"] is not None:
            reg = st["target"]
            try:
                lines = self.paper.find(phrase, page=page or reg.page)
                R = reg.pdf_rect
                lifted = lines.page == reg.page and all(abs(r & R) >= 0.6 * abs(r) for r in lines.pdf_rects)
            except KeyError:
                lifted = False
            if not lifted and st["mode"] == "look":
                self.unstage()
        if not lifted:
            lines = self.paper.find(phrase, page=page) if page else self.here(phrase)
        boxes = self.paper.word_boxes(lines)
        if lifted:                       # read along on the enlarged region
            boxes = [self._lifted(b) for b in boxes]
        marks = VGroup(*[Rectangle(width=b.width, height=b.height, stroke_width=0, fill_color=color,
                                   fill_opacity=opacity).move_to(b) for b in boxes])
        with self.voice(text or phrase, pad=pad) as n:
            words = n.clip.words or []
            if text and words:   # only the spoken words of the quoted phrase drive the highlight
                try:
                    t0, t1 = n.at(phrase.split()[0]), n.at(phrase.split()[-1], end=True)
                    sub = [w for w in words if w[1] >= t0 - 0.01 and w[2] <= t1 + 0.01]
                    words = sub or words
                except Exception:
                    pass
            ns, nb = max(len(words), 1), max(len(marks), 1)
            for i, m in enumerate(marks):
                j = min(ns - 1, round(i * (ns - 1) / max(nb - 1, 1))) if words else None
                start = words[j][1] if words else n.duration * i / nb
                end = words[j][2] if words else n.duration * (i + 1) / nb
                d = start - n.now()
                if d > 1 / 60:
                    self.wait(d)
                self.play(GrowFromEdge(m, LEFT), run_time=max(0.06, min(0.5, end - max(start, n.now()))))
        self.qa_event("read", phrase)
        return marks


class Video(WalkScene):
    """The whole video as one program. Subclass it in video.py, set `title` (and `paper` if not paper.pdf),
    and write one method per chapter decorated with @chapter("Name"), taking (self, p).
    `vk build <dir> all` renders each chapter as its own clip in parallel and joins them; each chapter starts
    where the previous one left the camera, with marks and stages closed."""

    title = ""
    paper_file = "paper.pdf"
    chapter_index: int | None = None     # set by the build for per-chapter renders; None = all chapters

    @classmethod
    def chapters(cls) -> list[tuple[str, str]]:
        out = []
        for klass in reversed(cls.__mro__):
            for name, fn in vars(klass).items():
                if callable(fn) and hasattr(fn, "_chapter") and name not in [n for n, _ in out]:
                    out.append((name, fn._chapter))
        return out

    def construct(self):
        p = self.open_paper(self.paper_file)
        self.page_counter()
        chs = self.chapters()
        cams_f = Path("build") / "camera.json"
        todo = range(len(chs)) if self.chapter_index is None else [self.chapter_index]
        if self.chapter_index is not None and cams_f.exists():
            cams = json.loads(cams_f.read_text())
            if self.chapter_index < len(cams):
                cx, cy, w = cams[self.chapter_index]
                self._set_frame(np.array([cx, cy, 0.0]), w)
                p._update(p.view)
        record = []
        CURRENT["starts"] = {}
        tolerant = bool(os.environ.get("VK_TOLERANT"))
        for i in todo:
            f = self.camera.frame
            record.append([float(f.get_center()[0]), float(f.get_center()[1]), float(f.width)])
            CURRENT["chapter"] = i
            CURRENT["starts"][i] = self.renderer.time
            self.qa_event("chapter", chs[i][1], 1.5)
            if not tolerant:
                getattr(self, chs[i][0])(p)
                self._end_chapter()
                continue
            try:                      # fast check: report the error and go on with the next chapter
                getattr(self, chs[i][0])(p)
            except Exception as e:
                import traceback
                tb = [fr for fr in traceback.extract_tb(e.__traceback__) if Path(fr.filename).resolve().parent == Path.cwd()]
                at = f"{Path(tb[-1].filename).name}:{tb[-1].lineno}" if tb else "?"
                CURRENT["note"](f"{type(e).__name__}: {str(e)[:140]} (rest of this chapter not checked)", at)
            try:
                self._end_chapter()
            except Exception:
                self._stage = None
                if self._frozen is not None:
                    self._frozen = None
        CURRENT["end"] = self.renderer.time
        if self.chapter_index is None and os.environ.get("VK_PREPASS"):
            cams_f.parent.mkdir(exist_ok=True)
            cams_f.write_text(json.dumps(record))

    def _end_chapter(self) -> None:
        if self._stage is not None:
            self.unstage()
        if self._frozen is not None:
            self.unfreeze()
        self.clear_marks()


def fetch_main(argv=None) -> None:
    """vk fetch <url> --out assets/ext/name.png [--credit "..."]: download a file + a provenance sidecar."""
    import argparse
    import datetime
    import urllib.request

    ap = argparse.ArgumentParser(prog="vk fetch")
    ap.add_argument("url")
    ap.add_argument("--out", required=True)
    ap.add_argument("--credit", default="")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(a.url, headers={"User-Agent": "video-kit/0.1"})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = r.read()
    out.write_bytes(data)
    if out.suffix.lower() not in (".png", ".jpg", ".jpeg"):
        print(f"warning: {out.suffix} may not load as an image")
    out.with_suffix(".json").write_text(json.dumps({"url": a.url, "credit": a.credit or a.url,
                                                    "fetched": datetime.date.today().isoformat(),
                                                    "bytes": len(data)}, indent=1))
    print(f"wrote {out} ({len(data) // 1024} KB) + {out.with_suffix('.json').name}")


__all__ = [n for n in dir() if not n.startswith("_")]
