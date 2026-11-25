"""House illustration components (3Blue1Brown style, house palette). Use these before hand-building.
Every builder returns plain manim mobjects, sized in default frame units; place them with `fit(mob, area)`.

  Data
    bar_chart(names, vals, colors, top)            -> (group, bars, nums)   bars on a zero rule (negatives hang below);
                                                      pass top= the largest value a later rebar reaches
    rebar(chart, new_vals, new_names)              -> animations that morph bars + numbers to new values
    line_plot({name: (xs, ys, color)}, x_range, y_range, x_label, y_label)
                                                   -> Plot (.ax, .curves[name], .labels[name]); self.play(*plot.draw())
    density(ax, xs, ys, color)                     -> filled curve under ys (distributions)
    live_density(ax, xs, base, tracker, transform, color) -> always_redraw area, e.g. sharpen p**alpha
    number(tracker, fmt, label, color)             -> live readout "label = 0.73"
    heat(matrix, row_labels, col_labels, color)    -> grid of shaded cells with values
  Structures
    tokens(words | n, color)                       -> TokenRow: .cells, .pointer(i), .recolor(i, j, color)
    prob_tree({"a": (p, {"aa": p, ...}), ...})     -> TreeDiagram: .nodes[path], .edges[path], .leaf_labels[path]
    flow(["prompt", "model", "answer"], colors)    -> text nodes joined by arrows (no boxes)
    dots(n, k, color)                              -> n dots in a grid, the first k coloured (magnitudes, rates)
  Text
    code(lines | "src")                            -> monospace block keeping indentation: .lines[i], .mark(i, colour)
    stack(*lines)                                  -> left-aligned lines with even spacing
    threads([(title, subtitle, color), ...])       -> coloured-rule list (follow-up threads, options)
    month_bars(labels, counts, color, year_marks)  -> citation-timeline bars
    compare(left, right, "title L", "title R")     -> two columns with small titles, a thin divider
"""
from __future__ import annotations

import numpy as np
from manim import *  # noqa: F403

from .engine import BG, DIM, FAINT, GOLD, INK, M, T, Tx

__all__ = ["code", "bar_chart", "rebar", "line_plot", "Plot", "density", "live_density", "number", "heat", "tokens",
           "TokenRow", "prob_tree", "TreeDiagram", "flow", "dots", "stack", "threads", "month_bars", "compare"]


# ---------------------------------------------------------------- data
def _wrap_label(text, width, size, color=DIM):
    """A bar label that fits `width`: one line, else 2-3 lines split at spaces, else shrunk (never below 70%)."""
    words = text.split()
    best = T(text, size=size, color=color)
    for n in (2, 3):
        if best.width <= width or len(words) < n:
            break
        per = -(-len(words) // n)
        chunks = [" ".join(words[i:i + per]) for i in range(0, len(words), per)]
        best = VGroup(*[T(c, size=size, color=color) for c in chunks]).arrange(DOWN, buff=0.06)
    if best.width > width:
        best.scale_to_fit_width(width)
    return best


def bar_chart(names, vals, colors, top=None, height=3.6, bar_w=0.72, gap=0.42, fmt="{:.1f}", size=24, bottom=None):
    """Bars on a zero rule, value at the bar's end, name under the rule. Negative values hang below the rule.
    `top` (and `bottom` for negatives) fix the scale: pass the largest value any later rebar() will reach, and the
    chart reserves that room from the start, so a fit() title never collides when bars grow. Names wrap/shrink to
    their slot. Animate: GrowFromEdge(bars[i], DOWN) + FadeIn(nums[i]). Returns (group, bars, nums)."""
    vals = list(vals)
    top = max([v for v in vals] + [1e-9]) if top is None else top
    bottom = min([v for v in vals] + [0.0]) if bottom is None else bottom
    unit = height / max(top - min(bottom, 0), 1e-9)
    pitch = bar_w + gap
    bars, nums, labs = VGroup(), VGroup(), VGroup()
    for i, (n, v, c) in enumerate(zip(names, vals, colors)):
        b = Rectangle(width=bar_w, height=max(1e-3, unit * abs(v)), stroke_width=0, fill_color=c, fill_opacity=0.9)
        b.move_to([i * pitch, 0, 0], aligned_edge=DOWN if v >= 0 else UP)
        bars.add(b)
        nums.add(T(fmt.format(v), size=size, color=c).next_to(b, UP if v >= 0 else DOWN, buff=0.1))
        lab = _wrap_label(n, pitch * 1.02, size - 2)
        low = unit * bottom - (T("0", size=size).height + 0.1) if bottom < 0 else 0
        labs.add(lab.next_to([b.get_x(), low, 0], DOWN, buff=0.2))
    x0, x1 = -bar_w / 2 - 0.2, (len(vals) - 1) * pitch + bar_w / 2 + 0.2
    rule = Line([x0, 0, 0], [x1, 0, 0], stroke_color=DIM, stroke_width=2)
    num_h = T("0", size=size).height + 0.1
    strut = Line([x0, unit * min(bottom, 0) - (num_h if bottom < 0 else 0), 0], [x0, unit * top + num_h, 0],
                 stroke_opacity=0, stroke_width=0)          # reserves room for the largest future value
    labs.add(strut)                                         # (kept inside labs so the group stays 4 parts)
    g = VGroup(rule, bars, nums, labs)
    g.fmt, g.top, g.height_units, g._w0, g.size, g.unit0 = fmt, top, height, rule.width, size, unit
    return g, bars, nums


def rebar(chart, new_vals, new_names=None):
    """Animations that morph an existing bar_chart (even after fit/scale) to new values (signs may change)."""
    rule, bars, nums, labs = chart
    k = rule.width / chart._w0
    unit = chart.unit0 * k
    anims = []
    for i, v in enumerate(new_vals):
        target = bars[i].copy().stretch_to_fit_height(max(1e-3, unit * abs(v))).move_to(
            [bars[i].get_x(), rule.get_y(), 0], aligned_edge=DOWN if v >= 0 else UP)
        num = T(chart.fmt.format(v), size=chart.size, color=bars[i].get_fill_color()).scale_to_fit_height(nums[i].height)
        anims += [Transform(bars[i], target),
                  Transform(nums[i], num.next_to(target, UP if v >= 0 else DOWN, buff=0.1 * k))]
        if new_names:
            pitch = (bars[1].get_x() - bars[0].get_x()) if len(bars) > 1 else bars[0].width * 1.6
            new = _wrap_label(new_names[i], pitch * 1.02 / k, chart.size - 2).scale(k)
            anims.append(Transform(labs[i], new.move_to(labs[i], aligned_edge=UP)))
    return anims


class Plot(VGroup):
    """Axes with thin ticks, curves labelled at their right ends. self.play(*plot.draw()) draws them in order."""

    def __init__(self, series: dict, x_range, y_range, x_label="", y_label="", x_len=6.0, y_len=4.0, smooth=True,
                 label_size=24):
        super().__init__()
        self.ax = Axes(x_range=x_range, y_range=y_range, x_length=x_len, y_length=y_len, tips=False,
                       axis_config={"color": DIM, "include_ticks": True, "tick_size": 0.05, "stroke_width": 2})
        self.ax.add_coordinates(font_size=20)
        self.add(self.ax)
        if x_label:
            self.add(T(x_label, size=22, color=DIM).next_to(self.ax, DOWN, buff=0.35))
        if y_label:
            self.add(T(y_label, size=22, color=DIM).rotate(PI / 2).next_to(self.ax, LEFT, buff=0.35))
        self.curves, self.labels = {}, {}
        ends = []
        for name, (xs, ys, color) in series.items():
            pts = [self.ax.c2p(x, y) for x, y in zip(xs, ys)]
            c = VMobject(stroke_color=color, stroke_width=4)
            c.set_points_smoothly(pts) if smooth else c.set_points_as_corners(pts)
            self.curves[name] = c
            lab = T(name, size=label_size, color=color).next_to(pts[-1], RIGHT, buff=0.12)
            self.labels[name] = lab
            ends.append(lab)
        ends.sort(key=lambda m: m.get_y())
        for a, b in zip(ends, ends[1:]):                       # de-collide end labels
            if b.get_bottom()[1] < a.get_top()[1] + 0.05:
                b.shift(UP * (a.get_top()[1] + 0.05 - b.get_bottom()[1]))
        self.add(*self.curves.values(), *self.labels.values())

    def draw(self, order=None, run_time=1.2, axes=True):
        """Animations: the axes (with tick numbers and axis titles) first, then each curve with its end label.
        self.play(*plot.draw()). axes=False if the axes are already on screen."""
        order = order or list(self.curves)
        steps = []
        if axes:
            extras = [m for m in self.submobjects if m is not self.ax and m not in self.curves.values()
                      and m not in self.labels.values()]
            steps.append(AnimationGroup(Create(self.ax), *[FadeIn(m) for m in extras], run_time=0.8))
        steps += [AnimationGroup(Create(self.curves[n]), FadeIn(self.labels[n]), run_time=run_time) for n in order]
        return [Succession(*steps)]


def line_plot(series, x_range, y_range, x_label="", y_label="", **kw) -> Plot:
    return Plot(series, x_range, y_range, x_label, y_label, **kw)


def density(ax: Axes, xs, ys, color, opacity=0.4) -> Polygon:
    pts = [ax.c2p(x, y) for x, y in zip(xs, ys)]
    return Polygon(ax.c2p(xs[0], 0), *pts, ax.c2p(xs[-1], 0), stroke_color=color, stroke_width=3,
                   fill_color=color, fill_opacity=opacity)


def live_density(ax: Axes, xs, base, tracker: ValueTracker, transform, color, opacity=0.4):
    """always_redraw density of transform(base, tracker.get_value()), e.g. lambda b, a: b**a / (b**a).sum() * b.sum()."""
    return always_redraw(lambda: density(ax, xs, transform(np.asarray(base), tracker.get_value()), color, opacity))


def number(tracker: ValueTracker, fmt="{:.2f}", label="", color=INK, size=34):
    """Live readout 'label = value' following a ValueTracker. Place it like any mobject (next_to, move_to, fit):
    it stays where you put it, left-aligned, while the value changes."""
    def make():
        return Tx((label + "=" if label else "") + fmt.format(tracker.get_value()), size=size, color=color)

    m = make()

    def upd(mob):
        new = make().scale_to_fit_height(mob.height) if mob.height > 0 else make()
        new.move_to(mob.get_left(), aligned_edge=LEFT)
        mob.become(new)
    m.add_updater(upd)
    return m


def code(lines, size=22, color=INK, line_gap=0.12, colors=None):
    """Monospace code block that keeps indentation. lines: a string (split on newlines) or a list.
    Returns a VGroup with .lines[i] (one mobject per line, empty lines included) and .mark(i, color) -> a
    marker rectangle behind line i (FadeIn it). colors={i: colour} tints whole lines."""
    if isinstance(lines, str):
        lines = lines.strip("\n").split("\n")
    probe = M("M" * 20, size=size)
    cw, lh = probe.width / 20, probe.height
    g = VGroup()
    rows = VGroup()
    for i, ln in enumerate(lines):
        body = ln.rstrip()
        ind = len(body) - len(body.lstrip(" "))
        if body.strip():
            t = M(body.strip(), size=size, color=(colors or {}).get(i, color))
        else:
            t = VMobject(stroke_opacity=0, stroke_width=0).set_points_as_corners([ORIGIN, RIGHT * 0.01])
        y = -i * (lh + line_gap)
        t.move_to([ind * cw, y, 0], aligned_edge=LEFT)
        t.set_y(y)
        rows.add(t)
    g.add(rows)
    g.lines = rows
    g.cw, g.lh, g.gap = cw, lh, line_gap

    g.h0 = g.height

    def mark(i, col=GOLD, opacity=0.22):
        k = g.height / g.h0 if g.h0 else 1.0
        return Rectangle(width=g.width + 0.2 * k, height=(lh + line_gap) * k, stroke_width=0, fill_color=col,
                         fill_opacity=opacity).move_to([g.get_center()[0], rows[i].get_y(), 0])
    g.mark = mark
    return g


def heat(matrix, row_labels=(), col_labels=(), color=GOLD, cell=0.7, fmt="{:.2f}", vmax=None):
    m = np.asarray(matrix, dtype=float)
    vmax = vmax or (m.max() or 1)
    g = VGroup()
    for i, row in enumerate(m):
        for j, v in enumerate(row):
            sq = Square(side_length=cell, stroke_color=BG, stroke_width=2, fill_color=color, fill_opacity=0.12 + 0.8 * v / vmax)
            sq.move_to([j * cell, -i * cell, 0])
            g.add(VGroup(sq, T(fmt.format(v), size=18, color=INK).move_to(sq)))
    for i, l in enumerate(row_labels):
        g.add(T(l, size=20, color=DIM).next_to([0, -i * cell, 0], LEFT, buff=cell / 2 + 0.1))
    for j, l in enumerate(col_labels):
        g.add(T(l, size=20, color=DIM).next_to([j * cell, 0, 0], UP, buff=cell / 2 + 0.1))
    return g


# ---------------------------------------------------------------- structures
class TokenRow(VGroup):
    """A row of token cells (filled squares, optional mono text). .pointer(i) -> a small triangle above cell i;
    .recolor(i, j, color) -> animations recolouring cells i..j-1."""

    def __init__(self, words, color, size=0.5, gap=0.06, font=26):
        super().__init__()
        n = words if isinstance(words, int) else len(words)
        self.cells = VGroup()
        for k in range(n):
            txt = None if isinstance(words, int) else words[k]
            w = size if txt is None else max(size, M(txt, size=font).width + 0.25)
            r = Rectangle(width=w, height=size, stroke_color=color, stroke_width=2, fill_color=color, fill_opacity=0.5)
            cell = VGroup(r) if txt is None else VGroup(r, M(txt, size=font, color=INK).move_to(r))
            self.cells.add(cell)
        self.cells.arrange(RIGHT, buff=gap)
        self.add(self.cells)

    def pointer(self, i, color=GOLD):
        return Triangle(color=color, fill_opacity=1).scale(0.1).rotate(PI).next_to(self.cells[i], UP, buff=0.08)

    def recolor(self, i, j, color):
        return [self.cells[k][0].animate.set_fill(color, 0.55).set_stroke(color) for k in range(i, j)]


def tokens(words, color, **kw) -> TokenRow:
    return TokenRow(words, color, **kw)


class TreeDiagram(VGroup):
    """Left-to-right probability tree. spec = {"a": (0.4, {"aa": 0.0, "ab": 0.4}), "b": (0.5, {...})}.
    .nodes[path] dots, .edges[path] lines into that node, .labels[path] (token), .values[path] (number)."""

    def __init__(self, spec, color=INK, value_color=None, dx=2.2, dy=1.6, fmt="{:.2f}"):
        super().__init__()
        self.nodes, self.edges, self.labels, self.values = {}, {}, {}, {}
        root = Dot(ORIGIN, color=color)
        self.nodes[""] = root
        self.add(root)

        def place(children, parent_key, depth, y0, y1):
            keys = list(children)
            for i, k in enumerate(keys):
                val = children[k]
                p, sub = val if isinstance(val, tuple) else (val, None)
                y = y0 + (y1 - y0) * (i + 0.5) / len(keys)
                d = Dot([dx * depth, y, 0], color=color, radius=0.06)
                e = Line(self.nodes[parent_key].get_center(), d.get_center(), stroke_color=DIM, stroke_width=2)
                self.nodes[k], self.edges[k] = d, e
                self.labels[k] = M(k, size=26, color=INK).next_to(d, UP if sub else RIGHT, buff=0.1)
                v = T(fmt.format(p), size=24, color=value_color or color)
                self.values[k] = v.next_to(self.labels[k], RIGHT, buff=0.18) if not sub else v.next_to(d, DOWN, buff=0.1)
                self.add(e, d, self.labels[k], self.values[k])
                if sub:
                    span = (y1 - y0) / len(keys)
                    place(sub, k, depth + 1, y - span / 2, y + span / 2)

        place(spec, "", 1, dy * len(spec) / 2, -dy * len(spec) / 2)


def prob_tree(spec, **kw) -> TreeDiagram:
    return TreeDiagram(spec, **kw)


def flow(names, colors=None, size=30, gap=1.3):
    """Text nodes joined by arrows, no boxes: flow(["prompt", "base model", "answer"])."""
    colors = colors or [INK] * len(names)
    nodes = VGroup(*[T(n, size=size, color=c) for n, c in zip(names, colors)]).arrange(RIGHT, buff=gap)
    arrows = VGroup(*[Arrow(a.get_right(), b.get_left(), buff=0.15, color=DIM, stroke_width=3,
                            max_tip_length_to_length_ratio=0.15) for a, b in zip(nodes, nodes[1:])])
    g = VGroup(nodes, arrows)
    g.nodes, g.arrows = nodes, arrows
    return g


def dots(n, k=0, color=GOLD, cols=None, r=0.06, gap=0.08):
    cols = cols or int(np.ceil(np.sqrt(n * 2)))
    g = VGroup(*[Dot(radius=r, color=color if i < k else FAINT) for i in range(n)])
    g.arrange_in_grid(cols=cols, buff=gap)
    return g


# ---------------------------------------------------------------- text layouts
def stack(*lines, size=28, color=INK, buff=0.25):
    items = [T(l, size=size, color=color) if isinstance(l, str) else l for l in lines]
    return VGroup(*items).arrange(DOWN, aligned_edge=LEFT, buff=buff)


def threads(items, size=28):
    """[(title, subtitle, color), ...] -> rows with a coloured rule on the left (no boxes)."""
    rows = VGroup()
    for title, sub, c in items:
        txt = VGroup(T(title, size=size, color=c), T(sub, size=size - 7, color=INK)).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        rule = Line(txt.get_corner(UL) + LEFT * 0.18, txt.get_corner(DL) + LEFT * 0.18, stroke_color=c, stroke_width=4)
        rows.add(VGroup(rule, txt))
    return rows.arrange(DOWN, aligned_edge=LEFT, buff=0.38)


def month_bars(labels, counts, color, year_marks=None, unit=0.26, bar_w=0.5):
    """Citation timeline: one bar per month, month initials below, optional {index: "2026"} year marks."""
    bars = VGroup(*[Rectangle(width=bar_w, height=max(1e-3, unit * c), stroke_width=0, fill_color=color, fill_opacity=0.85)
                    for c in counts]).arrange(RIGHT, buff=0.12, aligned_edge=DOWN)
    labs = VGroup(*[T(l, size=20, color=DIM).next_to(b, DOWN, buff=0.12) for l, b in zip(labels, bars)])
    g = VGroup(bars, labs)
    for i, y in (year_marks or {}).items():
        g.add(T(y, size=20, color=DIM).next_to(labs[i], DOWN, buff=0.12))
    g.bars = bars
    return g


def compare(left, right, title_l="", title_r="", gap=1.2):
    cols = []
    for mob, title in ((left, title_l), (right, title_r)):
        col = VGroup(T(title, size=24, color=DIM), mob).arrange(DOWN, buff=0.35) if title else VGroup(mob)
        cols.append(col)
    g = VGroup(*cols).arrange(RIGHT, buff=gap, aligned_edge=UP)
    div = Line(UP, DOWN, stroke_color=FAINT, stroke_width=1.5).stretch_to_fit_height(g.height).move_to(
        [(cols[0].get_right()[0] + cols[1].get_left()[0]) / 2, g.get_y(), 0])
    g.add(div)
    return g
