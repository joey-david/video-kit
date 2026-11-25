---
name: paper-video
description: Turn a research paper (PDF) into a narrated 10-15 min walkthrough video. The PDF scrolls on screen with synced highlights, and 3Blue1Brown-style manim illustrations take over where an idea needs a picture. Use when asked to make an explainer or walkthrough video of a paper.
---

# Paper walkthrough video

`vk` below means `<this repo>/bin/vk`. One-time install: `./setup.sh` (needs ffmpeg, espeak-ng, imagemagick, LaTeX, uv).

1. Read `BRIEF.md` in full. It is the complete spec: program shape, API, storytelling rules, style, workflow.
2. `vk new paper.pdf` scaffolds `videos/<name>/` with a `video.py` stub and the paper's text in `build/paper.txt`.
3. Write `video.py`: one `@chapter` method per section, in paper order (plus optional `anims.py`).
4. `vk build videos/<name> check` (~10 s) lists every problem and estimates the length. Fix, repeat.
5. `vk build videos/<name> all -q l`, then look with `vk qa key videos/<name>`. Fix, repeat.
6. `vk build videos/<name> all -q h` writes `videos/<name>/out/<name>.mp4` with subtitles and chapters.
