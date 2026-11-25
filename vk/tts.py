"""Narration voice: Kokoro-82M, voice af_heart, with word timings.

    synth(text) -> Clip(path, duration, words=[(word, start, end), ...])

Audio is cached by content hash in video_kit/.cache/audio (shared by every video; atomic writes).
Display text conventions (also used on screen by the engine):
    `code`             spoken by spelling punctuation out ("`pass@k`" -> "pass at k")
    `code`[spoken]     shown as code, spoken as given
    {display|spoken}   shown as `display`, spoken as `spoken`
    [word](/fənˈɛtɪks/) Kokoro inline phoneme override (spoken only)
Pronunciations: built-in PRONOUNCE list + optional `pronounce.toml` in the video dir:
    [pronounce]
    "\\bGEPA\\b" = "jeppa"

CLI (audition a line):  ./bin/vk say "text" [--voice af_heart] [--speed 1.0] [--play]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tomllib
import wave
from dataclasses import dataclass, field
from pathlib import Path

KIT = Path(__file__).resolve().parent.parent
CACHE = KIT / ".cache" / "audio"
VOICE = os.environ.get("VK_VOICE", "af_heart")
SPEED = float(os.environ.get("VK_SPEED", "1.0"))
SR = 24000
TAIL = 0.15  # seconds of silence appended to every clip
SILENT = os.environ.get("VK_SILENT") == "1"  # layout previews without audio
WPM = 158  # af_heart at speed 1.0 (measured), used for silent estimates

PRONOUNCE: list[tuple[str, str]] = [
    (r"\bpass@(\w+)", r"pass at \1"), (r"\bavg@(\w+)", r"average at \1"), (r"\bmaj@(\w+)", r"majority at \1"),
    (r"\bRLVR\b", "R L V R"), (r"\bRLHF\b", "R L H F"), (r"\bGRPO\b", "G R P O"), (r"\bPPO\b", "P P O"),
    (r"\bDPO\b", "D P O"), (r"\bDAPO\b", "dah-po"), (r"\bGSPO\b", "G S P O"), (r"\bRLOO\b", "R L O O"),
    (r"\bLLMs\b", "L L Ms"), (r"\bLLM\b", "L L M"), (r"\bSFT\b", "S F T"), (r"\bKL\b", "K L"), (r"\bRL\b", "R L"),
    (r"\bCoT\b", "chain of thought"), (r"\bMCMC\b", "M C M C"), (r"\bMCTS\b", "M C T S"), (r"\bMDP\b", "M D P"),
    (r"(\d)-(\d+(?:\.\d+)?)[Bb]\b", r"\1, \2 B"), (r"\bQwen\b", "Kwen"), (r"\bQwen(\d)", r"Kwen \1"), (r"\bLLaMA\b|\bLlama\b", "Lahma"), (r"\bDeepSeek\b", "Deep Seek"),
    (r"\bAIME\b", "aim"), (r"\bAIME(\d+)", r"aim \1"), (r"\bMATH500\b", "math five hundred"), (r"\bGSM8K\b", "G S M eight K"),
    (r"\bSWE-bench\b", "swee bench"), (r"\bSWE\b", "swee"), (r"\bHumanEval\b", "Human Eval"), (r"\bLiveCodeBench\b", "Live Code Bench"),
    (r"\bDSPy\b", "D S pie"), (r"\bMIPROv2\b", "mipro v two"), (r"\bGPT-4o\b", "G P T four oh"), (r"\bGPT-(\d)", r"G P T \1"),
    (r"\bvLLM\b", "v L L M"), (r"\bLoRA\b", "Laura"), (r"\bGPU(s?)\b", r"G P U\1"), (r"\bAPI\b", "A P I"),
    (r"\bHGM\b", "H G M"), (r"\bDGM\b", "D G M"), (r"\bAZR\b", "A Z R"), (r"\bRAG\b", "rag"), (r"\barXiv\b", "archive"),
    (r"\bNeurIPS\b", "neur-ips"), (r"\bICLR\b", "I C L R"), (r"\bICML\b", "I C M L"), (r"\bPMLR\b", "P M L R"),
    (r"\bTL;DR\b", "T L D R"), (r"\be\.g\.", "for example"), (r"\bi\.e\.", "that is"), (r"\betc\.", "et cetera"),
    (r"\bvs\.?\b", "versus"), (r"(\d)\s?%", r"\1 percent"), (r"\b(\d+(?:\.\d+)?)B\b", r"\1 B"), (r"\b(\d+)M\b", r"\1 million"),
    (r"(\d)k\b", r"\1 thousand"), (r"\bk=(\d+)", r"k equals \1"), (r"\bpp\b", "percentage points"),
]


def _video_pronounce() -> list[tuple[str, str]]:
    f = Path(os.environ.get("VK_VIDEO_DIR", ".")) / "pronounce.toml"
    if f.exists():
        return list(tomllib.loads(f.read_text()).get("pronounce", {}).items())
    return []


def spoken(text: str) -> str:
    """Display text -> the exact string sent to the voice."""

    def code(m: re.Match) -> str:
        if m.group(2):
            return m.group(2)
        c = m.group(1)
        c = c.replace("@", " at ").replace("->", " to ").replace("==", " equals ").replace("<=", " at most ")
        c = c.replace(">=", " at least ").replace("_", " ").replace("/", " slash ").replace("=", " equals ")
        c = re.sub(r"[()\[\]{}\"'`,:]", " ", c)
        return re.sub(r"\s+", " ", c).strip()

    text = re.sub(r"`([^`]+)`(?:\[([^\]]+)\])?", code, text)
    text = re.sub(r"\{([^|}]*)\|([^}]*)\}", r"\2", text)
    text = text.replace("**", "").replace("→", ", to ,").replace("—", ", ").replace("–", " to ").replace("×", " times ")
    for pattern, say in _video_pronounce() + PRONOUNCE:
        text = re.sub(pattern, say, text)
    return re.sub(r"\s+", " ", text).strip()


def display(text: str) -> str:
    """Display text -> what appears on screen (drops spoken-only parts)."""
    text = re.sub(r"`([^`]+)`\[[^\]]+\]", r"`\1`", text)
    text = re.sub(r"\[([^\]]+)\]\(/[^)]+/\)", r"\1", text)
    return re.sub(r"\{([^|}]*)\|([^}]*)\}", r"\1", text)


@dataclass
class Clip:
    path: Path | None
    duration: float
    words: list[tuple[str, float, float]] = field(default_factory=list)
    speech: str = ""

    def at(self, phrase: str, nth: int = 1, end: bool = False) -> float:
        """Seconds from clip start at which `phrase` (one or more words, case-insensitive) starts (or ends)."""
        target = [_norm(w) for w in spoken(phrase).split() if _norm(w)]
        toks = [_norm(w) for w, _, _ in self.words]
        hits = 0
        for i in range(len(toks)):
            if toks[i:i + len(target)] == target:
                hits += 1
                if hits == nth:
                    return self.words[i + len(target) - 1][2] if end else self.words[i][1]
        # fallback: match on the letters alone, so "Schmidhuber's" finds "Schmidhuber" + "'s" and split/merged tokens agree
        flat, owner = "", []
        for i, t in enumerate(toks):
            flat += t
            owner += [i] * len(t)
        want = "".join(target)
        pos, hits = -1, 0
        while want:
            pos = flat.find(want, pos + 1)
            if pos < 0:
                break
            hits += 1
            if hits == nth:
                i0, i1 = owner[pos], owner[pos + len(want) - 1]
                return self.words[i1][2] if end else self.words[i0][1]
        if not self.words:  # silent preview: proportional estimate
            idx = spoken(self.speech).lower().find(spoken(phrase).lower())
            if idx >= 0:
                return self.duration * idx / max(1, len(self.speech))
        raise KeyError(f"phrase {phrase!r} not found in narration: {self.speech!r}")


def _norm(w: str) -> str:
    return re.sub(r"[^a-z0-9]", "", w.lower())


_PIPE = None


def _pipeline():
    global _PIPE
    if _PIPE is None:
        import warnings

        warnings.filterwarnings("ignore")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        if (Path.home() / ".cache/huggingface/hub/models--hexgrad--Kokoro-82M").exists():
            os.environ.setdefault("HF_HUB_OFFLINE", "1")  # already downloaded: no network round-trips
        from kokoro import KPipeline

        _PIPE = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
    return _PIPE


def synth(text: str, voice: str | None = None, speed: float | None = None) -> Clip:
    voice, speed = voice or VOICE, speed or SPEED
    speech = spoken(text)
    if not speech:
        return Clip(None, 0.0, [], speech)
    if SILENT:
        return Clip(None, max(1.2, len(speech.split()) / WPM * 60), [], speech)
    key = hashlib.sha1(f"kokoro|{voice}|{speed}|{speech}".encode()).hexdigest()[:16]
    CACHE.mkdir(parents=True, exist_ok=True)
    wav, meta = CACHE / f"{key}.wav", CACHE / f"{key}.json"
    if not (wav.exists() and meta.exists()):
        import numpy as np
        import soundfile as sf

        from .slots import slot

        with slot("tts", quiet=True):
            chunks, words, offset = [], [], 0.0
            for result in _pipeline()(speech, voice=voice, speed=speed):
                audio = result.audio.numpy() if hasattr(result.audio, "numpy") else np.asarray(result.audio)
                for t in result.tokens or []:
                    if t.start_ts is not None and _norm(t.text):
                        words.append((t.text, round(offset + t.start_ts, 3), round(offset + (t.end_ts or t.start_ts), 3)))
                chunks.append(audio)
                offset += len(audio) / SR
            audio = np.concatenate(chunks + [np.zeros(int(SR * TAIL), dtype=chunks[0].dtype)])
            tmp = wav.with_suffix(f".{os.getpid()}.tmp.wav")
            sf.write(str(tmp), audio, SR, subtype="PCM_16")
            meta.with_suffix(f".{os.getpid()}.tmp").write_text(json.dumps({"speech": speech, "voice": voice, "speed": speed, "words": words}))
            os.replace(meta.with_suffix(f".{os.getpid()}.tmp"), meta)
            os.replace(tmp, wav)
    with wave.open(str(wav)) as h:
        dur = h.getnframes() / h.getframerate()
    words = [tuple(w) for w in json.loads(meta.read_text())["words"]]
    return Clip(wav, dur, words, speech)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="vk say", description="Audition narration: synthesize, print timing, optionally play.")
    ap.add_argument("text")
    ap.add_argument("--voice", default=VOICE)
    ap.add_argument("--speed", type=float, default=SPEED)
    ap.add_argument("--play", action="store_true", help="play through the speakers (afplay)")
    a = ap.parse_args(argv)
    clip = synth(a.text, a.voice, a.speed)
    print(f"spoken: {clip.speech}\nfile:   {clip.path}\nlength: {clip.duration:.2f}s  ({len(clip.speech.split()) / clip.duration * 60:.0f} wpm)")
    print("words:  " + " ".join(f"{w}@{s:.2f}" for w, s, _ in clip.words))
    if a.play and clip.path:
        subprocess.run(["afplay", str(clip.path)])


if __name__ == "__main__":
    main(sys.argv[1:])
