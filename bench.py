"""Compare transcription engines on your own recordings: latency and accuracy against reference texts.

Put 16 kHz mono WAV files into samples/ and their exact texts into samples/references.txt, one line per file
in name order. Accuracy is the character error rate (CER) after lower-casing and removing punctuation and spaces.

    python bench.py              # all engines
    python bench.py gemini       # only engines whose name contains "gemini"
"""
import glob
import os
import re
import sys
import time
import wave

import numpy as np

from earpiece import SR, GeminiEngine, LocalEngine, OpenAIEngine

FACTORIES = {
    "gemini:gemini-3.1-flash-lite": lambda: GeminiEngine("gemini-3.1-flash-lite"),
    "gemini:gemini-flash-latest": lambda: GeminiEngine("gemini-flash-latest"),
    "openai:gpt-4o-transcribe": lambda: OpenAIEngine("gpt-4o-transcribe"),
    "openai:gpt-4o-mini-transcribe": lambda: OpenAIEngine("gpt-4o-mini-transcribe"),
    "local:small": lambda: LocalEngine("small", 8),
}


def load(path):
    with wave.open(path) as w:
        assert w.getframerate() == SR and w.getnchannels() == 1, f"{path}: need 16 kHz mono"
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768


def norm(text):
    return re.sub(r"[\W_]+", "", text.lower())


def cer(ref, hyp):
    a, b = norm(ref), norm(hyp)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] / max(len(a), 1)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    samples = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")
    files = sorted(glob.glob(os.path.join(samples, "*.wav")))
    refs = open(os.path.join(samples, "references.txt"), encoding="utf-8").read().splitlines()
    audios = [(os.path.basename(f), load(f), refs[i]) for i, f in enumerate(files)]
    only = sys.argv[1:]
    summary = []
    for name, make in FACTORIES.items():
        if only and not any(o in name for o in only):
            continue
        try:
            engine = make()
            engine.transcribe(audios[0][1][: SR * 2])  # warm-up: model load, connection setup
        except Exception as e:
            print(f"\n=== {name}: unavailable ({type(e).__name__}: {str(e)[:100]})")
            continue
        print(f"\n=== {name}")
        times, errs = [], []
        for fname, audio, ref in audios:
            t0 = time.time()
            try:
                text = engine.transcribe(audio)
            except Exception as e:
                text = f"ERROR {type(e).__name__}: {str(e)[:120]}"
            dt = time.time() - t0
            e = cer(ref, text)
            times.append(dt)
            errs.append(e)
            print(f"  {fname} {dt:4.1f} s  CER {e:5.1%}  {text}")
        summary.append((name, np.mean(times), np.max(times), np.mean(errs)))
    print("\n=== Summary (mean time / max time / mean CER)")
    for name, avg, mx, err in sorted(summary, key=lambda s: s[3]):
        print(f"  {name:32s} {avg:4.1f} s / {mx:4.1f} s / CER {err:5.1%}")


if __name__ == "__main__":
    main()
