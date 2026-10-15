"""Live transcript of a video call you hear through headphones.

The other side is captured from loopback of the output devices (what you hear), your own voice from the
microphone. Speech is cut at pauses, transcribed, and appended to a text file:

    [15:03:12] THEM: ...
    [15:03:20] ME: ...

Devices are rescanned every few seconds, so headphones connected after start are picked up.
"""
import argparse
import base64
import concurrent.futures
import datetime
import io
import json
import os
import re
import sys
import threading
import time
import urllib.request
import wave

import numpy as np
import soundcard as sc

SR = 16000
FRAME = 1600  # 100 ms at 16 kHz

# Phrases Whisper-family models invent on silence or noise.
HALLUCINATION = re.compile(r"thanks? (you )?for watching|subscribe to|subtitles by|amara\.org", re.I)


def log(msg):
    print(f"[{datetime.datetime.now():%H:%M:%S}] {msg}", flush=True)


def to_wav(audio):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
    return buf.getvalue()


class GeminiEngine:
    def __init__(self, model, lang="en", terms=""):
        self.key = os.environ.get("GEMINI_API_KEY") or os.environ["GOOGLE_API_KEY"]
        self.model, self.lang, self.terms = model, lang, terms
        self.name = f"gemini:{model}"

    def transcribe(self, audio):
        prompt = (f"Transcribe the speech in this audio verbatim (language: {self.lang}). Return only the spoken "
                  f"text, no comments. If there is no speech, return an empty string.")
        if self.terms:
            prompt += " Names and terms that may occur: " + self.terms
        body = {
            "contents": [{"parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "audio/wav", "data": base64.b64encode(to_wav(audio)).decode()}},
            ]}],
            "generationConfig": {"temperature": 0},
        }
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "x-goog-api-key": self.key})
        with urllib.request.urlopen(req, timeout=20) as r:
            js = json.loads(r.read())
        parts = js["candidates"][0].get("content", {}).get("parts", [])
        return " ".join(p.get("text", "") for p in parts).strip()


class OpenAIEngine:
    def __init__(self, model, lang="en", terms=""):
        from openai import OpenAI
        self.client = OpenAI(timeout=20, max_retries=1)
        self.model, self.lang, self.terms = model, lang, terms
        self.name = f"openai:{model}"

    def transcribe(self, audio):
        r = self.client.audio.transcriptions.create(
            model=self.model, file=("segment.wav", to_wav(audio)), language=self.lang,
            prompt=self.terms, response_format="text")
        return (r if isinstance(r, str) else r.text).strip()


class LocalEngine:
    def __init__(self, size, threads, lang="en", terms=""):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(size, device="cpu", compute_type="int8", cpu_threads=threads)
        self.lang, self.terms = lang, terms
        self.name = f"local:{size}"

    def transcribe(self, audio):
        segments, _ = self.model.transcribe(
            audio, language=self.lang, beam_size=1, initial_prompt=self.terms or None,
            condition_on_previous_text=False, vad_filter=False)
        return " ".join(s.text.strip() for s in segments).strip()


class Segmenter:
    """Energy-based voice activity detection: emits one chunk of audio per utterance."""

    def __init__(self, label, emit, max_len=12.0, end_silence=0.7):
        self.label, self.emit = label, emit
        self.max_frames = int(max_len * 10)
        self.end_frames = int(end_silence * 10)
        self.noise_db = -60.0
        self.preroll, self.buf = [], []
        self.silent, self.start_ts, self.last_feed = 0, None, time.time()
        self.lock = threading.Lock()

    def feed(self, frame):
        now = time.time()
        with self.lock:
            self.last_feed = now
            db = 20 * np.log10(np.sqrt(np.mean(frame ** 2)) + 1e-9)
            threshold = max(self.noise_db + 12, -48)
            speech = db > threshold
            if not self.buf:
                if not speech:
                    self.noise_db = 0.95 * self.noise_db + 0.05 * db if db > -90 else self.noise_db
                    self.preroll = (self.preroll + [frame])[-3:]
                    return
                self.start_ts = now - 0.1 * (len(self.preroll) + 1)
                self.buf = self.preroll + [frame]
                self.preroll, self.silent = [], 0
                return
            self.buf.append(frame)
            self.silent = 0 if speech else self.silent + 1
            if self.silent >= self.end_frames or len(self.buf) >= self.max_frames:
                self._flush()

    def tick(self):
        """Flush when the device stopped delivering audio in the middle of an utterance."""
        with self.lock:
            if self.buf and time.time() - self.last_feed > 1.0:
                self._flush()

    def _flush(self):
        voiced = len(self.buf) - self.silent
        if voiced >= 4:  # at least ~0.4 s of speech
            self.emit(self.label, self.start_ts, np.concatenate(self.buf))
        self.buf, self.silent = [], 0


class Capture(threading.Thread):
    def __init__(self, device, segmenter):
        super().__init__(daemon=True)
        self.device, self.segmenter = device, segmenter
        self.stopped = threading.Event()

    def run(self):
        try:
            with self.device.recorder(samplerate=SR, channels=1, blocksize=FRAME) as rec:
                while not self.stopped.is_set():
                    data = rec.record(numframes=FRAME)
                    self.segmenter.feed(data[:, 0].astype(np.float32))
        except Exception as e:  # device vanished, switched profile, or access denied
            if "0x80070005" in str(e):
                log(f"no access to microphone {self.device.name}: turn on 'Microphone access' and "
                    f"'Let desktop apps access your microphone' in Settings > Privacy > Microphone")
            else:
                log(f"device unavailable: {self.device.name} ({type(e).__name__}: {str(e)[:80]})")


class Transcriber:
    def __init__(self, args):
        self.out = open(args.out, "a", encoding="utf-8", buffering=1)
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self.terms = args.terms
        gemini = (lambda: GeminiEngine(args.gemini_model, args.lang, args.terms))
        openai = (lambda: OpenAIEngine(args.openai_model, args.lang, args.terms))
        local = (lambda: LocalEngine(args.local_model, args.threads, args.lang, args.terms))
        # Engines are tried in order; each is created on first use, so a missing key or model costs nothing.
        self.chain = {"auto": [gemini, openai, local], "gemini": [gemini], "openai": [openai],
                      "local": [local]}[args.engine]
        self.engines = [None] * len(self.chain)
        self.engine_lock = threading.Lock()
        self.mic_pattern = re.compile(args.mic, re.I)
        self.captures = {}
        self.segmenters = {}
        self.retry_at = {}  # device id -> time when a failed device may be reopened
        order = {"auto": "Gemini > OpenAI > local model"}.get(args.engine, args.engine)
        log(f"engines: {order}; transcript: {args.out}")

    def submit(self, label, ts, audio):
        self.pool.submit(self._work, label, ts, audio)

    def _engine(self, i):
        with self.engine_lock:
            if self.engines[i] is None:
                self.engines[i] = self.chain[i]()
            return self.engines[i]

    def _work(self, label, ts, audio):
        t0 = time.time()
        text = None
        for i in range(len(self.chain)):
            try:
                engine = self._engine(i)
                text = engine.transcribe(audio)
                break
            except Exception as e:
                log(f"transcription failed ({type(e).__name__}: {str(e)[:80]}), trying the next engine")
        if not text or HALLUCINATION.search(text):
            return
        # On silence an LLM engine sometimes reads back the glossary from its prompt.
        if len(self.terms) >= 30 and self.terms[:30].lower() in text.lower():
            return
        stamp = datetime.datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        line = f"[{stamp}] {label}: {text}"
        self.out.write(line + "\n")
        print(f"{line}   ({len(audio) / SR:.1f} s of speech, {engine.name} in {time.time() - t0:.1f} s)", flush=True)

    def _wanted(self):
        wanted = {}
        for spk in sc.all_speakers():
            mic = sc.get_microphone(spk.id, include_loopback=True)
            wanted[mic.id] = (mic, "THEM")
        mics = [m for m in sc.all_microphones() if not m.isloopback]
        preferred = [m for m in mics if self.mic_pattern.search(m.name)]
        for m in preferred or [sc.default_microphone()]:
            wanted[m.id] = (m, "ME")
        return wanted

    def rescan(self):
        try:
            wanted = self._wanted()
        except Exception as e:
            log(f"could not list audio devices: {e}")
            return
        for dev_id in list(self.captures):
            cap = self.captures[dev_id]
            if dev_id not in wanted or not cap.is_alive():
                if not cap.is_alive():
                    self.retry_at[dev_id] = time.time() + 30
                cap.stopped.set()
                del self.captures[dev_id]
        for dev_id, (dev, label) in wanted.items():
            if dev_id not in self.captures and time.time() >= self.retry_at.get(dev_id, 0):
                seg = self.segmenters.setdefault(dev_id, Segmenter(label, self.submit))
                cap = Capture(dev, seg)
                cap.start()
                self.captures[dev_id] = cap
                log(f"listening [{label}] {dev.name}")

    def run(self):
        last_scan = 0
        while True:
            if time.time() - last_scan > 3:
                self.rescan()
                last_scan = time.time()
            for seg in self.segmenters.values():
                seg.tick()
            time.sleep(0.2)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # auto = Gemini, then OpenAI, then the local model. On our benchmark (bench.py) gpt-4o-transcribe and
    # gemini-3.1-flash-lite were equally accurate and fast; Gemini is ~7x cheaper (~$0.1 per hour of call).
    p.add_argument("--engine", choices=["auto", "gemini", "openai", "local"], default="auto")
    p.add_argument("--gemini-model", default="gemini-3.1-flash-lite")
    p.add_argument("--openai-model", default="gpt-4o-transcribe")
    p.add_argument("--local-model", default="small")
    p.add_argument("--threads", type=int, default=8, help="CPU threads for the local model")
    p.add_argument("--lang", default="en", help="spoken language, ISO 639-1 code")
    p.add_argument("--terms", default="", help="names and jargon to help recognition, e.g. 'Kubernetes, Jane Doe'")
    p.add_argument("--mic", default="AirPods|Headset|Hands-Free", help="regex of the preferred microphone")
    p.add_argument("--out", default=os.path.join(here, "calls", f"call-{datetime.datetime.now():%Y%m%d-%H%M}.txt"))
    args = p.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(os.path.join(os.path.dirname(os.path.abspath(args.out)), "LATEST"), "w", encoding="utf-8") as f:
        f.write(os.path.abspath(args.out))
    sys.stdout.reconfigure(encoding="utf-8")
    try:
        Transcriber(args).run()
    except KeyboardInterrupt:
        log("stopped")


if __name__ == "__main__":
    main()
