# earpiece

A whisper in your ear during video calls. `earpiece.py` transcribes the call live: what you hear (THEM) and
what you say (ME). [Claude Code](https://claude.com/claude-code) watches the transcript and drops a short hint
as soon as the other side asks a question.

Works with any call app (Zoom, Meet, Teams…) because it records the system audio, not the app.
Windows and Linux.

```
[15:03:12] THEM: Do you support mTLS on the test environment?
💡 Yes, mTLS on both test and prod.
   - client cert signed by our CA, 1-year validity
   - test env: allow-list your IPs first
```

## Setup

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt   # Linux: .venv/bin/pip
```

Set at least one key: `GEMINI_API_KEY` (default, ~$0.1 per hour of call) or `OPENAI_API_KEY`.
With no key it falls back to a local Whisper model (`faster-whisper`, CPU).

## Use

1. Put your call prep (agenda, facts, numbers, likely questions) into `notes/`.
2. Run `claude` in this folder and type **start**. Join the call.
3. Hints appear in the terminal. Type **?** to get a suggestion on what to say next.
4. Type **stop** to end and get a summary plus a follow-up email draft.

Transcripts go to `calls/`. Transcriber alone, without Claude: `run.cmd` or `python earpiece.py --help`.

## Notes

- Headphones are recommended, so your mic doesn't pick up the other side.
- `--lang` sets the spoken language (default `en`), `--terms` passes names and jargon to improve recognition.
- `bench.py` compares engines on your own recordings in `samples/`.
- Call audio is sent to Google or OpenAI unless you use `--engine local`.
- If Claude Code stops the transcriber when RAM is low, start it with `CLAUDE_CODE_DISABLE_BG_SHELL_PRESSURE_REAP=1`.
