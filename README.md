# Patient Caller — voice bot for the PGAI agent test line

An automated caller that phones a medical-office voice agent, plays a realistic
patient with a goal, records both sides, and triages the transcripts for defects.

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # fill in Twilio + OpenAI credentials
```

`ffmpeg` is required by pydub: `brew install ffmpeg` / `apt install ffmpeg`.

Expose the local server so Twilio can reach it, and put the host (no scheme)
in `PUBLIC_HOST`:

```bash
ngrok http 8765
```

## Run

Two terminals.

```bash
# 1 — the bot server
python server.py

# 2 — place calls
python runner.py --scenario 01-new-appointment   # one
python runner.py --all                            # all ten
```

Then transcribe and triage:

```bash
python analysis/analyze.py --all
```

Artifacts land in `calls/`:

| file | what |
|---|---|
| `<scenario>-<callsid>.mp3` | dual-channel recording |
| `<scenario>-<callsid>.txt` | timestamped, speaker-labelled transcript |
| `<scenario>-<callsid>.segments.json` | segments with start/end times |
| `<scenario>-<callsid>.bugs.json` | candidate issues for review |

## Layout

```
scenarios/scenarios.py   goal, persona, steering, exit condition per call
bot/pipeline.py          Pipecat pipeline — Twilio <-> OpenAI Realtime
server.py                TwiML endpoint + media-stream websocket
runner.py                places outbound calls, downloads recordings
analysis/analyze.py      Whisper transcription + LLM bug triage
```

## Notes

- Every call goes to `TARGET_NUMBER` and nowhere else.
- All calls originate from a single `TWILIO_FROM_NUMBER`, as the brief requires.
- `bugs.json` is a *candidate* list. The submitted bug report is written by hand
  after listening to the audio.
- Tuning knob for conversational feel: `silence_duration_ms` in
  `bot/pipeline.py`. Raise it if the bot cuts the agent off, lower it if there
  are awkward gaps.
